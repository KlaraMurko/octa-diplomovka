"""
training.py
-----------
Tréningová slučka, validácia, grid search, ukladanie/načítanie modelov,
training curves.
"""

import copy
import itertools
import logging
import math
import random
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    LABEL_NAMES,
    build_dataloaders,
    load_dataframes,
)
from .model import FUSION_TYPES, OctaClassifier, build_model
from .utils import (
    Timer,
    already_done,
    append_to_summary_csv,
    compute_class_weights,
    cosine_lr,
    cosine_lr_value,
    make_run_dir,
    now_str,
    safe_to_string,
    save_json,
)

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Paths
    encoder_path : str  = ""
    experiment_name: str = "full"   # "full" | "svp_dcp" | "svp_dcp_ballanced"

    # Model
    fusion_type  : str        = "transformer"
    encoder_mode : str        = "mean_patch"
    hidden_dims  : List[int]  = field(default_factory=lambda: [512])
    dropout      : float      = 0.4
    use_bn       : bool       = True
    num_classes  : int        = 4

    # Transformer fusion
    tf_num_heads  : int   = 4
    tf_num_layers : int   = 1
    tf_dropout    : float = 0.1

    # Training
    epochs          : int   = 100
    batch_size      : int   = 32
    lr              : float = 1e-4
    weight_decay    : float = 1e-4
    warmup_epochs   : int   = 10
    min_lr          : float = 1e-6
    grad_clip       : float = 1.0
    use_amp         : bool  = True
    label_smoothing : float = 0.0
    class_weights   : object = "auto"
    modality_dropout_prob: float = 0.0

    # Encoder finetuning
    unfreeze_last_n_blocks: int   = 4
    encoder_lr_multiplier : float = 0.05

    # Misc
    seed        : int  = 42
    num_workers : int  = 4
    pin_memory  : bool = True


# ── Single training run ───────────────────────────────────────────────────────

def train_one_run(
    cfg: TrainConfig,
    excel_path: Path,
    data_root: Path,
    run_dir: Path,
    device: torch.device,
    label_names: List[str] = LABEL_NAMES,
) -> Dict:
    """
    Spustí jeden tréning podľa cfg.
    Uloží model.pth, config.json, metrics.csv do run_dir.
    Vracia dict s metrikami.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    timer = Timer()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_dataframes(excel_path, cfg.experiment_name)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        data_root=data_root,
        batch_size=cfg.batch_size,
        modality_dropout_prob=cfg.modality_dropout_prob,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        seed=cfg.seed,
        label_names=label_names,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model, enc_params = build_model(
        encoder_path=cfg.encoder_path,
        fusion_type=cfg.fusion_type,
        encoder_mode=cfg.encoder_mode,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
        use_bn=cfg.use_bn,
        tf_num_heads=cfg.tf_num_heads,
        tf_num_layers=cfg.tf_num_layers,
        tf_dropout=cfg.tf_dropout,
        unfreeze_last_n_blocks=cfg.unfreeze_last_n_blocks,
        device=device,
    )

    # ── Class weights & loss ──────────────────────────────────────────────────
    cw = compute_class_weights(
        train_loader.dataset, label_names, cfg.class_weights
    )
    cw_device = cw.to(device) if cw is not None else None
    criterion = nn.CrossEntropyLoss(
        weight=cw_device, label_smoothing=cfg.label_smoothing
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    cls_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("encoder.")
    ]
    param_groups = [{"params": cls_params, "lr": cfg.lr, "name": "classifier"}]
    if enc_params:
        enc_lr = cfg.lr * cfg.encoder_lr_multiplier
        param_groups.append({"params": enc_params, "lr": enc_lr, "name": "encoder"})
        logger.info(
            f"Optimizer: cls_lr={cfg.lr:.2e} | enc_lr={enc_lr:.2e}"
        )
    else:
        logger.info(f"Optimizer: cls_lr={cfg.lr:.2e} | encoder=frozen")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scaler    = torch.cuda.amp.GradScaler(
        enabled=cfg.use_amp and torch.cuda.is_available()
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_ba = 0.0
    best_epoch  = 0
    best_ckpt   = run_dir / "model.pth"
    history     = []

    for epoch in range(cfg.epochs):
        model.train()

        # Zmrazené časti enkódera musia ostať v eval móde
        if cfg.unfreeze_last_n_blocks < 12:
            model.encoder.eval()
            total_blocks = len(model.encoder.blocks)
            start_block  = max(0, total_blocks - cfg.unfreeze_last_n_blocks)
            for i in range(start_block, total_blocks):
                model.encoder.blocks[i].train()
            if cfg.unfreeze_last_n_blocks > 0:
                model.encoder.norm.train()

        # LR update
        lr = cosine_lr(
            optimizer, cfg.lr, cfg.min_lr, cfg.epochs, cfg.warmup_epochs, epoch
        )
        if enc_params:
            enc_lr      = cfg.lr * cfg.encoder_lr_multiplier
            enc_lr_now  = cosine_lr_value(
                enc_lr, cfg.min_lr * cfg.encoder_lr_multiplier,
                cfg.epochs, cfg.warmup_epochs, epoch
            )
            optimizer.param_groups[-1]["lr"] = enc_lr_now

        # Train step
        tr_loss, n = 0.0, 0
        for svp, dcp, has_dcp, labels in train_loader:
            svp     = svp.to(device).float()
            dcp     = dcp.to(device).float()
            has_dcp = has_dcp.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=cfg.use_amp and torch.cuda.is_available()
            ):
                logits = model(svp, dcp, has_dcp)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()
            n += 1

        avg_tr_loss = tr_loss / max(n, 1)

        # Validation
        from .evaluation import evaluate
        val_res = evaluate(model, val_loader, label_names, device)
        val_ba  = val_res["balanced_accuracy"]
        val_f1  = val_res["macro_f1"]
        val_acc = val_res["accuracy"]

        history.append({
            "epoch"      : epoch + 1,
            "lr"         : lr,
            "train_loss" : avg_tr_loss,
            "val_ba"     : val_ba,
            "val_f1"     : val_f1,
            "val_acc"    : val_acc,
        })

        logger.info(
            f"Ep {epoch+1:>4}/{cfg.epochs} | lr={lr:.2e} | "
            f"loss={avg_tr_loss:.4f} | val_BA={val_ba:.4f} | "
            f"val_F1={val_f1:.4f} | {timer.elapsed_minutes():.1f}m"
        )

        if val_ba > best_val_ba:
            best_val_ba = val_ba
            best_epoch  = epoch + 1
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"  ★ Nový best val BA: {best_val_ba:.4f} (ep {best_epoch})")

    # ── Test ──────────────────────────────────────────────────────────────────
    logger.info(f"Načítavam best model z epochy {best_epoch}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    from .evaluation import evaluate, compute_per_class_recall, compute_min_recall
    val_res_final = evaluate(model, val_loader,  label_names, device)
    test_res      = evaluate(model, test_loader, label_names, device)

    training_time = timer.elapsed_minutes()

    # ── Save config & metrics ─────────────────────────────────────────────────
    config_dict = {k: safe_to_string(v) for k, v in cfg.__dict__.items()}
    save_json(config_dict, run_dir / "config.json")

    pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)

    results = {
        "best_epoch"      : best_epoch,
        "val_ba"          : round(val_res_final["balanced_accuracy"], 4),
        "test_ba"         : round(test_res["balanced_accuracy"], 4),
        "test_f1"         : round(test_res["macro_f1"], 4),
        "min_recall"      : round(compute_min_recall(test_res, label_names), 4),
        "training_time_minutes": round(training_time, 2),
        "history"         : history,
        "val_res"         : val_res_final,
        "test_res"        : test_res,
        **compute_per_class_recall(test_res, label_names),
    }

    logger.info(
        f"\n{'='*60}\n"
        f"best_epoch={best_epoch} | val_BA={results['val_ba']:.4f} | "
        f"test_BA={results['test_ba']:.4f} | test_F1={results['test_f1']:.4f}\n"
        f"{'='*60}"
    )

    return results


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid_search(
    experiment_name: str,
    grid_config: dict,
    encoder_path,
    excel_path: Path,
    data_root: Path,
    results_root: Path,
    n_runs: int,
    base_cfg: Optional[TrainConfig] = None,
    device: torch.device = None,
    seed: int = 42,
    label_names: List[str] = LABEL_NAMES,
) -> None:
    """
    Spustí grid search pre daný experiment.

    n_runs sa rozdelí rovnomerne medzi všetky fusion types:
        runs_per_fusion = n_runs // len(FUSION_TYPES)

    Výsledky sa ukladajú do:
        results_root / experiment_name / run_XXX / ...
        results_root / experiment_name / summary.csv
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if base_cfg is None:
        base_cfg = TrainConfig()

    exp_dir     = results_root / experiment_name
    summary_csv = exp_dir / "summary.csv"
    exp_dir.mkdir(parents=True, exist_ok=True)

    runs_per_fusion = max(1, n_runs // len(FUSION_TYPES))

    # Všetky kombinácie z grid_config (bez fusion_type)
    grid_keys   = list(grid_config.keys())
    grid_values = list(grid_config.values())
    all_combos  = [
        dict(zip(grid_keys, c))
        for c in itertools.product(*grid_values)
    ]

    rng = np.random.default_rng(seed)

    # Pre každú fusion type vyber runs_per_fusion nových kombinácií
    planned_runs = []
    for fusion in FUSION_TYPES:
        new_combos = []
        for combo in all_combos:
            full_combo = dict(combo)
            full_combo["fusion_type"] = fusion
            if not already_done(full_combo, summary_csv):
                new_combos.append(full_combo)

        rng.shuffle(new_combos)
        selected = new_combos[: min(runs_per_fusion, len(new_combos))]
        planned_runs.extend(selected)

        logger.info(
            f"Fusion {fusion}: nových kombinácií={len(new_combos)} | "
            f"naplánovaných={len(selected)}"
        )

    rng.shuffle(planned_runs)
    total = len(planned_runs)

    print(f"\n{'='*70}")
    print(f"GRID SEARCH — experiment={experiment_name}")
    print(f"encoder_path    : {encoder_path}")
    print(f"n_runs          : {n_runs} ({runs_per_fusion} per fusion)")
    print(f"nových runov    : {total}")
    print(f"output          : {exp_dir}")
    print(f"{'='*70}\n")

    if total == 0:
        print("Nie je čo spúšťať — všetky kombinácie už existujú.")
        return

    for idx, combo in enumerate(planned_runs, 1):
        run_dir, run_name = make_run_dir(exp_dir)

        # Zostav config pre tento run
        run_cfg = copy.deepcopy(base_cfg)
        run_cfg.encoder_path    = str(encoder_path)
        run_cfg.experiment_name = experiment_name

        for key, val in combo.items():
            if hasattr(run_cfg, key):
                setattr(run_cfg, key, val)

        fusion = combo.get("fusion_type", "?")
        print(
            f"\n[{idx}/{total}] {run_name} | fusion={fusion} | "
            + " | ".join(
                f"{k}={v}" for k, v in combo.items() if k != "fusion_type"
            )
        )

        try:
            results = train_one_run(
                cfg=run_cfg,
                excel_path=excel_path,
                data_root=data_root,
                run_dir=run_dir,
                device=device,
                label_names=label_names,
            )

            row = {
                "run_name"    : run_name,
                "best_epoch"  : results["best_epoch"],
                "val_ba"      : results["val_ba"],
                "test_ba"     : results["test_ba"],
                "test_f1"     : results["test_f1"],
                "min_recall"  : results["min_recall"],
                "status"      : "ok",
                "encoder_path": str(encoder_path),
                "recall_AMD"  : results.get("recall_AMD", 0.0),
                "recall_DR"   : results.get("recall_DR", 0.0),
                "recall_Healthy": results.get("recall_Healthy", 0.0),
                "recall_RVO"  : results.get("recall_RVO", 0.0),
                "unfreeze_last_n_blocks": run_cfg.unfreeze_last_n_blocks,
                "lr"          : run_cfg.lr,
                "hidden_dims" : safe_to_string(run_cfg.hidden_dims),
                "dropout"     : run_cfg.dropout,
                "label_smoothing": run_cfg.label_smoothing,
                "encoder_mode": run_cfg.encoder_mode,
                "fusion_type" : run_cfg.fusion_type,
                "data_split_mode": experiment_name,
                "modality_dropout_prob": run_cfg.modality_dropout_prob,
                "timestamp"   : now_str(),
                "training_time_minutes": results["training_time_minutes"],
            }

            print(
                f"  ✅ val_BA={results['val_ba']:.4f} | "
                f"test_BA={results['test_ba']:.4f} | "
                f"test_F1={results['test_f1']:.4f} | "
                f"min_recall={results['min_recall']:.4f}"
            )

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            row = {
                "run_name"    : run_name,
                "status"      : f"error: {str(e)[:200]}",
                "encoder_path": str(encoder_path),
                "data_split_mode": experiment_name,
                "timestamp"   : now_str(),
                **{k: safe_to_string(v) for k, v in combo.items()},
            }

        append_to_summary_csv(row, summary_csv)

    print(f"\n{'='*70}")
    print(f"Grid search hotový. Summary: {summary_csv}")
    print(f"{'='*70}")


# ── Save / load best model ────────────────────────────────────────────────────

def save_best_model(
    model: OctaClassifier,
    cfg: TrainConfig,
    results: dict,
    best_model_dir: Path,
) -> None:
    """Uloží model, config a metrics do results/best_model/."""
    best_model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), best_model_dir / "best_model.pth")

    config_dict = {k: safe_to_string(v) for k, v in cfg.__dict__.items()}
    save_json(config_dict, best_model_dir / "config.json")

    if "history" in results:
        pd.DataFrame(results["history"]).to_csv(
            best_model_dir / "metrics.csv", index=False
        )

    logger.info(f"Best model uložený → {best_model_dir}")


def load_best_model(
    best_model_dir: Path,
    device: torch.device,
    encoder_path_override=None,
) -> Tuple[OctaClassifier, TrainConfig]:
    """Načíta model a config z results/best_model/."""
    from .utils import load_json
    import json

    config_path = best_model_dir / "config.json"
    model_path  = best_model_dir / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Config nenájdený: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model nenájdený: {model_path}")

    raw_cfg = load_json(config_path)
    cfg = TrainConfig()

    for key, val in raw_cfg.items():
        if not hasattr(cfg, key):
            continue
        # hidden_dims je uložený ako JSON string
        if key == "hidden_dims" and isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                pass
        if key in ("lr", "dropout", "label_smoothing", "modality_dropout_prob",
                   "min_lr", "weight_decay", "grad_clip", "encoder_lr_multiplier",
                   "tf_dropout"):
            val = float(val)
        elif key in ("epochs", "batch_size", "warmup_epochs", "num_classes",
                     "unfreeze_last_n_blocks", "tf_num_heads", "tf_num_layers",
                     "seed", "num_workers"):
            val = int(val)
        elif key in ("use_bn", "use_amp", "pin_memory"):
            val = str(val).lower() == "true"
        setattr(cfg, key, val)

    if encoder_path_override is not None:
        cfg.encoder_path = str(encoder_path_override)

    model, _ = build_model(
        encoder_path=cfg.encoder_path,
        fusion_type=cfg.fusion_type,
        encoder_mode=cfg.encoder_mode,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
        use_bn=cfg.use_bn,
        tf_num_heads=cfg.tf_num_heads,
        tf_num_layers=cfg.tf_num_layers,
        tf_dropout=cfg.tf_dropout,
        unfreeze_last_n_blocks=0,  # pri inference nie je potrebný finetuning setup
        device=device,
    )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    logger.info(f"Best model načítaný z {best_model_dir}")
    return model, cfg


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history: List[dict],
    save_dir: Path,
    show: bool = True,
) -> None:
    """
    Vykreslí a uloží training curves:
        - train loss
        - validation loss (ak existuje)
        - train accuracy (val_acc)
        - validation BA (val_ba)
    """
    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#4C72B0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy / BA
    ax = axes[1]
    if "val_acc" in df.columns:
        ax.plot(
            df["epoch"], df["val_acc"],
            label="Val Accuracy", color="#55A868", linestyle="--"
        )
    if "val_ba" in df.columns:
        ax.plot(
            df["epoch"], df["val_ba"],
            label="Val Balanced Accuracy", color="#C44E52"
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Training curves uložené → {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
