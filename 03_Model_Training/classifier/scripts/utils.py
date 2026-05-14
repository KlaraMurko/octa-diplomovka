"""
utils.py
--------
Helper funkcie: timestamps, ukladanie JSON/CSV, folder creation,
class weights, cosine LR.
"""

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ── Timestamps ────────────────────────────────────────────────────────────────

def now_str() -> str:
    """Vráti aktuálny čas ako reťazec: 20260419_144221"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Folder creation ───────────────────────────────────────────────────────────

def make_run_dir(base_dir: Path, prefix: str = "run") -> tuple:
    """
    Vytvorí ďalší inkrementálny run priečinok v base_dir.
    Vracia (run_dir, run_name), napr. (Path(".../run_042"), "run_042").
    """
    import re
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    nums = [
        int(m.group(1))
        for p in base_dir.iterdir()
        if p.is_dir() and (m := pattern.match(p.name))
    ]
    next_num  = max(nums) + 1 if nums else 1
    run_name  = f"{prefix}_{next_num:03d}"
    run_dir   = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


# ── JSON / CSV helpers ────────────────────────────────────────────────────────

def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def append_to_summary_csv(row: dict, csv_path: Path) -> None:
    """Priebežne aktualizuje summary.csv — pridá riadok."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = pd.concat([df, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_csv(csv_path, index=False)


def load_summary_csv(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


# ── Serialization helpers ─────────────────────────────────────────────────────

def safe_to_string(v) -> str:
    """Prevedie hodnotu na string vhodný pre CSV."""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def normalize_for_compare(v) -> str:
    """Zjednotí reprezentáciu hodnôt pri porovnávaní s CSV."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return str(float(v))
    return str(v)


def already_done(overrides: dict, summary_csv: Path) -> bool:
    """
    Vráti True ak rovnaká kombinácia hyperparametrov
    už úspešne dobehla (status == "ok") v summary.csv.
    """
    df = load_summary_csv(summary_csv)
    if df.empty or "status" not in df.columns:
        return False
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return False
    compare_keys = list(overrides.keys())
    for _, row in ok.iterrows():
        if all(
            normalize_for_compare(row.get(k, "")) == normalize_for_compare(overrides[k])
            for k in compare_keys
        ):
            return True
    return False


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(
    dataset,
    label_names: List[str],
    cfg_weights,
) -> Optional[torch.Tensor]:
    """
    cfg_weights:
        "auto"  → 1/freq, normalizované
        "none"  → None
        dict    → {"AMD": 2.0, ...}
    """
    if cfg_weights == "none" or cfg_weights is None:
        return None

    counts = np.zeros(len(label_names))
    for s in dataset.samples:
        counts[s["label"]] += 1

    if cfg_weights == "auto":
        counts = np.maximum(counts, 1)
        w = 1.0 / counts
        w = w / w.sum() * len(label_names)
    elif isinstance(cfg_weights, dict):
        w = np.array([cfg_weights.get(l, 1.0) for l in label_names], dtype=float)
    else:
        return None

    logger.info(
        "Class weights: "
        + ", ".join(f"{l}={w[i]:.3f}" for i, l in enumerate(label_names))
    )
    return torch.tensor(w, dtype=torch.float32)


# ── Cosine LR ─────────────────────────────────────────────────────────────────

def cosine_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    min_lr: float,
    total_epochs: int,
    warmup_epochs: int,
    epoch: int,
) -> float:
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        t  = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def cosine_lr_value(
    base_lr: float,
    min_lr: float,
    total_epochs: int,
    warmup_epochs: int,
    epoch: int,
) -> float:
    """Vráti LR hodnotu bez zmeny optimizera."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    t = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))


# ── Timer ─────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed_minutes(self) -> float:
        return (time.time() - self._start) / 60.0

    def reset(self):
        self._start = time.time()
