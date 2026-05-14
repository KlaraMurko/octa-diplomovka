"""
evaluation.py
-------------
Metriky, confusion matrix, classification report.
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .dataset import LABEL_NAMES

logger = logging.getLogger(__name__)


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(
    model,
    loader: DataLoader,
    label_names: List[str],
    device: torch.device,
) -> Dict:
    """
    Spustí inferenciu na celom loaderi a vráti metriky.

    Vracia dict:
        balanced_accuracy, macro_f1, accuracy,
        confusion_matrix, classification_report,
        preds, labels
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for svp, dcp, has_dcp, labels in loader:
            svp     = svp.to(device).float()
            dcp     = dcp.to(device).float()
            has_dcp = has_dcp.to(device)

            logits = model(svp, dcp, has_dcp)
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    ba  = balanced_accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = (all_preds == all_labels).mean()
    cm  = confusion_matrix(all_labels, all_preds, labels=list(range(len(label_names))))
    rep = classification_report(
        all_labels, all_preds, target_names=label_names, zero_division=0
    )

    return {
        "balanced_accuracy"     : ba,
        "macro_f1"              : f1,
        "accuracy"              : acc,
        "confusion_matrix"      : cm,
        "classification_report" : rep,
        "preds"                 : all_preds,
        "labels"                : all_labels,
    }


# ── Per-class recall helpers ──────────────────────────────────────────────────

def compute_per_class_recall(
    eval_res: Dict, label_names: List[str]
) -> Dict[str, float]:
    preds  = np.asarray(eval_res["preds"])
    labels = np.asarray(eval_res["labels"])
    result = {}
    for i, name in enumerate(label_names):
        mask = labels == i
        result[f"recall_{name}"] = (
            float((preds[mask] == i).mean()) if mask.sum() > 0 else 0.0
        )
    return result


def compute_min_recall(
    eval_res: Dict, label_names: List[str]
) -> float:
    per_class = compute_per_class_recall(eval_res, label_names)
    return float(min(per_class.values()))


# ── Print metrics ─────────────────────────────────────────────────────────────

def print_metrics(
    eval_res: Dict,
    label_names: List[str],
    split_name: str = "Test",
) -> None:
    print(f"\n{'='*60}")
    print(f"{split_name} Metrics")
    print(f"{'='*60}")
    print(f"Balanced Accuracy : {eval_res['balanced_accuracy']:.4f}")
    print(f"Macro F1          : {eval_res['macro_f1']:.4f}")
    print(f"Accuracy          : {eval_res['accuracy']:.4f}")
    print(f"\nPer-class Recall:")
    per_class = compute_per_class_recall(eval_res, label_names)
    for name, recall in per_class.items():
        print(f"  {name:20s}: {recall:.4f}")
    print(f"\nClassification Report:\n{eval_res['classification_report']}")
    print(f"{'='*60}")


# ── Confusion matrix plots ────────────────────────────────────────────────────

def plot_confusion_matrices(
    eval_res: Dict,
    label_names: List[str],
    save_dir: Path = None,
    show: bool = True,
) -> None:
    """
    Vykreslí confusion matrix a normalized confusion matrix (%)
    vedľa seba v jednom figure — seaborn štýl.
    """
    import seaborn as sns

    cm      = eval_res["confusion_matrix"]
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis].clip(min=1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0], linewidths=0.5,
        square=True,
        annot_kws={"size": 15},
    )
    axes[0].set_title("Konfúzna matica", fontsize=17, pad=14)
    axes[0].set_xlabel("Predikované triedy", fontsize=12)
    axes[0].set_ylabel("Správne triedy", fontsize=12)
    axes[0].tick_params(axis="both", labelsize=11)

    sns.heatmap(
        cm_norm, annot=True, fmt=".1f", cmap="Reds",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1], linewidths=0.5,
        square=True,
        annot_kws={"size": 15},
    )
    axes[1].set_title("Normalizovaná konfúzna matica (%)", fontsize=17, pad=14)
    axes[1].set_xlabel("Predikované triedy", fontsize=12)
    axes[1].set_ylabel("Správne triedy", fontsize=12)
    axes[1].tick_params(axis="both", labelsize=11)

    plt.tight_layout(pad=3.0)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "confusion_matrices.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrices uložené → {save_path}")

    if show:
        plt.show()
    else:
        plt.close()