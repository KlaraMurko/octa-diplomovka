"""
dataset.py
----------
Dataset class, Excel loading, filtering, DataLoaders.

Experiment configs:
    EXPERIMENT_CONFIGS = {
        "full":             use_column="use_for_cls_full",        split_column="split_full"
        "svp_dcp":          use_column="use_for_cls_svp_dcp",     split_column="split_svp_dcp"
        "svp_dcp_ballanced":use_column="use_for_cls_svp_dcp_ballanced", split_column="split_svp_dcp_ballanced"
    }
"""

import random
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Label definitions ─────────────────────────────────────────────────────────
LABEL_NAMES = ["AMD", "DR", "Healthy", "RVO"]

# ── Experiment → Excel column mapping ────────────────────────────────────────
EXPERIMENT_CONFIGS = {
    "full": {
        "use_column":   "use_for_cls_full",
        "split_column": "split_full",
    },
    "svp_dcp": {
        "use_column":   "use_for_cls_svp_dcp",
        "split_column": "split_svp_dcp",
    },
    "svp_dcp_ballanced": {
        "use_column":   "use_for_cls_svp_dcp_ballanced",
        "split_column": "split_svp_dcp_ballanced",
    },
}


# ── Augmentations ─────────────────────────────────────────────────────────────

def build_augmentations(image_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.85, 1.0), interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


# ── Image loading ─────────────────────────────────────────────────────────────

def safe_load_image(
    path: str, data_root: Path, transform
) -> Optional[torch.Tensor]:
    try:
        full = data_root / path
        if not full.exists():
            return None
        img = Image.open(full).convert("L")
        return transform(img)
    except Exception:
        return None


# ── Excel loading & filtering ─────────────────────────────────────────────────

def load_dataframes(
    excel_path: Path,
    experiment_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Načíta master Excel a vráti (train_df, val_df, test_df)
    podľa experiment_name.

    Filtrovanie:
        df[use_column] == 1  →  použité snímky
        df[split_column] == "train" / "val" / "test"
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Neznámy experiment: '{experiment_name}'. "
            f"Možnosti: {list(EXPERIMENT_CONFIGS.keys())}"
        )

    cfg = EXPERIMENT_CONFIGS[experiment_name]
    use_col   = cfg["use_column"]
    split_col = cfg["split_column"]

    df = pd.read_excel(excel_path)

    # Skontroluj že stĺpce existujú
    for col in (use_col, split_col):
        if col not in df.columns:
            raise ValueError(
                f"Stĺpec '{col}' sa nenašiel v Exceli. "
                f"Dostupné stĺpce: {list(df.columns)}"
            )

    base = df[df[use_col] == 1].copy()

    train_df = base[base[split_col] == "train"].copy()
    val_df   = base[base[split_col] == "val"].copy()
    test_df  = base[base[split_col] == "test"].copy()

    logger.info(
        f"Experiment '{experiment_name}' | "
        f"use_col={use_col} | split_col={split_col} | "
        f"train={len(train_df)} | val={len(val_df)} | test={len(test_df)}"
    )

    return train_df, val_df, test_df


# ── Dataset ───────────────────────────────────────────────────────────────────

class OctaClsDataset(Dataset):
    """
    Každý sample = jeden riadok z Excelu.
    SVP je vždy prítomný. DCP je načítaný ak has_dcp == 1.
    Počas tréningu: modality_dropout_prob → náhodne ignoruje DCP.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        label_names: List[str] = LABEL_NAMES,
        is_train: bool = True,
        modality_dropout_prob: float = 0.0,
    ):
        self.data_root   = Path(data_root)
        self.label_names = label_names
        self.label_to_idx = {l: i for i, l in enumerate(label_names)}
        self.transform   = build_augmentations(224, is_train)
        self.is_train    = is_train
        self.dropout_p   = modality_dropout_prob if is_train else 0.0

        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            label = str(row["label_clean_5class"]).strip()
            if label not in self.label_to_idx:
                skipped += 1
                continue
            self.samples.append({
                "svp_path"  : str(row["svp_path"]),
                "dcp_path"  : str(row.get("dcp_path", "")),
                "has_dcp"   : int(row.get("has_dcp", 0)) == 1,
                "label"     : self.label_to_idx[label],
                "patient_id": str(row.get("patient_id", "")),
            })

        logger.info(
            f"OctaClsDataset: {len(self.samples)} samples "
            f"(skipped={skipped}, is_train={is_train})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        svp = safe_load_image(s["svp_path"], self.data_root, self.transform)
        if svp is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        has_dcp = s["has_dcp"]

        # Modality dropout
        if has_dcp and self.dropout_p > 0 and random.random() < self.dropout_p:
            has_dcp = False

        if has_dcp and s["dcp_path"]:
            dcp = safe_load_image(s["dcp_path"], self.data_root, self.transform)
            if dcp is None:
                has_dcp = False

        if not has_dcp:
            dcp = torch.zeros_like(svp)

        return (
            svp,
            dcp,
            torch.tensor(int(has_dcp), dtype=torch.long),
            torch.tensor(s["label"],   dtype=torch.long),
        )


# ── Patient-Aware Batch Sampler ───────────────────────────────────────────────

class PatientAwareBatchSampler(Sampler):
    """
    Balanced sampler:
    - mixuje triedy v rámci batchu
    - jeden patient_id sa nenachádza v tom istom batchi viac ako raz
    """

    def __init__(
        self,
        dataset: OctaClsDataset,
        batch_size: int,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.seed       = seed
        self.drop_last  = drop_last

        self.label_to_idx = defaultdict(list)
        self.patient_to_idx = defaultdict(list)

        for idx, s in enumerate(dataset.samples):
            self.label_to_idx[s["label"]].append(idx)
            self.patient_to_idx[s["patient_id"]].append(idx)

        self.labels      = sorted(self.label_to_idx.keys())
        self.num_samples = len(dataset)
        self.num_batches = (
            self.num_samples // batch_size
            if drop_last
            else math.ceil(self.num_samples / batch_size)
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed)

        pools = {}
        orig  = {}
        for lbl, idxs in self.label_to_idx.items():
            c = idxs.copy()
            rng.shuffle(c)
            pools[lbl] = c
            orig[lbl]  = idxs.copy()

        label_cycle = self.labels.copy()
        rng.shuffle(label_cycle)
        label_ptr = 0

        for _ in range(self.num_batches):
            batch     = []
            seen_pids = set()
            n_labels  = min(len(self.labels), max(2, self.batch_size // 8))
            chosen_lbls = [
                label_cycle[(label_ptr + i) % len(label_cycle)]
                for i in range(n_labels)
            ]
            label_ptr += n_labels

            attempts = 0
            while len(batch) < self.batch_size and attempts < self.batch_size * 4:
                attempts += 1
                progressed = False
                for lbl in chosen_lbls:
                    if len(batch) >= self.batch_size:
                        break
                    if not pools[lbl]:
                        r = orig[lbl].copy()
                        rng.shuffle(r)
                        pools[lbl] = r
                    for _ in range(len(pools[lbl])):
                        if not pools[lbl]:
                            break
                        candidate = pools[lbl][-1]
                        pid = self.dataset.samples[candidate]["patient_id"]
                        if pid not in seen_pids:
                            pools[lbl].pop()
                            batch.append(candidate)
                            seen_pids.add(pid)
                            progressed = True
                            break
                        else:
                            pools[lbl].insert(0, pools[lbl].pop())
                            break
                if not progressed:
                    break

            # Doplň ak treba
            if len(batch) < self.batch_size:
                all_idxs = list(range(len(self.dataset)))
                rng.shuffle(all_idxs)
                for idx in all_idxs:
                    if len(batch) >= self.batch_size:
                        break
                    batch.append(idx)

            if len(batch) == self.batch_size or (not self.drop_last and batch):
                yield batch[: self.batch_size]


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_root: Path,
    batch_size: int = 32,
    modality_dropout_prob: float = 0.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    label_names: List[str] = LABEL_NAMES,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = OctaClsDataset(
        train_df, data_root, label_names,
        is_train=True, modality_dropout_prob=modality_dropout_prob,
    )
    val_ds = OctaClsDataset(
        val_df, data_root, label_names, is_train=False,
    )
    test_ds = OctaClsDataset(
        test_df, data_root, label_names, is_train=False,
    )

    train_sampler = PatientAwareBatchSampler(
        train_ds, batch_size=batch_size, seed=seed, drop_last=True
    )

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
