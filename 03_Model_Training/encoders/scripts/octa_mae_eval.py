"""
octa_mae_eval.py
================
Evaluácia OCTA MAE encoderov — linear probe na embeddingoch.

Obsah:
  - ViTEncoder (eval verzia — jednoduchý forward, CLS token)
  - load_encoder
  - EvalDataset
  - load_or_extract_embeddings  (cache na disk ako parquet)
  - build_sample_level_concat
  - run_probe
  - run_encoder_eval_final

Použitie:
  import sys
  sys.path.insert(0, "scripts")
  from octa_mae_eval import run_encoder_eval_final
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOUR_DISEASES = ["AMD", "DR", "Healthy", "RVO"]

IMAGE_SIZE  = 224
PATCH_SIZE  = 16
IN_CHANS    = 1
ENC_EMBED   = 384
ENC_DEPTH   = 12
ENC_HEADS   = 6
ENC_MLP     = 4.0
BATCH_SIZE  = 64
NUM_WORKERS = 4


# ══════════════════════════════════════════════════════════════════════════════
# POMOCNÉ FUNKCIE — cesty
# ══════════════════════════════════════════════════════════════════════════════

def _derive_eval_dir(encoder_path) -> Path:
    """
    Odvodí eval priečinok vedľa encodera.
    results/phase1/baseline/<run_id>/final_encoder.pth
    -> results/phase1/baseline/<run_id>/eval/
    """
    return Path(encoder_path).parent / "eval"


def _derive_eval_name(encoder_path) -> str:
    """
    Odvodí názov súborov z cesty encodera.
    results/phase1/baseline/<run_id>/final_encoder.pth -> phase1_baseline
    """
    parts = Path(encoder_path).parts
    for i, part in enumerate(parts):
        if part in ("phase1", "phase2") and i + 1 < len(parts):
            return f"{part}_{parts[i + 1]}"
    return Path(encoder_path).parent.parent.name


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITEKTÚRA — identická s eval notebookom, nemeníme
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv  = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out  = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, num_modalities=2):
        super().__init__()
        self.patch_embed    = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        N                   = self.patch_embed.num_patches
        self.cls_token      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed      = nn.Parameter(torch.zeros(1, N + 1, embed_dim))
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)
        self.blocks         = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio)
                                             for _ in range(depth)])
        self.norm           = nn.LayerNorm(embed_dim)

    def forward(self, x, layer_id):
        B   = x.shape[0]
        mod = self.modality_embed(layer_id).unsqueeze(1)
        tok = self.patch_embed(x) + self.pos_embed[:, 1:, :] + mod
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tok = torch.cat([cls, tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)[:, 0, :]


# ══════════════════════════════════════════════════════════════════════════════
# NAČÍTANIE ENCODERA
# ══════════════════════════════════════════════════════════════════════════════

def _remap_mlp_keys(state_dict):
    new = {}
    for k, v in state_dict.items():
        k = k.replace(".mlp.fc1.", ".mlp.0.")
        k = k.replace(".mlp.fc2.", ".mlp.2.")
        new[k] = v
    return new


def load_encoder(encoder_path: str) -> ViTEncoder:
    encoder = ViTEncoder(
        img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS,
        embed_dim=ENC_EMBED, depth=ENC_DEPTH, num_heads=ENC_HEADS, mlp_ratio=ENC_MLP,
    )
    state = torch.load(encoder_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ["encoder", "model", "state_dict"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {k.replace("module.", ""): v for k, v in state.items()}
    state = _remap_mlp_keys(state)
    incomp = encoder.load_state_dict(state, strict=False)
    print(f"Encoder nacitany: {encoder_path}")
    print(f"  Missing keys   : {len(incomp.missing_keys)}")
    print(f"  Unexpected keys: {len(incomp.unexpected_keys)}")
    encoder.eval().to(device)
    return encoder


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _build_eval_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class EvalDataset(Dataset):
    """
    Flat list: kazdy riadok excelu -> SVP snimka + (ak has_dcp >= 1) DCP snimka.
    Berie vsetky split_mae hodnoty ktore su v eval_splits.
    """
    def __init__(self, df: pd.DataFrame, data_root: str, eval_splits=("train", "test")):
        self.data_root = Path(data_root)
        self.transform = _build_eval_transform()
        rows = []
        df_filtered = df[df["split_mae"].str.lower().isin([s.lower() for s in eval_splits])]
        for _, row in df_filtered.iterrows():
            base = dict(
                label     = str(row.get("label_clean_5class", "UNK")),
                dataset   = str(row.get("dataset", "UNK")),
                split     = str(row.get("split_mae", "")).lower().strip(),
                sample_id = str(row.get("sample_id", "")),
            )
            svp = row.get("svp_path", None)
            if pd.notna(svp) and str(svp).strip():
                rows.append({**base, "img_path": str(svp), "layer_id": 0, "modality": "SVP"})

            has_dcp = int(row.get("has_dcp", 0)) if pd.notna(row.get("has_dcp", 0)) else 0
            dcp = row.get("dcp_path", None)
            if has_dcp >= 1 and pd.notna(dcp) and str(dcp).strip():
                rows.append({**base, "img_path": str(dcp), "layer_id": 1, "modality": "DCP"})

        self.rows = rows
        logger.info(f"EvalDataset: {len(rows)} snimok")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        try:
            img = Image.open(self.data_root / r["img_path"]).convert("L")
            t   = self.transform(img).float()
        except Exception as e:
            logger.warning(f"Chyba pri nacitani {r['img_path']}: {e}")
            t = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        return t, torch.tensor(r["layer_id"], dtype=torch.long), idx


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAKCIA EMBEDDINGOV + CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _extract_embeddings(encoder: ViTEncoder, dataset: EvalDataset) -> pd.DataFrame:
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    all_embs, all_idxs = [], []
    encoder.eval()
    with torch.no_grad():
        for imgs, layer_ids, idxs in loader:
            imgs      = imgs.to(device).float()
            layer_ids = layer_ids.to(device)
            embs      = encoder(imgs, layer_ids)
            all_embs.append(embs.cpu().numpy().astype(np.float32))
            all_idxs.extend(idxs.tolist())
            n_done = len(all_idxs)
            if n_done % 500 == 0 or n_done == len(dataset):
                print(f"  {n_done}/{len(dataset)} snimok spracovanych")

    embs_np = np.concatenate(all_embs, axis=0)
    meta    = pd.DataFrame([dataset.rows[i] for i in all_idxs])
    meta["emb"] = [row.tolist() for row in embs_np]
    logger.info(f"Extrahovanych {len(meta)} embeddingov")
    return meta


def load_or_extract_embeddings(
    encoder_path: str,
    excel_path: str,
    data_root: str,
    eval_splits: tuple = ("train", "test"),
) -> pd.DataFrame:
    """
    Skontroluje ci uz existuju ulozene embeddingy vedla encodera v eval/.
    Ak ano — nacita z disku.
    Ak nie — extrahuje cez encoder a ulozi.

    Subor sa ulozi sem:
      results/phase1/baseline/<run_id>/eval/phase1_baseline_embeddings.parquet
    """
    eval_dir  = _derive_eval_dir(encoder_path)
    eval_name = _derive_eval_name(encoder_path)
    out_path  = eval_dir / f"{eval_name}_embeddings.parquet"

    if out_path.exists():
        print(f"Nacitavam ulozene embeddingy: {out_path}")
        emb_df = pd.read_parquet(out_path)
        emb_df["emb"] = emb_df["emb"].apply(
            lambda x: x if isinstance(x, list) else x.tolist()
        )
        print(f"  Nacitanych {len(emb_df)} embeddingov")
        return emb_df

    print("Embeddingy nenajdene — spustam extrakciu...")
    eval_dir.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder(encoder_path)
    df      = pd.read_excel(excel_path)
    dataset = EvalDataset(df, data_root, eval_splits=eval_splits)
    emb_df  = _extract_embeddings(encoder, dataset)

    emb_df.to_parquet(out_path, index=False)
    print(f"Embeddingy ulozene: {out_path}")

    return emb_df


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE-LEVEL CONCAT
# ══════════════════════════════════════════════════════════════════════════════

def build_sample_level_concat(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Jeden riadok = jeden pacient/meranie.
    SVP embedding + DCP embedding (alebo nuly ak chyba) + dcp_present flag.
    """
    svp = emb_df[emb_df["modality"] == "SVP"].copy()
    dcp = emb_df[emb_df["modality"] == "DCP"].copy()

    if len(svp) == 0:
        raise ValueError("Ziadne SVP embeddingy v emb_df.")

    sample_emb = svp.iloc[0]["emb"]
    emb_dim = len(sample_emb) if isinstance(sample_emb, list) else sample_emb.shape[0]

    merged = svp.merge(
        dcp[["sample_id", "split", "emb"]],
        on=["sample_id", "split"],
        how="left",
        suffixes=("_svp", "_dcp"),
    )
    merged["dcp_present"] = merged["emb_dcp"].notna()

    def _to_np(val, dim):
        if isinstance(val, list):
            return np.array(val, dtype=np.float32)
        if isinstance(val, np.ndarray):
            return val.astype(np.float32)
        return np.zeros(dim, dtype=np.float32)

    X_svp = np.stack(merged["emb_svp"].apply(lambda x: _to_np(x, emb_dim)).values)
    X_dcp = np.stack(merged["emb_dcp"].apply(lambda x: _to_np(x, emb_dim)).values)
    flag  = merged["dcp_present"].astype(np.float32).values.reshape(-1, 1)
    X     = np.hstack([X_svp, X_dcp, flag])

    out = merged[["sample_id", "split", "dataset", "label", "dcp_present"]].copy()
    out["emb"]      = [row.tolist() for row in X]
    out["modality"] = np.where(out["dcp_present"], "SVP+DCP", "SVP-only")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# LINEAR PROBE
# ══════════════════════════════════════════════════════════════════════════════

def run_probe(
    emb_df: pd.DataFrame,
    target_col: str,
    title: str,
    filter_labels: list = None,
    C: float = 1.0,
) -> dict:
    """
    Trenuje linear probe na split=train, testuje na split=test.
    Vrati bal_acc a vypise classification report.
    """
    df = emb_df.copy()
    if filter_labels:
        df = df[df[target_col].isin(filter_labels)]

    train_labels = set(df[df["split"] == "train"][target_col].unique())
    test_labels  = set(df[df["split"] == "test"][target_col].unique())
    valid_labels = train_labels & test_labels
    df = df[df[target_col].isin(valid_labels)]

    X     = np.stack(df["emb"].apply(lambda x: np.array(x, dtype=np.float32)).values)
    y_raw = df[target_col].astype(str).values
    split = df["split"].astype(str).str.lower().str.strip().values

    tr_mask   = split == "train"
    test_mask = split == "test"

    le      = LabelEncoder()
    y       = le.fit_transform(y_raw)
    scaler  = StandardScaler()
    X_tr    = scaler.fit_transform(X[tr_mask])
    X_te    = scaler.transform(X[test_mask])

    clf = LogisticRegression(
        C=C, max_iter=2000, random_state=42,
        class_weight="balanced", solver="lbfgs",
    )
    clf.fit(X_tr, y[tr_mask])
    y_pred = clf.predict(X_te)
    y_true = y[test_mask]

    bacc = balanced_accuracy_score(y_true, y_pred)

    sep = "=" * 62
    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"Train vzoriek: {tr_mask.sum()} | Test vzoriek: {test_mask.sum()}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

    return {"bal_acc": float(bacc), "clf": clf, "le": le, "scaler": scaler}


# ══════════════════════════════════════════════════════════════════════════════
# HLAVNA EVAL FUNKCIA
# ══════════════════════════════════════════════════════════════════════════════

def run_encoder_eval_final(
    encoder_path: str,
    excel_path: str,
    data_root: str,
    eval_splits: tuple = ("train", "test"),
) -> dict:
    """
    Hlavna eval funkcia.

    Cache logika:
      - ak existuju embeddingy aj vysledky -> len vypise summary, nevypocitava znova
      - ak existuju embeddingy ale nie vysledky -> nacita embeddingy, spusti proby
      - ak neexistuje nic -> extrahuje, spusti proby, ulozi oboje

    Subory sa ulozia sem:
      results/phase1/baseline/<run_id>/eval/phase1_baseline_embeddings.parquet
      results/phase1/baseline/<run_id>/eval/phase1_baseline_results.json
    """
    eval_dir  = _derive_eval_dir(encoder_path)
    eval_name = _derive_eval_name(encoder_path)
    results_path = eval_dir / f"{eval_name}_results.json"

    print("\n========== ENCODER EVAL ==========")
    print(f"Encoder   : {encoder_path}")
    print(f"Eval dir  : {eval_dir}")
    print(f"Eval name : {eval_name}\n")

    # ak mame uz vysledky, len ich nacitame a vypiseme
    if results_path.exists():
        print(f"Vysledky najdene: {results_path}")
        with open(results_path, "r") as f:
            saved = json.load(f)
        print("\n===== SUMMARY (z cache) =====")
        for k, v in saved.items():
            print(f"{k}: {v:.4f}")
        return {"results": {k: {"bal_acc": v} for k, v in saved.items()}}

    # embeddingy
    emb_df = load_or_extract_embeddings(
        encoder_path=encoder_path,
        excel_path=excel_path,
        data_root=data_root,
        eval_splits=eval_splits,
    )

    results = {}

    # SVP only
    svp_df = emb_df[emb_df["modality"] == "SVP"].copy()
    results["SVP only - Dataset"] = run_probe(
        svp_df, target_col="dataset",
        title="SVP only - Dataset classification",
    )
    results["SVP only - Disease"] = run_probe(
        svp_df, target_col="label",
        title="SVP only - Disease classification",
        filter_labels=FOUR_DISEASES,
    )

    # DCP only
    dcp_df = emb_df[emb_df["modality"] == "DCP"].copy()
    if len(dcp_df) > 0:
        results["DCP only - Dataset"] = run_probe(
            dcp_df, target_col="dataset",
            title="DCP only - Dataset classification",
        )
        dcp_dis = dcp_df[dcp_df["label"].isin(FOUR_DISEASES)].copy()
        if len(dcp_dis) > 0:
            results["DCP only - Disease"] = run_probe(
                dcp_dis, target_col="label",
                title="DCP only - Disease classification",
                filter_labels=FOUR_DISEASES,
            )
    else:
        print("Ziadne DCP embeddingy — DCP probe preskocene.")

    # sample-level concat
    sample_df = build_sample_level_concat(emb_df)
    results["Concat - Dataset"] = run_probe(
        sample_df, target_col="dataset",
        title="Sample-level concat - Dataset classification",
    )
    results["Concat - Disease"] = run_probe(
        sample_df[sample_df["label"].isin(FOUR_DISEASES)].copy(),
        target_col="label",
        title="Sample-level concat - Disease classification",
        filter_labels=FOUR_DISEASES,
    )

    # SVP vs DCP
    results["Single - SVP vs DCP"] = run_probe(
        emb_df, target_col="modality",
        title="Single-image - SVP vs DCP",
    )

    # summary + ulozenie vysledkov
    print("\n===== SUMMARY =====")
    summary = {}
    for k, v in results.items():
        print(f"{k}: {v['bal_acc']:.4f}")
        summary[k] = v["bal_acc"]

    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nVysledky ulozene: {results_path}")

    return {
        "results":    results,
        "emb_single": emb_df,
        "emb_sample": sample_df,
    }