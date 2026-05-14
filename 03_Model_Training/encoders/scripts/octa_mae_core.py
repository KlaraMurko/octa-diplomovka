"""
octa_mae_core.py
================
Zdieľané komponenty pre OCTA MAE pretraining.

Obsah:
  - Konfigurácie      (Phase1Config, Phase1BalancedConfig, Phase2Config)
  - Augmentácie + bezpečné načítanie obrázkov
  - Načítanie excelu  (load_mae_df)
  - Datasety          (Phase1Dataset, Phase1BalancedDataset, Phase2Dataset)
  - Sampler           (BalancedSVPDCPBatchSampler)
  - Modely            (PatchEmbed, Attention, MLP, Block, ViTEncoder,
                       MAEDecoder, MaskedAutoencoder, MultiMAE,
                       GradientReversal, DomainClassifier, MultiMAEWithGRL)
  - Pomocné funkcie   (cosine_lr, log_vram, save_checkpoint, save_encoder)
  - Tréningové funkcie (train_phase1, train_phase1_balanced,
                        train_phase2, train_phase2_grl)

Použitie v notebooku:
  import sys
  sys.path.append("scripts")
  from octa_mae_core import *
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import math
import time
import random
import logging
import datetime
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
BASE_DIR = Path(__file__).parent


try:
    import wandb
    WANDB_AVAILABLE = True
    print("wandb available")
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed — logging disabled")

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURÁCIE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Phase1Config:
    # ── Paths ──────────────────────────────────────────────
    excel_path     : str = str(BASE_DIR.parent / "data" / "master_excels" / "master_table.xlsx")
    data_root      : str = str(BASE_DIR.parent / "data")
    output_base    : str = str(BASE_DIR / "results" / "phase1" / "baseline")

    # ── Image ──────────────────────────────────────────────
    image_size     : int   = 224
    patch_size     : int   = 16
    in_chans       : int   = 1          # grayscale

    # ── MAE ────────────────────────────────────────────────
    mask_ratio     : float = 0.75

    # ── Encoder (ViT-Small) ────────────────────────────────
    enc_embed_dim  : int   = 384
    enc_depth      : int   = 12
    enc_num_heads  : int   = 6
    enc_mlp_ratio  : float = 4.0

    # ── Decoder ────────────────────────────────────────────
    dec_embed_dim  : int   = 256
    dec_depth      : int   = 4
    dec_num_heads  : int   = 8
    dec_mlp_ratio  : float = 4.0

    # ── Training ───────────────────────────────────────────
    epochs         : int   = 250
    batch_size     : int   = 64
    num_workers    : int   = 8
    lr             : float = 1.5e-4
    weight_decay   : float = 0.05
    warmup_epochs  : int   = 40
    min_lr         : float = 1e-6
    grad_clip      : float = 1.0
    use_amp        : bool  = True

    # ── Misc ───────────────────────────────────────────────
    seed           : int   = 42
    save_every     : int   = 25
    wandb_project  : str   = "octa-mae"
    wandb_entity   : Optional[str] = None
    wandb_run_name : Optional[str] = None
    pin_memory     : bool  = True
    drop_last      : bool  = True

    run_id: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


@dataclass
class Phase1BalancedConfig(Phase1Config):
    """Phase 1 s balanced 50/50 SVP:DCP batch samplingom."""
    output_base: str = str(BASE_DIR / "results" / "phase1" / "balanced")


@dataclass
class Phase2Config:
    # ── Paths ──────────────────────────────────────────────
    excel_path            : str  = str(BASE_DIR.parent / "data" / "master_excels" / "master_table.xlsx")
    data_root             : str  = str(BASE_DIR.parent / "data")
    output_base           : str   = str(BASE_DIR / "results" / "phase2" / "baseline")
    phase1_encoder_path   : Optional[str] = None   # cesta k Phase1 encoderu

    # ── Image ──────────────────────────────────────────────
    image_size     : int   = 224
    patch_size     : int   = 16
    in_chans       : int   = 1

    # ── MAE ────────────────────────────────────────────────
    mask_ratio_svp         : float = 0.75
    mask_ratio_dcp         : float = 0.75
    modality_dropout_prob  : float = 0.20

    # ── Encoder / Decoder ──────────────────────────────────
    enc_embed_dim  : int   = 384
    enc_depth      : int   = 12
    enc_num_heads  : int   = 6
    enc_mlp_ratio  : float = 4.0
    dec_embed_dim  : int   = 256
    dec_depth      : int   = 4
    dec_num_heads  : int   = 8
    dec_mlp_ratio  : float = 4.0

    # ── Loss ───────────────────────────────────────────────
    loss_weight_svp    : float = 1.0
    loss_weight_dcp    : float = 1.0

    # ── GRL / domain-adversarial ───────────────────────────
    domain_loss_weight : float = 0.10
    grl_max_lambda     : float = 0.10
    domain_hidden_dim  : int   = 256
    domain_dropout     : float = 0.20

    # ── Training ───────────────────────────────────────────
    epochs         : int   = 150
    batch_size     : int   = 32        # dva obrázky na vzorku → polovičný batch
    num_workers    : int   = 8
    lr             : float = 1.5e-4
    weight_decay   : float = 0.05
    warmup_epochs  : int   = 20
    min_lr         : float = 1e-6
    grad_clip      : float = 1.0
    use_amp        : bool  = True

    # ── Misc ───────────────────────────────────────────────
    seed           : int   = 42
    save_every     : int   = 25
    wandb_project  : str   = "octa-mae"
    wandb_entity   : Optional[str] = None
    wandb_run_name : Optional[str] = None
    pin_memory     : bool  = True
    drop_last      : bool  = True

    run_id: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTÁCIE + NAČÍTANIE OBRÁZKOV
# ══════════════════════════════════════════════════════════════════════════════

def build_augmentations(image_size: int, is_train: bool = True) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
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


def safe_load_image(path: str, data_root: str, transform) -> Optional[torch.Tensor]:
    """Načíta grayscale obrázok bezpečne. Vráti None pri akejkoľvek chybe."""
    try:
        full = Path(data_root) / path
        if not full.exists():
            logger.warning(f"Obrázok nenájdený, preskakujem: {full}")
            return None
        img = Image.open(full).convert("L")
        return transform(img)
    except Exception as e:
        logger.warning(f"Chyba pri načítaní {path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# NAČÍTANIE EXCELU
# ══════════════════════════════════════════════════════════════════════════════

def load_mae_df(excel_path: str) -> pd.DataFrame:
    """
    Načíta master Excel a filtruje na MAE-bezpečné riadky.

    Použité stĺpce:
      - use_for_mae : int  — filter, berieme len == 1
      - split_mae   : str  — 'train' / 'val' / 'test', vylučujeme 'test'
      - svp_path    : str  — relatívna cesta k SVP snímke (od data_root)
      - dcp_path    : str  — relatívna cesta k DCP snímke (od data_root)
      - has_dcp     : int  — 0=len SVP, 1=SVP+DCP, 2=len DCP
      - dataset     : str  — názov datasetu (pre GRL domain classifier)
    """
    df = pd.read_excel(excel_path)
    logger.info(f"Excel načítaný: {len(df)} riadkov | stĺpce: {list(df.columns)}")
    mae = df[(df['use_for_mae'] == 1) & (df['split_mae'].str.lower() != 'test')].copy()
    logger.info(
        f"MAE-eligible riadkov: {len(mae)} "
        f"(train: {(mae['split_mae'].str.lower() == 'train').sum()}, "
        f"val: {(mae['split_mae'].str.lower() == 'val').sum()})"
    )
    return mae


# ══════════════════════════════════════════════════════════════════════════════
# DATASETY
# ══════════════════════════════════════════════════════════════════════════════

class Phase1Dataset(Dataset):
    """
    Plochý zoznam jednotlivých OCTA snímok pre Phase 1.

    Logika výberu (use_for_mae==1, split_mae!=test):
      has_dcp == 0  →  len SVP  (layer_id=0)
      has_dcp == 1  →  SVP + DCP ako dve samostatné vzorky
      has_dcp == 2  →  len DCP  (layer_id=1)
    """
    def __init__(self, df: pd.DataFrame, data_root: str, image_size: int, is_train: bool = True):
        self.data_root = data_root
        self.transform = build_augmentations(image_size, is_train)
        samples = []
        for _, row in df.iterrows():
            svp     = row.get("svp_path", None)
            dcp     = row.get("dcp_path", None)
            has_dcp = int(row.get("has_dcp", 0))
            if pd.notna(svp) and str(svp).strip():
                samples.append({"path": str(svp), "layer_id": 0})
            if has_dcp in [1, 2] and pd.notna(dcp) and str(dcp).strip():
                samples.append({"path": str(dcp), "layer_id": 1})
        self.samples = samples
        n_svp = sum(1 for s in samples if s["layer_id"] == 0)
        n_dcp = sum(1 for s in samples if s["layer_id"] == 1)
        logger.info(f"Phase1Dataset: {len(samples)} vzoriek  (SVP={n_svp}, DCP={n_dcp})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = safe_load_image(s["path"], self.data_root, self.transform)
        if img is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return img, torch.tensor(s["layer_id"], dtype=torch.long)


class Phase1BalancedDataset(Dataset):
    """
    Rovnaká logika ako Phase1Dataset, ale drží zvlášť indexy SVP a DCP
    pre BalancedSVPDCPBatchSampler (50/50 SVP:DCP v každom batchi).
    """
    def __init__(self, df: pd.DataFrame, data_root: str, image_size: int, is_train: bool = True):
        self.data_root   = data_root
        self.transform   = build_augmentations(image_size, is_train)
        self.samples     = []
        self.svp_indices = []
        self.dcp_indices = []

        for _, row in df.iterrows():
            svp     = row.get("svp_path", None)
            dcp     = row.get("dcp_path", None)
            has_dcp = int(row.get("has_dcp", 0))

            if pd.notna(svp) and str(svp).strip():
                self.svp_indices.append(len(self.samples))
                self.samples.append({"path": str(svp), "layer_id": 0})

            if has_dcp in [1, 2] and pd.notna(dcp) and str(dcp).strip():
                self.dcp_indices.append(len(self.samples))
                self.samples.append({"path": str(dcp), "layer_id": 1})

        logger.info(
            f"Phase1BalancedDataset: {len(self.samples)} vzoriek "
            f"(SVP={len(self.svp_indices)}, DCP={len(self.dcp_indices)})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = safe_load_image(s["path"], self.data_root, self.transform)
        if img is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return img, torch.tensor(s["layer_id"], dtype=torch.long)


class Phase2Dataset(Dataset):
    """
    Paired SVP + DCP vzorky pre Phase 2.

    Berie len riadky kde has_dcp == 1 (má oba obrázky).
    Podporuje modality dropout: DCP vynulované s pravdepodobnosťou dropout_p.
    Vracia aj dataset_name pre GRL domain classifier.
    """
    def __init__(self, df: pd.DataFrame, data_root: str, image_size: int,
                 is_train: bool = True, modality_dropout_prob: float = 0.2):
        self.data_root = data_root
        self.transform = build_augmentations(image_size, is_train)
        self.dropout_p = modality_dropout_prob if is_train else 0.0

        paired = df[df["has_dcp"] == 1]
        self.samples = []
        for _, row in paired.iterrows():
            svp          = row.get("svp_path", None)
            dcp          = row.get("dcp_path", None)
            dataset_name = str(row.get("dataset", "UNKNOWN"))
            if pd.notna(svp) and str(svp).strip() and pd.notna(dcp) and str(dcp).strip():
                self.samples.append({
                    "svp_path":     str(svp),
                    "dcp_path":     str(dcp),
                    "dataset_name": dataset_name,
                })
        logger.info(f"Phase2Dataset: {len(self.samples)} paired vzoriek")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        svp = safe_load_image(s["svp_path"], self.data_root, self.transform)
        dcp = safe_load_image(s["dcp_path"], self.data_root, self.transform)
        if svp is None or dcp is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        dcp_ok = True
        if self.dropout_p > 0 and random.random() < self.dropout_p:
            dcp    = torch.zeros_like(dcp)
            dcp_ok = False

        return (
            svp,
            dcp,
            torch.tensor(int(dcp_ok), dtype=torch.long),
            s["dataset_name"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# BALANCED SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

class BalancedSVPDCPBatchSampler(Sampler):
    """
    Každý batch: 50% SVP + 50% DCP.
    Oversampluje menšiu skupinu ak treba.
    """
    def __init__(self, dataset: Phase1BalancedDataset, batch_size: int, drop_last: bool = True):
        assert batch_size % 2 == 0, "batch_size musí byť párne pre 50/50 sampling"
        self.svp_indices = dataset.svp_indices
        self.dcp_indices = dataset.dcp_indices
        self.batch_size  = batch_size
        self.half        = batch_size // 2
        self.drop_last   = drop_last
        self.n_batches   = len(self.svp_indices) // self.half
        self.n_samples   = self.n_batches * self.batch_size
        logger.info(
            f"BalancedSVPDCPBatchSampler: {self.n_batches} batchov | "
            f"{self.half} SVP + {self.half} DCP na batch"
        )

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        svp = np.random.permutation(self.svp_indices).tolist()
        # oversampling DCP ak je ich menej ako SVP
        repeat_factor = (len(svp) // max(len(self.dcp_indices), 1)) + 2
        dcp_pool      = (self.dcp_indices * repeat_factor)[:len(svp)]
        dcp           = np.random.permutation(dcp_pool).tolist()

        indices = []
        for i in range(0, len(svp) - self.half + 1, self.half):
            batch = svp[i:i + self.half] + dcp[i:i + self.half]
            random.shuffle(batch)
            indices.extend(batch)
        return iter(indices)


# ══════════════════════════════════════════════════════════════════════════════
# MODELY — STAVEBNÉ BLOKY
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # [B, N, D]


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    ViT-Small encoder s modality embeddingom.
    Modality embedding (SVP=0, DCP=1) je pridaný ku každému patch tokenu.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0, num_modalities=2):
        super().__init__()
        self.patch_embed    = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches         = self.patch_embed.num_patches
        self.cls_token      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed      = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)
        self.blocks         = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio)
                                               for _ in range(depth)])
        self.norm           = nn.LayerNorm(embed_dim)
        self.embed_dim      = embed_dim
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, layer_id: torch.Tensor, mask_bool=None):
        """
        Args:
            x:         [B, 1, H, W]
            layer_id:  [B]  — 0=SVP, 1=DCP
            mask_bool: [B, N] bool — True=visible, None=all visible
        Returns:
            tokens:      [B, n_vis+1, D]
            ids_restore: [B, N] or None
        """
        B   = x.shape[0]
        tok = self.patch_embed(x)
        tok = tok + self.modality_embed(layer_id).unsqueeze(1)
        tok = tok + self.pos_embed[:, 1:, :]

        if mask_bool is not None:
            ids_shuffle = torch.argsort((~mask_bool).float(), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            num_visible = int(mask_bool[0].sum().item())
            vis_ids     = ids_shuffle[:, :num_visible]
            tok         = torch.gather(tok, 1,
                              vis_ids.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        else:
            ids_restore = None

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tok = torch.cat([cls, tok], dim=1)
        tok = self.blocks(tok)
        tok = self.norm(tok)
        return tok, ids_restore


class MAEDecoder(nn.Module):
    def __init__(self, num_patches, enc_dim=384, dec_dim=256,
                 depth=4, num_heads=8, mlp_ratio=4.0, patch_size=16, in_chans=1):
        super().__init__()
        self.num_patches = num_patches
        self.embed       = nn.Linear(enc_dim, dec_dim)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, dec_dim))
        self.blocks      = nn.Sequential(*[Block(dec_dim, num_heads, mlp_ratio)
                                           for _ in range(depth)])
        self.norm        = nn.LayerNorm(dec_dim)
        self.pred        = nn.Linear(dec_dim, patch_size * patch_size * in_chans)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)

    def forward(self, enc_tokens, ids_restore):
        """enc_tokens: [B, n_vis+1, enc_dim], ids_restore: [B, N]"""
        B        = enc_tokens.shape[0]
        N        = self.num_patches
        x        = self.embed(enc_tokens)
        n_vis    = x.shape[1] - 1
        n_masked = N - n_vis
        mask_tok = self.mask_token.expand(B, n_masked, -1)
        x_full   = torch.cat([x[:, 1:, :], mask_tok], dim=1)
        x_full   = torch.gather(x_full, 1,
                       ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
        x_full   = x_full + self.pos_embed[:, 1:, :]
        cls_dec  = x[:, :1, :] + self.pos_embed[:, :1, :]
        x_full   = torch.cat([cls_dec, x_full], dim=1)
        x_full   = self.blocks(x_full)
        x_full   = self.norm(x_full)
        return self.pred(x_full[:, 1:, :])   # [B, N, p²]


# ── Phase 1 MAE ───────────────────────────────────────────────────────────────

class MaskedAutoencoder(nn.Module):
    def __init__(self, cfg: Phase1Config):
        super().__init__()
        self.cfg         = cfg
        self.patch_size  = cfg.patch_size
        num_patches      = (cfg.image_size // cfg.patch_size) ** 2
        self.num_patches = num_patches
        self.encoder     = ViTEncoder(
            img_size=cfg.image_size, patch_size=cfg.patch_size, in_chans=cfg.in_chans,
            embed_dim=cfg.enc_embed_dim, depth=cfg.enc_depth, num_heads=cfg.enc_num_heads,
            mlp_ratio=cfg.enc_mlp_ratio)
        self.decoder     = MAEDecoder(
            num_patches=num_patches, enc_dim=cfg.enc_embed_dim, dec_dim=cfg.dec_embed_dim,
            depth=cfg.dec_depth, num_heads=cfg.dec_num_heads, mlp_ratio=cfg.dec_mlp_ratio,
            patch_size=cfg.patch_size, in_chans=cfg.in_chans)

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(B, h * w, p * p * C)

    def random_mask(self, B, N, ratio, device):
        n_vis       = int(N * (1 - ratio))
        noise       = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_bool   = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_bool.scatter_(1, ids_shuffle[:, :n_vis], True)
        return mask_bool, ids_restore

    def forward(self, imgs, layer_ids, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.cfg.mask_ratio
        mask_bool, ids_restore = self.random_mask(
            imgs.shape[0], self.num_patches, mask_ratio, imgs.device)
        enc_tokens, _ = self.encoder(imgs, layer_ids, mask_bool)
        pred          = self.decoder(enc_tokens, ids_restore)
        target        = self.patchify(imgs)
        loss          = ((pred - target) ** 2)[~mask_bool].mean()
        return loss, pred, mask_bool


# ── Phase 2 MultiMAE ──────────────────────────────────────────────────────────

class MultiMAE(nn.Module):
    """
    Zdieľaný encoder spracuje každú modalitu samostatne (s modality embeddingom).
    Dva separátne dekodéry rekonštruujú SVP a DCP patche.
    Podporuje modality dropout.
    """
    def __init__(self, cfg: Phase2Config):
        super().__init__()
        self.cfg         = cfg
        num_patches      = (cfg.image_size // cfg.patch_size) ** 2
        self.num_patches = num_patches
        self.patch_size  = cfg.patch_size
        self.encoder     = ViTEncoder(
            img_size=cfg.image_size, patch_size=cfg.patch_size, in_chans=cfg.in_chans,
            embed_dim=cfg.enc_embed_dim, depth=cfg.enc_depth, num_heads=cfg.enc_num_heads,
            mlp_ratio=cfg.enc_mlp_ratio)
        dec_kw = dict(
            num_patches=num_patches, enc_dim=cfg.enc_embed_dim, dec_dim=cfg.dec_embed_dim,
            depth=cfg.dec_depth, num_heads=cfg.dec_num_heads, mlp_ratio=cfg.dec_mlp_ratio,
            patch_size=cfg.patch_size, in_chans=cfg.in_chans)
        self.decoder_svp = MAEDecoder(**dec_kw)
        self.decoder_dcp = MAEDecoder(**dec_kw)

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(B, h * w, p * p * C)

    def random_mask(self, B, N, ratio, device):
        n_vis       = int(N * (1 - ratio))
        noise       = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_bool   = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_bool.scatter_(1, ids_shuffle[:, :n_vis], True)
        return mask_bool, ids_restore

    def forward(self, svp_imgs, dcp_imgs, dcp_available):
        B, dev = svp_imgs.shape[0], svp_imgs.device

        # ── SVP (vždy) ───────────────────────────────────────────────────────
        lid_svp      = torch.zeros(B, dtype=torch.long, device=dev)
        m_svp, r_svp = self.random_mask(B, self.num_patches, self.cfg.mask_ratio_svp, dev)
        enc_svp, _   = self.encoder(svp_imgs, lid_svp, m_svp)
        pred_svp     = self.decoder_svp(enc_svp, r_svp)
        tgt_svp      = self.patchify(svp_imgs)
        loss_svp     = ((pred_svp - tgt_svp) ** 2)[~m_svp].mean()

        # ── DCP (len kde je dostupný) ─────────────────────────────────────────
        dcp_ok   = dcp_available.bool()
        loss_dcp = torch.tensor(0.0, device=dev)
        if dcp_ok.any():
            dcp_v        = dcp_imgs[dcp_ok]
            Bd           = dcp_v.shape[0]
            lid_dcp      = torch.ones(Bd, dtype=torch.long, device=dev)
            m_dcp, r_dcp = self.random_mask(Bd, self.num_patches, self.cfg.mask_ratio_dcp, dev)
            enc_dcp, _   = self.encoder(dcp_v, lid_dcp, m_dcp)
            pred_dcp     = self.decoder_dcp(enc_dcp, r_dcp)
            tgt_dcp      = self.patchify(dcp_v)
            loss_dcp     = ((pred_dcp - tgt_dcp) ** 2)[~m_dcp].mean()

        loss = self.cfg.loss_weight_svp * loss_svp + self.cfg.loss_weight_dcp * loss_dcp
        return loss, loss_svp, loss_dcp


# ── GRL komponenty ────────────────────────────────────────────────────────────

def dann_lambda(progress: float, max_lambda: float = 1.0) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def build_dataset_vocab(dataset: Phase2Dataset):
    names       = sorted({s["dataset_name"] for s in dataset.samples})
    name_to_idx = {name: i for i, name in enumerate(names)}
    return names, name_to_idx


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x, alpha=1.0):
        return GradientReversalFn.apply(x, alpha)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ── Phase 2 MultiMAE + GRL ────────────────────────────────────────────────────

class MultiMAEWithGRL(nn.Module):
    """
    Phase 2 MultiMAE + adversarial dataset confusion via GRL.
    Rekonštrukcia ostáva rovnaká ako v baseline MultiMAE.
    Z CLS tokenu sa predikuje dataset, cez GRL encoder potláča dataset informáciu.
    """
    def __init__(self, cfg: Phase2Config, num_datasets: int):
        super().__init__()
        self.cfg         = cfg
        self.patch_size  = cfg.patch_size
        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2

        self.encoder = ViTEncoder(
            img_size=cfg.image_size, patch_size=cfg.patch_size, in_chans=cfg.in_chans,
            embed_dim=cfg.enc_embed_dim, depth=cfg.enc_depth, num_heads=cfg.enc_num_heads,
            mlp_ratio=cfg.enc_mlp_ratio)

        dec_kw = dict(
            num_patches=self.num_patches, enc_dim=cfg.enc_embed_dim, dec_dim=cfg.dec_embed_dim,
            depth=cfg.dec_depth, num_heads=cfg.dec_num_heads, mlp_ratio=cfg.dec_mlp_ratio,
            patch_size=cfg.patch_size, in_chans=cfg.in_chans)
        self.decoder_svp = MAEDecoder(**dec_kw)
        self.decoder_dcp = MAEDecoder(**dec_kw)

        self.grl         = GradientReversal()
        self.domain_head = DomainClassifier(
            in_dim=cfg.enc_embed_dim, hidden_dim=cfg.domain_hidden_dim,
            out_dim=num_datasets, dropout=cfg.domain_dropout)

    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(B, h * w, p * p * C)

    def random_mask(self, B, N, ratio, device):
        n_vis       = int(N * (1 - ratio))
        noise       = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_bool   = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_bool.scatter_(1, ids_shuffle[:, :n_vis], True)
        return mask_bool, ids_restore

    def _domain_loss_from_tokens(self, enc_tokens, dataset_labels, grl_alpha):
        cls_feat = enc_tokens[:, 0, :]
        feat_rev = self.grl(cls_feat, alpha=grl_alpha)
        logits   = self.domain_head(feat_rev)
        loss     = F.cross_entropy(logits, dataset_labels)
        return loss, logits

    def forward(self, svp_imgs, dcp_imgs, dcp_available, dataset_labels, grl_alpha=0.0):
        B, dev = svp_imgs.shape[0], svp_imgs.device

        # ── SVP ──────────────────────────────────────────────────────────────
        lid_svp      = torch.zeros(B, dtype=torch.long, device=dev)
        m_svp, r_svp = self.random_mask(B, self.num_patches, self.cfg.mask_ratio_svp, dev)
        enc_svp, _   = self.encoder(svp_imgs, lid_svp, m_svp)
        pred_svp     = self.decoder_svp(enc_svp, r_svp)
        tgt_svp      = self.patchify(svp_imgs)
        loss_svp     = ((pred_svp - tgt_svp) ** 2)[~m_svp].mean()
        domain_loss_svp, _ = self._domain_loss_from_tokens(enc_svp, dataset_labels, grl_alpha)

        # ── DCP ──────────────────────────────────────────────────────────────
        dcp_ok          = dcp_available.bool()
        loss_dcp        = torch.tensor(0.0, device=dev)
        domain_loss_dcp = torch.tensor(0.0, device=dev)

        if dcp_ok.any():
            dcp_v        = dcp_imgs[dcp_ok]
            y_dcp        = dataset_labels[dcp_ok]
            Bd           = dcp_v.shape[0]
            lid_dcp      = torch.ones(Bd, dtype=torch.long, device=dev)
            m_dcp, r_dcp = self.random_mask(Bd, self.num_patches, self.cfg.mask_ratio_dcp, dev)
            enc_dcp, _   = self.encoder(dcp_v, lid_dcp, m_dcp)
            pred_dcp     = self.decoder_dcp(enc_dcp, r_dcp)
            tgt_dcp      = self.patchify(dcp_v)
            loss_dcp     = ((pred_dcp - tgt_dcp) ** 2)[~m_dcp].mean()
            domain_loss_dcp, _ = self._domain_loss_from_tokens(enc_dcp, y_dcp, grl_alpha)

        recon_loss  = self.cfg.loss_weight_svp * loss_svp + self.cfg.loss_weight_dcp * loss_dcp
        domain_loss = (0.5 * (domain_loss_svp + domain_loss_dcp)
                       if dcp_ok.any() else domain_loss_svp)
        total_loss  = recon_loss + self.cfg.domain_loss_weight * domain_loss

        return {
            "loss":             total_loss,
            "recon_loss":       recon_loss,
            "loss_svp":         loss_svp,
            "loss_dcp":         loss_dcp,
            "domain_loss":      domain_loss,
            "domain_loss_svp":  domain_loss_svp,
            "domain_loss_dcp":  domain_loss_dcp,
        }


# ══════════════════════════════════════════════════════════════════════════════
# POMOCNÉ FUNKCIE
# ══════════════════════════════════════════════════════════════════════════════

def cosine_lr(optimizer, base_lr, min_lr, total_epochs, warmup_epochs, epoch):
    """Cosine schedule s lineárnym warmupom. Aktualizuje optimizer in-place."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        t  = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def log_vram(prefix=""):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logger.info(f"{prefix}VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")


def save_checkpoint(model, optimizer, epoch, loss, ckpt_dir: Path, name: str) -> Path:
    path = ckpt_dir / name
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
    }, path)
    logger.info(f"  ✓ Checkpoint uložený → {path}")
    return path


def save_encoder(encoder, out_dir: Path, name="final_encoder.pth") -> Path:
    path = out_dir / name
    torch.save(encoder.state_dict(), path)
    logger.info(f"  ✓ Encoder váhy uložené → {path}")
    return path


def load_phase1_encoder(model_encoder, encoder_path: Optional[str], device):
    """
    Načíta Phase1 encoder váhy do modelu.
    Ak cesta neexistuje alebo je None → warning + random init.
    """
    if encoder_path and Path(encoder_path).exists():
        state = torch.load(encoder_path, map_location=device)
        # podpora pre checkpoint aj čistý state_dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
            # vyberieme len encoder kľúče ak je to celý model
            state = {k.replace("encoder.", "", 1): v
                     for k, v in state.items() if k.startswith("encoder.")} or state
        missing, unexpected = model_encoder.load_state_dict(state, strict=False)
        logger.info(f"Phase1 encoder načítaný z: {encoder_path}")
        if missing:
            logger.warning(f"  Chýbajúce kľúče: {missing}")
        if unexpected:
            logger.warning(f"  Neočakávané kľúče: {unexpected}")
    else:
        logger.warning("Phase1 encoder nenájdený — tréning začína s random váhami!")


# ══════════════════════════════════════════════════════════════════════════════
# TRÉNINGOVÉ FUNKCIE
# ══════════════════════════════════════════════════════════════════════════════

def _build_phase1_loaders(cfg):
    """Interná pomocná funkcia — načíta dáta a vráti train/val loadery."""
    mae_df    = load_mae_df(cfg.excel_path)
    tr_df     = mae_df[mae_df["split_mae"].str.lower() == "train"]
    va_df     = mae_df[mae_df["split_mae"].str.lower() == "val"]
    tr_ds     = Phase1Dataset(tr_df, cfg.data_root, cfg.image_size, is_train=True)
    va_ds     = Phase1Dataset(va_df, cfg.data_root, cfg.image_size, is_train=False)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           drop_last=cfg.drop_last, persistent_workers=cfg.num_workers > 0)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           persistent_workers=cfg.num_workers > 0)
    return tr_loader, va_loader


def _run_phase1_training(cfg, tr_loader, va_loader, wandb_tags):
    """Interná pomocná funkcia — hlavná tréningová slučka pre Phase 1."""
    out_dir  = Path(cfg.output_base) / cfg.run_id
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Výstup: {out_dir}")

    if WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                   name=cfg.wandb_run_name or f"{wandb_tags[0]}_{cfg.run_id}",
                   config=cfg.__dict__, tags=wandb_tags, reinit=True)

    model     = MaskedAutoencoder(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_val_loss = float("inf")
    t0 = time.time()
    wandb_prefix = wandb_tags[0].replace("phase1", "p1").replace("_balanced", "b").replace("-", "")

    for epoch in range(cfg.epochs):
        model.train()
        lr         = cosine_lr(optimizer, cfg.lr, cfg.min_lr, cfg.epochs, cfg.warmup_epochs, epoch)
        total_loss = 0.0

        for imgs, layer_ids in tr_loader:
            imgs, layer_ids = (imgs.to(device, non_blocking=True),
                               layer_ids.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss, _, _ = model(imgs, layer_ids, cfg.mask_ratio)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train = total_loss / len(tr_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, layer_ids in va_loader:
                imgs, layer_ids = (imgs.to(device, non_blocking=True),
                                   layer_ids.to(device, non_blocking=True))
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    loss, _, _ = model(imgs, layer_ids, cfg.mask_ratio)
                val_loss += loss.item()
        avg_val = val_loss / max(len(va_loader), 1)

        elapsed = (time.time() - t0) / 60
        logger.info(f"[{wandb_prefix.upper()}] Ep {epoch+1:>4}/{cfg.epochs} | lr={lr:.2e} | "
                    f"train={avg_train:.4f} | val={avg_val:.4f} | {elapsed:.1f}m")

        if WANDB_AVAILABLE:
            wandb.log({f"{wandb_prefix}/train_loss": avg_train,
                       f"{wandb_prefix}/val_loss":   avg_val,
                       f"{wandb_prefix}/lr":         lr,
                       f"{wandb_prefix}/epoch":      epoch + 1}, step=epoch + 1)

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(model, optimizer, epoch + 1, avg_val,
                            ckpt_dir, f"epoch_{epoch+1:04d}.pth")
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(model, optimizer, epoch + 1, avg_val, ckpt_dir, "best.pth")
            logger.info(f"  Nový najlepší val loss: {best_val_loss:.4f}")

        torch.cuda.empty_cache()
        log_vram(f"[ep{epoch+1}] ")

    enc_path = save_encoder(model.encoder, out_dir)
    if WANDB_AVAILABLE:
        wandb.save(str(enc_path))
        wandb.finish()

    logger.info(f"Hotovo. Najlepší val loss: {best_val_loss:.4f}")
    return model.encoder, str(enc_path)


# ── Verejné tréningové funkcie ────────────────────────────────────────────────

def train_phase1(cfg: Phase1Config):
    """Phase 1 baseline"""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    logger.info("=" * 60)
    logger.info("PHASE 1 — BASELINE")
    logger.info("=" * 60)
    tr_loader, va_loader = _build_phase1_loaders(cfg)
    return _run_phase1_training(cfg, tr_loader, va_loader,
                                wandb_tags=["phase1", "mae", "octa"])


def train_phase1_balanced(cfg: Phase1BalancedConfig):
    """Phase 1 balanced — 50/50 SVP:DCP v každom batchi."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    logger.info("=" * 60)
    logger.info("PHASE 1 — BALANCED (50/50 SVP:DCP)")
    logger.info("=" * 60)

    mae_df    = load_mae_df(cfg.excel_path)
    tr_df     = mae_df[mae_df["split_mae"].str.lower() == "train"]
    va_df     = mae_df[mae_df["split_mae"].str.lower() == "val"]
    tr_ds     = Phase1BalancedDataset(tr_df, cfg.data_root, cfg.image_size, is_train=True)
    va_ds     = Phase1BalancedDataset(va_df, cfg.data_root, cfg.image_size, is_train=False)

    sampler   = BalancedSVPDCPBatchSampler(tr_ds, cfg.batch_size, cfg.drop_last)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, sampler=sampler,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           drop_last=False, persistent_workers=cfg.num_workers > 0)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           persistent_workers=cfg.num_workers > 0)

    return _run_phase1_training(cfg, tr_loader, va_loader,
                                wandb_tags=["phase1", "mae", "octa", "balanced"])


def train_phase2(cfg: Phase2Config):
    """Phase 2 baseline MultiMAE — bez GRL."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    logger.info("=" * 60)
    logger.info("PHASE 2 — BASELINE MultiMAE")
    logger.info("=" * 60)

    out_dir  = Path(cfg.output_base) / cfg.run_id
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                   name=cfg.wandb_run_name or f"phase2_{cfg.run_id}",
                   config=cfg.__dict__, tags=["phase2", "multimae", "octa"], reinit=True)

    mae_df    = load_mae_df(cfg.excel_path)
    tr_df     = mae_df[mae_df["split_mae"].str.lower() == "train"]
    va_df     = mae_df[mae_df["split_mae"].str.lower() == "val"]
    tr_ds     = Phase2Dataset(tr_df, cfg.data_root, cfg.image_size, is_train=True,
                               modality_dropout_prob=cfg.modality_dropout_prob)
    va_ds     = Phase2Dataset(va_df, cfg.data_root, cfg.image_size, is_train=False)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           drop_last=cfg.drop_last, persistent_workers=cfg.num_workers > 0)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           persistent_workers=cfg.num_workers > 0)

    model = MultiMAE(cfg).to(device)
    load_phase1_encoder(model.encoder, cfg.phase1_encoder_path, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        lr = cosine_lr(optimizer, cfg.lr, cfg.min_lr, cfg.epochs, cfg.warmup_epochs, epoch)
        tot, tot_s, tot_d, n = 0.0, 0.0, 0.0, 0

        for svp, dcp, dcp_ok, _ in tr_loader:   
            svp    = svp.to(device, non_blocking=True)
            dcp    = dcp.to(device, non_blocking=True)
            dcp_ok = dcp_ok.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss, ls, ld = model(svp, dcp, dcp_ok)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            tot += loss.item(); tot_s += ls.item(); tot_d += ld.item(); n += 1

        avg_t, avg_s, avg_d = tot / n, tot_s / n, tot_d / n

        model.eval()
        val_tot, val_n = 0.0, 0
        with torch.no_grad():
            for svp, dcp, dcp_ok, _ in va_loader:
                svp, dcp, dcp_ok = (svp.to(device, non_blocking=True),
                                    dcp.to(device, non_blocking=True),
                                    dcp_ok.to(device, non_blocking=True))
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    loss, _, _ = model(svp, dcp, dcp_ok)
                val_tot += loss.item(); val_n += 1
        avg_val = val_tot / max(val_n, 1)

        elapsed = (time.time() - t0) / 60
        logger.info(f"[P2] Ep {epoch+1:>4}/{cfg.epochs} | lr={lr:.2e} | "
                    f"train={avg_t:.4f} (SVP:{avg_s:.4f} DCP:{avg_d:.4f}) | "
                    f"val={avg_val:.4f} | {elapsed:.1f}m")

        if WANDB_AVAILABLE:
            wandb.log({"p2/train_loss": avg_t, "p2/train_loss_svp": avg_s,
                       "p2/train_loss_dcp": avg_d, "p2/val_loss": avg_val,
                       "p2/lr": lr, "p2/epoch": epoch + 1}, step=epoch + 1)

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(model, optimizer, epoch + 1, avg_val,
                            ckpt_dir, f"epoch_{epoch+1:04d}.pth")
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_checkpoint(model, optimizer, epoch + 1, avg_val, ckpt_dir, "best.pth")
            logger.info(f"  Nový najlepší val loss: {best_val_loss:.4f}")

        torch.cuda.empty_cache()
        log_vram(f"[P2 ep{epoch+1}] ")

    enc_path = save_encoder(model.encoder, out_dir)
    if WANDB_AVAILABLE:
        wandb.save(str(enc_path)); wandb.finish()
    logger.info(f"Phase 2 hotovo. Najlepší val loss: {best_val_loss:.4f}")
    return model.encoder, str(enc_path)


def train_phase2_grl(cfg: Phase2Config):
    """Phase 2 MultiMAE + GRL """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    logger.info("=" * 60)
    logger.info("PHASE 2 — MultiMAE + GRL")
    logger.info("=" * 60)

    out_dir  = Path(cfg.output_base) / cfg.run_id
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                   name=cfg.wandb_run_name or f"phase2_grl_{cfg.run_id}",
                   config=cfg.__dict__, tags=["phase2", "multimae", "octa", "grl"], reinit=True)

    mae_df    = load_mae_df(cfg.excel_path)
    tr_df     = mae_df[mae_df["split_mae"].str.lower() == "train"]
    va_df     = mae_df[mae_df["split_mae"].str.lower() == "val"]
    tr_ds     = Phase2Dataset(tr_df, cfg.data_root, cfg.image_size, is_train=True,
                               modality_dropout_prob=cfg.modality_dropout_prob)
    va_ds     = Phase2Dataset(va_df, cfg.data_root, cfg.image_size, is_train=False)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           drop_last=cfg.drop_last, persistent_workers=cfg.num_workers > 0)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                           persistent_workers=cfg.num_workers > 0)

    dataset_names, dataset_to_idx = build_dataset_vocab(tr_ds)
    logger.info(f"Domain classes ({len(dataset_names)}): {dataset_names}")

    model = MultiMAEWithGRL(cfg, num_datasets=len(dataset_names)).to(device)
    load_phase1_encoder(model.encoder, cfg.phase1_encoder_path, device)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler       = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    best_val_recon = float("inf")
    t0           = time.time()
    total_steps  = cfg.epochs * max(len(tr_loader), 1)
    global_step  = 0

    for epoch in range(cfg.epochs):
        model.train()
        lr = cosine_lr(optimizer, cfg.lr, cfg.min_lr, cfg.epochs, cfg.warmup_epochs, epoch)
        tr_loss = tr_recon = tr_svp = tr_dcp = tr_domain = 0.0
        n = 0
        grl_alpha_epoch = 0.0

        for svp, dcp, dcp_ok, dataset_names_batch in tr_loader:
            svp    = svp.to(device, non_blocking=True)
            dcp    = dcp.to(device, non_blocking=True)
            dcp_ok = dcp_ok.to(device, non_blocking=True)
            dataset_labels = torch.tensor(
                [dataset_to_idx[str(x)] for x in dataset_names_batch],
                dtype=torch.long, device=device)

            progress        = global_step / max(total_steps - 1, 1)
            grl_alpha       = dann_lambda(progress, max_lambda=cfg.grl_max_lambda)
            grl_alpha_epoch = grl_alpha

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                out  = model(svp_imgs=svp, dcp_imgs=dcp, dcp_available=dcp_ok,
                             dataset_labels=dataset_labels, grl_alpha=grl_alpha)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tr_loss   += out["loss"].item()
            tr_recon  += out["recon_loss"].item()
            tr_svp    += out["loss_svp"].item()
            tr_dcp    += out["loss_dcp"].item()
            tr_domain += out["domain_loss"].item()
            n += 1; global_step += 1

        avg_t, avg_recon = tr_loss/n, tr_recon/n
        avg_s, avg_d     = tr_svp/n,  tr_dcp/n
        avg_domain       = tr_domain/n

        model.eval()
        val_tot = val_recon = val_s = val_d = val_domain = 0.0
        val_n   = 0
        with torch.no_grad():
            for svp, dcp, dcp_ok, dataset_names_batch in va_loader:
                svp    = svp.to(device, non_blocking=True)
                dcp    = dcp.to(device, non_blocking=True)
                dcp_ok = dcp_ok.to(device, non_blocking=True)
                dataset_labels = torch.tensor(
                    [dataset_to_idx[str(x)] for x in dataset_names_batch],
                    dtype=torch.long, device=device)
                with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                    out = model(svp_imgs=svp, dcp_imgs=dcp, dcp_available=dcp_ok,
                                dataset_labels=dataset_labels,
                                grl_alpha=cfg.grl_max_lambda)
                val_tot    += out["loss"].item()
                val_recon  += out["recon_loss"].item()
                val_s      += out["loss_svp"].item()
                val_d      += out["loss_dcp"].item()
                val_domain += out["domain_loss"].item()
                val_n += 1

        avg_val        = val_tot   / max(val_n, 1)
        avg_val_recon  = val_recon / max(val_n, 1)
        avg_val_s      = val_s     / max(val_n, 1)
        avg_val_d      = val_d     / max(val_n, 1)
        avg_val_domain = val_domain/ max(val_n, 1)

        elapsed = (time.time() - t0) / 60
        logger.info(
            f"[P2+GRL] Ep {epoch+1:>4}/{cfg.epochs} | lr={lr:.2e} | grl={grl_alpha_epoch:.3f} | "
            f"train={avg_t:.4f} (recon:{avg_recon:.4f} svp:{avg_s:.4f} "
            f"dcp:{avg_d:.4f} dom:{avg_domain:.4f}) | "
            f"val={avg_val:.4f} (recon:{avg_val_recon:.4f} svp:{avg_val_s:.4f} "
            f"dcp:{avg_val_d:.4f} dom:{avg_val_domain:.4f}) | {elapsed:.1f}m"
        )

        if WANDB_AVAILABLE:
            wandb.log({
                "p2grl/train_loss":       avg_t,
                "p2grl/train_recon_loss": avg_recon,
                "p2grl/train_loss_svp":   avg_s,
                "p2grl/train_loss_dcp":   avg_d,
                "p2grl/train_domain_loss":avg_domain,
                "p2grl/val_loss":         avg_val,
                "p2grl/val_recon_loss":   avg_val_recon,
                "p2grl/val_loss_svp":     avg_val_s,
                "p2grl/val_loss_dcp":     avg_val_d,
                "p2grl/val_domain_loss":  avg_val_domain,
                "p2grl/lr":               lr,
                "p2grl/grl_alpha":        grl_alpha_epoch,
                "p2grl/epoch":            epoch + 1,
            }, step=epoch + 1)

        if (epoch + 1) % cfg.save_every == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(model, optimizer, epoch + 1, avg_val_recon,
                            ckpt_dir, f"epoch_{epoch+1:04d}.pth")
        if avg_val_recon < best_val_recon:
            best_val_recon = avg_val_recon
            save_checkpoint(model, optimizer, epoch + 1, avg_val_recon,
                            ckpt_dir, "best.pth")
            logger.info(f"  Nový najlepší val recon loss: {best_val_recon:.4f}")

        torch.cuda.empty_cache()
        log_vram(f"[P2+GRL ep{epoch+1}] ")

    enc_path = save_encoder(model.encoder, out_dir)
    if WANDB_AVAILABLE:
        wandb.save(str(enc_path)); wandb.finish()
    logger.info(f"Phase 2 + GRL hotovo. Najlepší val recon loss: {best_val_recon:.4f}")
    return model.encoder, str(enc_path)
