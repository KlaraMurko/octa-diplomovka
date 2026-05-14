"""
model.py
--------
ViT Encoder, Fusion modules, MLP Classifier, OctaClassifier.

use_dataset_cond je odstránený — všetky behy boli vykonané s False,
táto vetva nikdy nebola aktívna.
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── ViT Encoder building blocks ───────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_modalities: int = 2,
    ):
        super().__init__()
        self.patch_embed    = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches         = self.patch_embed.num_patches
        self.cls_token      = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed      = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.modality_embed = nn.Embedding(num_modalities, embed_dim)
        self.blocks         = nn.Sequential(
            *[Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm      = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(
        self, x: torch.Tensor, layer_id: torch.Tensor, mask_bool=None
    ) -> torch.Tensor:
        B   = x.shape[0]
        tok = self.patch_embed(x)
        tok = tok + self.modality_embed(layer_id).unsqueeze(1)
        tok = tok + self.pos_embed[:, 1:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tok = torch.cat([cls, tok], dim=1)
        tok = self.blocks(tok)
        tok = self.norm(tok)
        return tok

    def extract_features(
        self, x: torch.Tensor, layer_id: torch.Tensor, mode: str = "cls_mean"
    ) -> torch.Tensor:
        tok       = self.forward(x, layer_id=layer_id)
        cls_tok   = tok[:, 0, :]
        patch_tok = tok[:, 1:, :]
        if mode == "cls":
            return cls_tok
        elif mode == "mean_patch":
            return patch_tok.mean(dim=1)
        elif mode == "cls_mean":
            return 0.5 * cls_tok + 0.5 * patch_tok.mean(dim=1)
        else:
            raise ValueError(f"Neznámy encoder_mode: {mode}")


def load_encoder(encoder_path) -> ViTEncoder:
    encoder = ViTEncoder()
    state   = torch.load(encoder_path, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Encoder chýbajúce kľúče: {missing}")
    if unexpected:
        logger.warning(f"Encoder neočakávané kľúče: {unexpected}")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    logger.info(f"Encoder načítaný (všetko zmrazené): {encoder_path}")
    return encoder


def setup_encoder_finetuning(
    encoder: ViTEncoder, unfreeze_last_n: int
) -> List[torch.nn.Parameter]:
    """
    Rozmrazí posledných unfreeze_last_n blokov + norm enkódera.
    Vracia zoznam parametrov pre optimizer.

    unfreeze_last_n = 0  → všetko zmrazené, vracia []
    """
    if unfreeze_last_n == 0:
        logger.info("Encoder: fully frozen")
        return []

    total_blocks = len(encoder.blocks)
    start_block  = max(0, total_blocks - unfreeze_last_n)

    for i, block in enumerate(encoder.blocks):
        if i >= start_block:
            for p in block.parameters():
                p.requires_grad = True

    for p in encoder.norm.parameters():
        p.requires_grad = True

    if unfreeze_last_n >= total_blocks:
        for p in encoder.patch_embed.parameters():
            p.requires_grad = True
        encoder.cls_token.requires_grad      = True
        encoder.pos_embed.requires_grad      = True
        for p in encoder.modality_embed.parameters():
            p.requires_grad = True

    unfreeze_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    frozen_params   = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    logger.info(
        f"Encoder: unfreeze_last_n={unfreeze_last_n} "
        f"(bloky {start_block}–{total_blocks - 1} + norm) | "
        f"trainable={unfreeze_params:,} | frozen={frozen_params:,}"
    )
    return [p for p in encoder.parameters() if p.requires_grad]


# ── Fusion modules ────────────────────────────────────────────────────────────

class GateFusion(nn.Module):
    """e_fused = e_svp + sigmoid(W·[e_svp, e_dcp]) · e_dcp"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, e_svp: torch.Tensor, e_dcp: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([e_svp, e_dcp], dim=-1)))
        return e_svp + g * e_dcp


class TransformerFusion(nn.Module):
    """Cross-attention: SVP = query, DCP = key + value."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(embed_dim)

    def forward(self, e_svp: torch.Tensor, e_dcp: torch.Tensor) -> torch.Tensor:
        q   = e_svp.unsqueeze(1)
        kv  = e_dcp.unsqueeze(1)
        out = self.decoder(q, kv)
        return self.norm(out.squeeze(1))


# ── MLP Classifier head ───────────────────────────────────────────────────────

class MLPClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout: float = 0.4,
        use_bn: bool = True,
    ):
        super().__init__()
        layers = []
        prev   = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full model ────────────────────────────────────────────────────────────────

FUSION_TYPES = ["svp_only", "concat", "gate", "transformer", "mean", "product"]


class OctaClassifier(nn.Module):
    """
    Multimodal OCTA classifier.

    Podporované fusion_type:
        "svp_only"    — len SVP embedding
        "concat"      — [e_svp || e_dcp] → MLP
        "gate"        — e_svp + sigmoid(W·[e_svp, e_dcp]) · e_dcp
        "transformer" — cross-attention: SVP=query, DCP=key/value
        "mean"        — (e_svp + e_dcp) / 2
        "product"     — e_svp * e_dcp

    use_dataset_cond bol odstránený (bol vždy False).
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        fusion_type: str,
        encoder_mode: str,
        hidden_dims: List[int],
        dropout: float,
        num_classes: int,
        use_bn: bool = True,
        tf_num_heads: int = 4,
        tf_num_layers: int = 1,
        tf_dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder      = encoder
        self.fusion_type  = fusion_type
        self.encoder_mode = encoder_mode
        self.embed_dim    = encoder.embed_dim

        # Mask token pre chýbajúci DCP
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Fusion
        if fusion_type == "gate":
            self.fusion = GateFusion(self.embed_dim)
            fused_dim   = self.embed_dim
        elif fusion_type == "transformer":
            self.fusion = TransformerFusion(
                self.embed_dim, tf_num_heads, tf_num_layers, tf_dropout
            )
            fused_dim = self.embed_dim
        elif fusion_type == "concat":
            self.fusion = None
            fused_dim   = self.embed_dim * 2
        elif fusion_type in ("mean", "product", "svp_only"):
            self.fusion = None
            fused_dim   = self.embed_dim
        else:
            raise ValueError(f"Neznámy fusion_type: {fusion_type}")

        self.classifier = MLPClassifier(
            fused_dim, hidden_dims, num_classes, dropout, use_bn
        )

        logger.info(
            f"OctaClassifier | fusion={fusion_type} | "
            f"encoder_mode={encoder_mode} | "
            f"fused_dim={fused_dim} | hidden_dims={hidden_dims}"
        )

    def _get_dcp_embedding(
        self,
        dcp_img: torch.Tensor,
        has_dcp: torch.Tensor,
    ) -> torch.Tensor:
        B     = dcp_img.shape[0]
        e_dcp = self.mask_token.expand(B, -1).clone()

        valid_mask = has_dcp.bool()
        if valid_mask.any():
            layer_ids = torch.ones(
                valid_mask.sum(), dtype=torch.long, device=dcp_img.device
            )
            e_dcp[valid_mask] = self.encoder.extract_features(
                dcp_img[valid_mask], layer_ids, mode=self.encoder_mode
            )
        return e_dcp

    def forward(
        self,
        svp_img: torch.Tensor,
        dcp_img: torch.Tensor,
        has_dcp: torch.Tensor,
    ) -> torch.Tensor:
        """
        svp_img : [B, 1, 224, 224]
        dcp_img : [B, 1, 224, 224]
        has_dcp : [B]  LongTensor (0 alebo 1)
        """
        svp_ids = torch.zeros(
            svp_img.shape[0], dtype=torch.long, device=svp_img.device
        )
        e_svp = self.encoder.extract_features(svp_img, svp_ids, mode=self.encoder_mode)

        if self.fusion_type == "svp_only":
            fused = e_svp
        else:
            e_dcp = self._get_dcp_embedding(dcp_img, has_dcp)

            if self.fusion_type == "gate":
                fused = self.fusion(e_svp, e_dcp)
            elif self.fusion_type == "transformer":
                fused = self.fusion(e_svp, e_dcp)
            elif self.fusion_type == "concat":
                fused = torch.cat([e_svp, e_dcp], dim=-1)
            elif self.fusion_type == "mean":
                fused = (e_svp + e_dcp) * 0.5
            elif self.fusion_type == "product":
                fused = e_svp * e_dcp

        return self.classifier(fused)


def build_model(
    encoder_path,
    fusion_type: str,
    encoder_mode: str,
    hidden_dims: List[int],
    dropout: float,
    num_classes: int = 4,
    use_bn: bool = True,
    tf_num_heads: int = 4,
    tf_num_layers: int = 1,
    tf_dropout: float = 0.1,
    unfreeze_last_n_blocks: int = 0,
    device: torch.device = None,
) -> tuple:
    """
    Načíta enkóder, nastaví finetuning, vytvorí model.
    Vracia (model, enc_params).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder    = load_encoder(encoder_path).to(device)
    enc_params = setup_encoder_finetuning(encoder, unfreeze_last_n_blocks)

    model = OctaClassifier(
        encoder=encoder,
        fusion_type=fusion_type,
        encoder_mode=encoder_mode,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_classes=num_classes,
        use_bn=use_bn,
        tf_num_heads=tf_num_heads,
        tf_num_layers=tf_num_layers,
        tf_dropout=tf_dropout,
    ).to(device)

    return model, enc_params
