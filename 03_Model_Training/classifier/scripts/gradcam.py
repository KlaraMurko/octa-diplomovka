"""
gradcam.py
----------
Grad-CAM generovanie pre OctaClassifier.

Gradient tečie:
    patch tokeny posledného bloku enkódera
    → encoder.norm
    → fusion (transformer/gate/...)
    → MLP klasifikátor
    → logit predikovanej triedy
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm_mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .dataset import LABEL_NAMES

logger = logging.getLogger(__name__)


# ── Image loading & display helpers ──────────────────────────────────────────

def build_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def load_image_tensor(
    image_path,
    data_root: Path,
    device: torch.device,
    image_size: int = 224,
) -> torch.Tensor:
    """Načíta obrázok a vráti tensor [1, 1, H, W]."""
    transform = build_val_transform(image_size)
    full_path = Path(data_root) / str(image_path)
    img = Image.open(full_path).convert("L")
    return transform(img).unsqueeze(0).to(device)


def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """Prevedie tensor na numpy array pre zobrazenie [H, W] ∈ [0, 1]."""
    img = t.detach().squeeze().cpu().numpy() * 0.5 + 0.5
    return np.clip(img, 0, 1)


# ── GradCAM core ──────────────────────────────────────────────────────────────

def _make_cam_from_tokens(
    acts: torch.Tensor,
    grads: torch.Tensor,
) -> np.ndarray:
    """
    acts:  [1, N+1, D]  (N+1 = CLS + patches)
    grads: [1, N+1, D]

    Vracia 224×224 CAM mapu normalizovanú na [0, 1].
    """
    patch_acts  = acts[0, 1:, :]    # [N, D]
    patch_grads = grads[0, 1:, :]   # [N, D]

    weights = patch_grads.mean(dim=0)                        # [D]
    cam     = (patch_acts * weights.unsqueeze(0)).sum(dim=-1) # [N]
    cam     = F.relu(cam)

    grid = int(round(cam.shape[0] ** 0.5))
    cam  = cam.reshape(grid, grid).detach().cpu().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam_pil  = Image.fromarray((cam * 255).astype(np.uint8))
    cam_full = np.array(cam_pil.resize((224, 224), Image.BICUBIC)) / 255.0
    return cam_full


def compute_gradcam(
    model,
    svp_tensor: torch.Tensor,
    dcp_tensor: torch.Tensor,
    has_dcp_val: int,
    device: torch.device,
) -> Dict:
    """
    Vypočíta Grad-CAM mapy pre SVP (a DCP ak existuje).

    Vracia dict:
        cam_svp   : np.ndarray [224, 224]
        cam_dcp   : np.ndarray [224, 224] alebo None
        pred_idx  : int
        pred_name : str
        probs     : np.ndarray [num_classes]
    """
    model.eval()

    features: List[torch.Tensor] = []
    grads:    List[torch.Tensor] = []

    def forward_hook(module, inp, out):
        features.append(out)
        out.register_hook(lambda g: grads.append(g))

    hook = model.encoder.norm.register_forward_hook(forward_hook)

    svp_t = svp_tensor.clone().requires_grad_(True)
    dcp_t = dcp_tensor.clone().requires_grad_(True)
    has_d = torch.tensor([has_dcp_val], dtype=torch.long, device=device)

    model.zero_grad(set_to_none=True)
    logits = model(svp_t, dcp_t, has_d)

    pred_idx  = int(logits.argmax(dim=-1).item())
    pred_name = LABEL_NAMES[pred_idx] if pred_idx < len(LABEL_NAMES) else str(pred_idx)

    score = logits[0, pred_idx]
    score.backward()

    hook.remove()

    # Gradienty prichádzajú v opačnom poradí backwardu
    grads_rev = grads[::-1]

    cam_svp = _make_cam_from_tokens(features[0], grads_rev[0])

    cam_dcp = None
    if has_dcp_val == 1 and len(features) > 1 and len(grads_rev) > 1:
        cam_dcp = _make_cam_from_tokens(features[1], grads_rev[1])

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    return {
        "cam_svp"  : cam_svp,
        "cam_dcp"  : cam_dcp,
        "pred_idx" : pred_idx,
        "pred_name": pred_name,
        "probs"    : probs,
    }


# ── Overlay helper ────────────────────────────────────────────────────────────

def overlay_heatmap(
    img_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.55,
    colormap: str = "inferno",
) -> np.ndarray:
    heat   = cm_mpl.get_cmap(colormap)(cam)[..., :3]
    img_3c = np.stack([img_np] * 3, axis=-1)
    return np.clip((1 - alpha) * img_3c + alpha * heat, 0, 1)


# ── Main GradCAM function ─────────────────────────────────────────────────────

def run_gradcam(
    model,
    encoder_path,
    image_list: List[Dict],
    data_root: Path,
    save_dir: Path,
    device: torch.device,
    alpha: float = 0.55,
    colormap: str = "inferno",
    show: bool = True,
) -> None:
    """
    Spustí Grad-CAM pre explicitný zoznam obrázkov.

    image_list: zoznam dictov, každý obsahuje:
        {
            "svp_path"  : "data/001_svp.png",
            "dcp_path"  : "data/001_dcp.png",  # voliteľné
            "has_dcp"   : 1,                    # 0 alebo 1
            "label"     : "AMD",                # skutočná trieda (pre title)
            "sample_id" : "001",                # pre názov súboru
        }

    Príklad použitia v notebooku:
        IMAGE_LIST = [
            {"svp_path": "data/p001_svp.png", "has_dcp": 0, "label": "AMD",  "sample_id": "p001"},
            {"svp_path": "data/p002_svp.png", "dcp_path": "data/p002_dcp.png", "has_dcp": 1, "label": "DR", "sample_id": "p002"},
        ]
        run_gradcam(model, encoder_path, IMAGE_LIST, DATA_ROOT, save_dir, device)
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(image_list):
        sample_id = sample.get("sample_id", f"sample_{i:03d}")
        true_label = sample.get("label", "?")
        has_dcp    = int(sample.get("has_dcp", 0))

        try:
            svp_tensor = load_image_tensor(
                sample["svp_path"], data_root, device
            )
            svp_disp = tensor_to_display(svp_tensor)

            if has_dcp and "dcp_path" in sample and sample["dcp_path"]:
                dcp_tensor = load_image_tensor(
                    sample["dcp_path"], data_root, device
                )
                dcp_disp = tensor_to_display(dcp_tensor)
            else:
                dcp_tensor = torch.zeros_like(svp_tensor)
                dcp_disp   = None
                has_dcp    = 0

            result = compute_gradcam(
                model=model,
                svp_tensor=svp_tensor,
                dcp_tensor=dcp_tensor,
                has_dcp_val=has_dcp,
                device=device,
            )

            overlay_svp = overlay_heatmap(svp_disp, result["cam_svp"], alpha, colormap)
            correct     = result["pred_name"] == true_label
            verdict     = "✓" if correct else f"✗→{result['pred_name']}"
            prob_pred   = float(result["probs"][result["pred_idx"]])

            if result["cam_dcp"] is not None and dcp_disp is not None:
                overlay_dcp = overlay_heatmap(dcp_disp, result["cam_dcp"], alpha, colormap)
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(svp_disp,    cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("SVP", fontsize=10)
                axes[1].imshow(overlay_svp)
                axes[1].set_title("SVP Grad-CAM", fontsize=10)
                axes[2].imshow(dcp_disp,    cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("DCP", fontsize=10)
                axes[3].imshow(overlay_dcp)
                axes[3].set_title("DCP Grad-CAM", fontsize=10)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(svp_disp, cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("SVP", fontsize=10)
                axes[1].imshow(overlay_svp)
                axes[1].set_title("SVP Grad-CAM", fontsize=10)

            for ax in np.array(axes).flat:
                ax.set_xticks([])
                ax.set_yticks([])

            title_color = "#2ecc71" if correct else "#e74c3c"
            fig.suptitle(
                f"{true_label} | pred: {result['pred_name']} ({prob_pred:.3f}) {verdict} | {sample_id}",
                fontsize=10, y=1.02, color=title_color,
            )
            plt.tight_layout()

            out_path = save_dir / f"{sample_id}_t-{true_label}_p-{result['pred_name']}.png"
            plt.savefig(out_path, dpi=130, bbox_inches="tight")
            logger.info(f"GradCAM uložený → {out_path}")

            if show:
                plt.show()
            else:
                plt.close()

            print(
                f"  [{i+1}/{len(image_list)}] {sample_id} | "
                f"true={true_label} | pred={result['pred_name']} ({prob_pred:.3f}) | {verdict}"
            )

        except Exception as e:
            logger.error(f"GradCAM chyba pre {sample_id}: {e}")
            plt.close("all")
            print(f"  [{i+1}/{len(image_list)}] ❌ {sample_id}: {e}")
