"""
GradCAM++ Visualization for Deepfake Detection
================================================
Generates class activation maps showing WHERE the model looks
to distinguish real vs fake images. Key figure for CVPR papers.

Shows:
  - Which spatial regions trigger the fake decision
  - How each stream attends differently (spatial vs freq vs semantic)
  - Side-by-side: original | CAM overlay | stream-specific CAMs

Usage:
    python scripts/visualize_gradcam.py --checkpoint checkpoints/best_model.pth
    python scripts/visualize_gradcam.py --checkpoint checkpoints/best_model.pth --img path/to/image.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from utils.utils import get_device


# ---------------------------------------------------------------------------
# GradCAM++ implementation
# ---------------------------------------------------------------------------

class GradCAMPlusPlus:
    """
    GradCAM++ for any target layer.
    Grad-weighted class activation maps with second-order gradient correction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._feat = None
        self._grad = None
        self._hooks = []
        self._register()

    def _register(self):
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_feat))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_grad))

    def _save_feat(self, module, inp, out):
        self._feat = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self._grad = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        logits, _ = self.model(x)
        score = logits.squeeze()
        score.backward()

        feats = self._feat   # [B, C, H, W]
        grads = self._grad   # [B, C, H, W]

        if feats is None or grads is None or feats.dim() != 4:
            return np.ones((x.shape[2], x.shape[3]))

        # GradCAM++: alpha = grads^2 / (2*grads^2 + feats * grads^3)
        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        denom = 2.0 * grads_sq + (feats * grads_cu).sum(dim=(2, 3), keepdim=True)
        alpha = grads_sq / (denom + 1e-7)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3))  # [B, C]

        cam = (weights.unsqueeze(-1).unsqueeze(-1) * feats).sum(dim=1)  # [B, H, W]
        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        return cam

    def remove(self):
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# Overlay helper
# ---------------------------------------------------------------------------

def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay jet heatmap on RGB image (both uint8 HWC)."""
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip(img_rgb * (1 - alpha) + heatmap * alpha, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(path_or_array, img_size=256):
    from PIL import Image
    if isinstance(path_or_array, np.ndarray):
        img = path_or_array
    else:
        img = np.array(Image.open(path_or_array).convert("RGB"))
    img_orig = cv2.resize(img, (img_size, img_size))
    # Preprocess
    norm = img_orig.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    norm = (norm - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_orig, tensor


def make_synthetic_images(n=4, img_size=256, seed=42):
    rng = np.random.RandomState(seed)
    images, labels = [], []
    for i in range(n):
        label = i % 2
        img = rng.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        if label == 1:
            img[::8, :, :] = np.clip(img[::8, :, :].astype(int) + 80, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(label)
    return images, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--img", default=None, help="Path to a single image")
    parser.add_argument("--img-dir", default=None, help="Dir with real/ and fake/ subdirs")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--out-dir", default="results/gradcam")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = MultiStreamDeepfakeDetector(pretrained_backbones=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    # Identify target layers (last conv of each stream)
    target_layers = {
        "spatial":  model.spatial_stream,
        "frequency": model.freq_stream,
        "semantic": model.semantic_stream,
    }

    # Find last Conv2d in each stream
    def find_last_conv(module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    cam_targets = {}
    for name, stream in target_layers.items():
        layer = find_last_conv(stream)
        if layer is not None:
            cam_targets[name] = GradCAMPlusPlus(model, layer)

    # Prepare images
    if args.img:
        images = [np.array(__import__("PIL").Image.open(args.img).convert("RGB"))]
        labels = [-1]  # unknown
    elif args.img_dir:
        from PIL import Image
        data_dir = Path(args.img_dir)
        IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
        real_files = sorted(p for p in (data_dir / "real").rglob("*") if p.suffix.lower() in IMG_EXT)
        fake_files = sorted(p for p in (data_dir / "fake").rglob("*") if p.suffix.lower() in IMG_EXT)
        n = min(args.n_samples // 2, len(real_files), len(fake_files))
        images = [np.array(Image.open(p).convert("RGB")) for p in real_files[:n] + fake_files[:n]]
        labels = [0] * n + [1] * n
    else:
        images, labels = make_synthetic_images(args.n_samples, args.img_size, args.seed)

    print(f"Generating GradCAM++ for {len(images)} images -> {out_dir}/")

    for idx, (img, label) in enumerate(zip(images, labels)):
        img_orig, tensor = load_image(img, args.img_size)
        tensor = tensor.to(device).requires_grad_(False)

        # Prediction
        with torch.no_grad():
            logits, _ = model(tensor)
        prob = float(torch.sigmoid(logits.squeeze()).cpu())
        pred = "fake" if prob > 0.5 else "real"
        true_label = {0: "real", 1: "fake", -1: "?"}[label]

        # Build figure: original + per-stream CAM overlays
        n_cols = 1 + len(cam_targets)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        axes[0].imshow(img_orig)
        axes[0].set_title(f"True: {true_label}\nPred: {pred} ({prob:.2f})", fontsize=9)
        axes[0].axis("off")

        for col, (stream_name, gradcam) in enumerate(cam_targets.items(), start=1):
            tensor_req = tensor.clone().requires_grad_(True)
            cam = gradcam(tensor_req)
            overlay = overlay_cam(img_orig, cam)
            axes[col].imshow(overlay)
            axes[col].set_title(f"{stream_name} stream\nGradCAM++", fontsize=9)
            axes[col].axis("off")

        fig.suptitle(f"Sample {idx} | pred={pred} ({prob:.3f})", fontsize=10)
        plt.tight_layout()
        out_path = out_dir / f"gradcam_{idx:03d}_{true_label}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    for gc in cam_targets.values():
        gc.remove()

    print(f"\nAll GradCAM++ figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
