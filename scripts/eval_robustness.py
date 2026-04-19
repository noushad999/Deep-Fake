"""
Robustness Evaluation: Image Degradation Tests
===============================================
Tests model robustness against real-world image degradations:
  - JPEG compression (quality 90 → 30)
  - Gaussian noise (σ = 5, 10, 20, 40 px)
  - Gaussian blur (σ = 0.5, 1.0, 2.0)
  - Downscaling + upscaling (×0.5, ×0.25)

CVPR significance: Shows the model is practically deployable, not just
a benchmark overfitter. Frequency stream should be most affected by JPEG.

Usage:
    python scripts/eval_robustness.py --checkpoint checkpoints/best_model.pth
    python scripts/eval_robustness.py --checkpoint checkpoints/best_model.pth --data-dir data/test
"""
import argparse
import io
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from utils.utils import get_device


# ---------------------------------------------------------------------------
# Degradation functions (operate on uint8 numpy HWC images)
# ---------------------------------------------------------------------------

def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = max(3, int(sigma * 6) | 1)  # odd kernel
    return cv2.GaussianBlur(img, (k, k), sigma)


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


DEGRADATION_SUITE = [
    ("clean",         lambda x: x),
    ("jpeg_q90",      lambda x: jpeg_compress(x, 90)),
    ("jpeg_q70",      lambda x: jpeg_compress(x, 70)),
    ("jpeg_q50",      lambda x: jpeg_compress(x, 50)),
    ("jpeg_q30",      lambda x: jpeg_compress(x, 30)),
    ("noise_s5",      lambda x: gaussian_noise(x, 5)),
    ("noise_s10",     lambda x: gaussian_noise(x, 10)),
    ("noise_s20",     lambda x: gaussian_noise(x, 20)),
    ("noise_s40",     lambda x: gaussian_noise(x, 40)),
    ("blur_s0.5",     lambda x: gaussian_blur(x, 0.5)),
    ("blur_s1.0",     lambda x: gaussian_blur(x, 1.0)),
    ("blur_s2.0",     lambda x: gaussian_blur(x, 2.0)),
    ("downscale_0.5", lambda x: downscale(x, 0.5)),
    ("downscale_0.25",lambda x: downscale(x, 0.25)),
]


# ---------------------------------------------------------------------------
# Synthetic test set builder (when no real data available)
# ---------------------------------------------------------------------------

def make_synthetic_batch(n: int = 200, img_size: int = 256, seed: int = 42):
    """
    Build a synthetic test batch with identifiable real/fake patterns.
    Real: natural-looking random images. Fake: contain frequency artifacts (grid).
    Labels are 100% reliable since we generate them.
    """
    rng = np.random.RandomState(seed)
    images, labels = [], []
    for i in range(n):
        label = i % 2
        img = rng.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        if label == 1:
            # inject grid artifact typical of GAN/diffusion outputs
            img[::8, :, :] = np.clip(img[::8, :, :].astype(int) + 60, 0, 255).astype(np.uint8)
            img[:, ::8, :] = np.clip(img[:, ::8, :].astype(int) + 60, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(label)
    return images, labels


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray, img_size: int = 256) -> torch.Tensor:
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_degradation(
    model: nn.Module,
    images: List[np.ndarray],
    labels: List[int],
    degradation_fn: Callable,
    device: torch.device,
    batch_size: int = 32,
    img_size: int = 256,
) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    all_scores = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        tensors = []
        for img in batch_imgs:
            degraded = degradation_fn(img)
            tensors.append(preprocess(degraded, img_size))
        batch = torch.stack(tensors).to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(batch)
        scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy().tolist()
        all_scores.extend(scores)

    labels_arr = np.array(labels)
    scores_arr = np.array(all_scores)
    acc = 100.0 * ((scores_arr > 0.5).astype(int) == labels_arr).mean()
    try:
        auc = roc_auc_score(labels_arr, scores_arr) * 100.0
    except Exception:
        auc = 0.0
    return auc, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default=None,
                        help="Directory with test images (real/ and fake/ subdirs). "
                             "If not provided, uses synthetic test data.")
    parser.add_argument("--n-synthetic", type=int, default=400,
                        help="Number of synthetic test images (used if --data-dir not set)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = get_device()
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    print("=" * 65)
    print("ROBUSTNESS EVALUATION")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")
    print("=" * 65)

    # Load model
    model = MultiStreamDeepfakeDetector(pretrained_backbones=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Build test set
    if args.data_dir:
        data_dir = Path(args.data_dir)
        IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
        real_files = [p for p in (data_dir / "real").rglob("*") if p.suffix.lower() in IMG_EXT]
        fake_files = [p for p in (data_dir / "fake").rglob("*") if p.suffix.lower() in IMG_EXT]
        rng = np.random.RandomState(args.seed)
        n = min(len(real_files), len(fake_files), 500)
        real_files = [real_files[i] for i in rng.choice(len(real_files), n, replace=False)]
        fake_files = [fake_files[i] for i in rng.choice(len(fake_files), n, replace=False)]
        images, labels = [], []
        for p in real_files:
            images.append(np.array(Image.open(p).convert("RGB")))
            labels.append(0)
        for p in fake_files:
            images.append(np.array(Image.open(p).convert("RGB")))
            labels.append(1)
        print(f"  Test set: {len(real_files)} real + {len(fake_files)} fake from {data_dir}")
    else:
        images, labels = make_synthetic_batch(args.n_synthetic, args.img_size, args.seed)
        print(f"  Test set: {len(images)} synthetic images (no real data provided)")

    # Run all degradations
    print(f"\n  {'Degradation':<20} {'AUC':>8}  {'Acc':>8}")
    print("  " + "-" * 42)

    results = {}
    for name, fn in tqdm(DEGRADATION_SUITE, desc="Running degradations"):
        auc, acc = evaluate_degradation(
            model, images, labels, fn, device,
            batch_size=args.batch_size, img_size=args.img_size)
        results[name] = {"auc": auc, "acc": acc}
        tag = " <- baseline" if name == "clean" else ""
        print(f"  {name:<20} {auc:>7.2f}%  {acc:>7.2f}%{tag}")

    # Relative AUC drop from clean
    clean_auc = results["clean"]["auc"]
    print(f"\n  {'Degradation':<20} {'dAUC':>8}  (vs clean={clean_auc:.2f}%)")
    print("  " + "-" * 35)
    for name, v in results.items():
        if name == "clean":
            continue
        delta = v["auc"] - clean_auc
        print(f"  {name:<20} {delta:>+7.2f}%")

    # Save results
    out_path = args.out or str(ckpt_path.parent / "robustness_results.txt")
    with open(out_path, "w") as fh:
        fh.write(f"Checkpoint: {ckpt_path}\n")
        fh.write(f"Clean AUC: {clean_auc:.4f}%\n\n")
        fh.write(f"{'Degradation':<20}  {'AUC':>8}  {'Acc':>8}  {'dAUC':>8}\n")
        fh.write("-" * 52 + "\n")
        for name, v in results.items():
            delta = v["auc"] - clean_auc if name != "clean" else 0.0
            fh.write(f"{name:<20}  {v['auc']:>7.2f}%  {v['acc']:>7.2f}%  {delta:>+7.2f}%\n")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
