"""
Test-Time Augmentation (TTA) Evaluation
=========================================
TTA averages predictions over multiple augmented views of each test image.
Common in top-performing CVPR detection papers — often gives +1-3% AUC.

Augmentations used:
  - Original (no augmentation)
  - Horizontal flip
  - 90/180/270-degree rotations
  - Light JPEG compression (q=90)
  - Center crop (80% of image) + resize

Usage:
    python scripts/eval_tta.py --checkpoint checkpoints/best_model.pth --data-dir data/test
    python scripts/eval_tta.py --checkpoint checkpoints/best_model.pth  # synthetic test
"""
import argparse
import io
import sys
from pathlib import Path

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
# TTA transforms
# ---------------------------------------------------------------------------

def tta_transforms(img: np.ndarray):
    """Returns list of augmented versions of img (uint8 HWC)."""
    views = [img]
    # Horizontal flip
    views.append(np.fliplr(img).copy())
    # 90/180/270 rotation
    for k in [1, 2, 3]:
        views.append(np.rot90(img, k).copy())
    # Center crop 80%
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    ch, cw = int(h * 0.8), int(w * 0.8)
    crop = img[cy - ch//2:cy + ch//2, cx - cw//2:cx + cw//2]
    views.append(cv2.resize(crop, (w, h)))
    # Light JPEG
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=90)
    buf.seek(0)
    views.append(np.array(Image.open(buf).convert("RGB")))
    return views


def preprocess(img: np.ndarray, img_size: int = 256) -> torch.Tensor:
    img = cv2.resize(img.astype(np.uint8), (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


@torch.no_grad()
def predict_tta(model, img: np.ndarray, device, img_size=256) -> float:
    """Predict fake probability using TTA (mean of all views)."""
    views = tta_transforms(img)
    tensors = torch.stack([preprocess(v, img_size) for v in views]).to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits, _ = model(tensors)
    probs = torch.sigmoid(logits.squeeze(-1))
    return float(probs.mean().cpu())


def make_synthetic_batch(n=200, img_size=256, seed=42):
    rng = np.random.RandomState(seed)
    images, labels = [], []
    for i in range(n):
        label = i % 2
        img = rng.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        if label == 1:
            img[::8, :, :] = np.clip(img[::8, :, :].astype(int) + 60, 0, 255).astype(np.uint8)
        images.append(img)
        labels.append(label)
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-synthetic", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    ckpt_path = Path(args.checkpoint)

    model = MultiStreamDeepfakeDetector(pretrained_backbones=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
        real_files = sorted(p for p in (data_dir / "real").rglob("*") if p.suffix.lower() in IMG_EXT)
        fake_files = sorted(p for p in (data_dir / "fake").rglob("*") if p.suffix.lower() in IMG_EXT)
        rng = np.random.RandomState(args.seed)
        n = min(len(real_files), len(fake_files), 500)
        real_files = [real_files[i] for i in rng.choice(len(real_files), n, replace=False)]
        fake_files = [fake_files[i] for i in rng.choice(len(fake_files), n, replace=False)]
        images = [np.array(Image.open(p).convert("RGB")) for p in real_files + fake_files]
        labels = [0] * len(real_files) + [1] * len(fake_files)
    else:
        images, labels = make_synthetic_batch(args.n_synthetic, args.img_size, args.seed)

    from sklearn.metrics import roc_auc_score

    # Without TTA
    scores_plain = []
    for img in tqdm(images, desc="Plain inference"):
        t = preprocess(img, args.img_size).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = model(t)
        scores_plain.append(float(torch.sigmoid(logits.squeeze()).cpu()))

    # With TTA
    scores_tta = []
    for img in tqdm(images, desc="TTA inference"):
        scores_tta.append(predict_tta(model, img, device, args.img_size))

    labels_arr = np.array(labels)
    auc_plain = roc_auc_score(labels_arr, np.array(scores_plain)) * 100
    auc_tta   = roc_auc_score(labels_arr, np.array(scores_tta))   * 100
    acc_plain = 100.0 * ((np.array(scores_plain) > 0.5).astype(int) == labels_arr).mean()
    acc_tta   = 100.0 * ((np.array(scores_tta)   > 0.5).astype(int) == labels_arr).mean()

    print("\n" + "=" * 50)
    print("TTA RESULTS")
    print("=" * 50)
    print(f"  Plain  AUC: {auc_plain:.2f}%  Acc: {acc_plain:.2f}%")
    print(f"  TTA    AUC: {auc_tta:.2f}%  Acc: {acc_tta:.2f}%")
    print(f"  Gain:  AUC: {auc_tta - auc_plain:+.2f}%  Acc: {acc_tta - acc_plain:+.2f}%")

    out_path = ckpt_path.parent / "tta_results.txt"
    with open(out_path, "w") as fh:
        fh.write(f"Checkpoint: {ckpt_path}\n")
        fh.write(f"Plain  AUC: {auc_plain:.4f}%  Acc: {acc_plain:.4f}%\n")
        fh.write(f"TTA    AUC: {auc_tta:.4f}%  Acc: {acc_tta:.4f}%\n")
        fh.write(f"Gain   AUC: {auc_tta - auc_plain:+.4f}%\n")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
