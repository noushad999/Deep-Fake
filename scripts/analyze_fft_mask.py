"""
FFT Mask Analysis & Frequency Visualization
=============================================
Provides deep analysis of the Learnable FFT Mask (LearnableFFTMask)
in the FreqBlender stream — a key novel component.

Analysis includes:
  1. Visualize the 2D learned frequency mask (per channel, averaged)
  2. Band weight values (Low / Mid-Low / Mid-High / High)
  3. Frequency spectrum comparison: real vs. fake faces
  4. Most discriminative frequency regions (via gradient × mask)
  5. Saves publication-quality figures for CVPR paper

This analysis motivates WHY frequency information helps and what
manipulation artifacts the model learns to detect in the frequency domain.

Usage:
    python scripts/analyze_fft_mask.py \\
        --checkpoint checkpoints/best_model.pth \\
        --data-dir data \\
        --output-dir logs/fft_analysis
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from models.freq_stream import LearnableFFTMask
from utils.utils import get_device


# ---------------------------------------------------------------------------
# Spectrum utilities
# ---------------------------------------------------------------------------

def compute_log_spectrum(image_np: np.ndarray) -> np.ndarray:
    """
    Compute log-magnitude FFT spectrum of a grayscale image.

    Args:
        image_np: [H, W] or [H, W, C] numpy array, float32, [0,1]

    Returns:
        log_spectrum: [H, W] centered log-magnitude spectrum
    """
    if image_np.ndim == 3:
        gray = image_np.mean(axis=2)
    else:
        gray = image_np

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_mag = np.log1p(magnitude)
    return log_mag


def average_spectra(data_loader, device, n_batches=50):
    """
    Compute average frequency spectra for real and fake images separately.
    Returns (real_spectrum, fake_spectrum) averaged across samples.
    """
    real_specs = []
    fake_specs = []

    for i, batch in enumerate(tqdm(data_loader, desc="Computing spectra",
                                   leave=False)):
        if i >= n_batches:
            break
        images = batch["image"]  # [B, 3, H, W], normalized
        labels = batch["label"]

        # Denormalize: roughly reverse ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_denorm = (images * std + mean).clamp(0, 1)

        for j in range(images.size(0)):
            img_np = images_denorm[j].permute(1, 2, 0).numpy()
            spec = compute_log_spectrum(img_np)
            if labels[j].item() == 0:
                real_specs.append(spec)
            else:
                fake_specs.append(spec)

    real_mean = np.mean(real_specs, axis=0) if real_specs else None
    fake_mean = np.mean(fake_specs, axis=0) if fake_specs else None
    return real_mean, fake_mean


# ---------------------------------------------------------------------------
# Mask analysis
# ---------------------------------------------------------------------------

def extract_fft_mask_info(model: MultiStreamDeepfakeDetector):
    """
    Extract learned FFT mask parameters from the frequency stream.

    Returns dict with:
      - 'mask_2d':      [H, W] averaged mask (over 3 channels)
      - 'band_weights': [4] softmax-normalized band weights
      - 'mask_per_channel': [3, H, W] per-channel masks
    """
    fft_mask_module: LearnableFFTMask = model.freq_stream.fft_mask

    with torch.no_grad():
        mask_raw = fft_mask_module.mask.cpu()          # [3, H, W]
        band_weights_raw = fft_mask_module.band_weights.cpu()  # [4]

    mask_np = mask_raw.numpy()
    mask_2d = mask_np.mean(axis=0)                     # [H, W]

    # Band weights: apply sigmoid (same as forward) and normalize for display
    band_weights = torch.sigmoid(band_weights_raw).numpy()

    return {
        "mask_2d": mask_2d,
        "mask_per_channel": mask_np,
        "band_weights": band_weights,
        "band_names": ["Low (0-25%)", "Mid-Low (25-50%)",
                       "Mid-High (50-75%)", "High (75-100%)"],
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_fft_mask(mask_info: dict, output_dir: Path):
    """Figure 1: Learned 2D FFT mask heatmap."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Panel 1: Average mask
    ax = axes[0]
    im = ax.imshow(mask_info["mask_2d"], cmap="hot", aspect="auto")
    ax.set_title("Learned Frequency Mask\n(Averaged over RGB channels)", fontsize=11)
    ax.set_xlabel("Frequency (horizontal)")
    ax.set_ylabel("Frequency (vertical)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 2: Per-channel masks (R, G, B)
    channel_names = ["Red channel", "Green channel", "Blue channel"]
    cmap = "seismic"
    for i in range(3):
        ax = axes[i + 1]
        ch_mask = mask_info["mask_per_channel"][i]
        vmax = max(abs(ch_mask.min()), abs(ch_mask.max()))
        im = ax.imshow(ch_mask, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Channel Mask: {channel_names[i]}", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out = output_dir / "fft_mask_2d.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_band_weights(mask_info: dict, output_dir: Path):
    """Figure 2: Frequency band weights bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))

    weights = mask_info["band_weights"]
    names = mask_info["band_names"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    bars = ax.bar(range(4), weights, color=colors, edgecolor="black",
                  linewidth=0.8, width=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Learned Band Weight (σ)", fontsize=11)
    ax.set_title("Frequency Band Attention Weights\nLearned by FreqBlender Stream",
                 fontsize=12)
    ax.set_ylim(0, 1.1)

    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{w:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = output_dir / "band_weights.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_spectrum_comparison(
    real_spectrum: np.ndarray,
    fake_spectrum: np.ndarray,
    output_dir: Path,
):
    """Figure 3: Real vs. fake frequency spectra and difference map."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    # Real spectrum
    ax = axes[0]
    im = ax.imshow(normalize(real_spectrum), cmap="viridis", aspect="auto")
    ax.set_title("Real Faces\nAverage Log-Spectrum", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Fake spectrum
    ax = axes[1]
    im = ax.imshow(normalize(fake_spectrum), cmap="viridis", aspect="auto")
    ax.set_title("Fake Faces (AI-Generated)\nAverage Log-Spectrum", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Difference: |fake - real|
    diff = np.abs(fake_spectrum - real_spectrum)
    ax = axes[2]
    im = ax.imshow(normalize(diff), cmap="hot", aspect="auto")
    ax.set_title("Spectral Difference\n|Fake − Real| (manipulation artifacts)",
                 fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out = output_dir / "spectrum_comparison.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_mask_vs_discriminative_regions(
    mask_info: dict,
    real_spectrum: np.ndarray,
    fake_spectrum: np.ndarray,
    output_dir: Path,
):
    """Figure 4: Overlay learned mask on spectral difference."""
    diff = np.abs(fake_spectrum - real_spectrum)

    # Resize mask to match spectrum dimensions if needed
    mask = mask_info["mask_2d"]
    if mask.shape != diff.shape:
        import cv2
        mask = cv2.resize(mask, (diff.shape[1], diff.shape[0]))

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    diff_norm = normalize(diff)
    mask_norm = normalize(mask)
    overlay = diff_norm * mask_norm

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.imshow(mask_norm, cmap="hot", aspect="auto")
    ax.set_title("Learned FFT Mask", fontsize=11)
    ax.axis("off")

    ax = axes[1]
    ax.imshow(diff_norm, cmap="hot", aspect="auto")
    ax.set_title("Spectral Difference\n(Manipulation Artifacts)", fontsize=11)
    ax.axis("off")

    ax = axes[2]
    ax.imshow(overlay, cmap="hot", aspect="auto")
    ax.set_title("Mask × Difference\n(Model Focus Alignment)", fontsize=11)
    ax.axis("off")

    correlation = np.corrcoef(mask_norm.flatten(), diff_norm.flatten())[0, 1]
    fig.suptitle(
        f"Mask–Artifact Alignment (Pearson r = {correlation:.3f})",
        fontsize=13, y=1.02
    )

    plt.tight_layout()
    out = output_dir / "mask_vs_artifacts.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    return correlation


def print_mask_statistics(mask_info: dict):
    mask = mask_info["mask_2d"]
    bw = mask_info["band_weights"]

    print("\n" + "=" * 60)
    print("FFT MASK STATISTICS")
    print("=" * 60)
    print(f"Mask shape:          {mask.shape}")
    print(f"Mask mean:           {mask.mean():.4f}")
    print(f"Mask std:            {mask.std():.4f}")
    print(f"Mask min/max:        {mask.min():.4f} / {mask.max():.4f}")

    h, w = mask.shape
    # Analyze center (low-freq) vs. periphery (high-freq)
    r_in = min(h, w) // 8
    center_val = mask[h//2 - r_in:h//2 + r_in, w//2 - r_in:w//2 + r_in].mean()
    periphery_mask = np.ones_like(mask, dtype=bool)
    periphery_mask[h//2 - r_in:h//2 + r_in, w//2 - r_in:w//2 + r_in] = False
    periphery_val = mask[periphery_mask].mean()

    print(f"\nCenter (low-freq):   {center_val:.4f}")
    print(f"Periphery (high-f):  {periphery_val:.4f}")
    print(f"High/Low ratio:      {periphery_val / max(center_val, 1e-8):.3f}")

    print("\nBand weights (σ(w)):")
    for name, w in zip(mask_info["band_names"], bw):
        bar = "█" * int(w * 20)
        print(f"  {name:<20} {w:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze learned FFT mask for CVPR paper figures"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained MultiStream model checkpoint")
    parser.add_argument("--data-dir", required=True,
                        help="Data directory for spectrum comparison")
    parser.add_argument("--dataset", default="custom",
                        choices=["custom", "ffpp"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-batches", type=int, default=100,
                        help="Number of batches for spectrum estimation")
    parser.add_argument("--output-dir", default="logs/fft_analysis")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model = MultiStreamDeepfakeDetector(pretrained_backbones=False)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print(f"Loaded model from: {args.checkpoint}")

    # 1. Extract and visualize FFT mask
    mask_info = extract_fft_mask_info(model)
    print_mask_statistics(mask_info)
    plot_fft_mask(mask_info, output_dir)
    plot_band_weights(mask_info, output_dir)

    # 2. Load test data for spectrum comparison
    if args.dataset == "ffpp":
        from data.ffpp_dataset import FFPPDataset, get_ffpp_transforms
        from torch.utils.data import DataLoader
        test_ds = FFPPDataset(
            root_dir=args.data_dir, split="test",
            transform=get_ffpp_transforms("test"),
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
    else:
        from data.dataset import DeepfakeDataset, get_transforms
        from torch.utils.data import DataLoader
        test_ds = DeepfakeDataset(
            data_root=args.data_dir, split="test",
            transform=get_transforms("test"),
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)

    # 3. Compute average spectra
    print("\nComputing average frequency spectra...")
    real_spec, fake_spec = average_spectra(
        test_loader, device, n_batches=args.n_batches
    )

    if real_spec is not None and fake_spec is not None:
        plot_spectrum_comparison(real_spec, fake_spec, output_dir)
        corr = plot_mask_vs_discriminative_regions(
            mask_info, real_spec, fake_spec, output_dir
        )
        print(f"\nMask-Artifact Pearson correlation: {corr:.4f}")
        print("  (High value → mask correctly focuses on manipulation artifacts)")
    else:
        print("Warning: Could not compute spectra (insufficient data).")

    print(f"\nAll figures saved to: {output_dir}")
    print("Use fft_mask_2d.pdf, band_weights.pdf, spectrum_comparison.pdf "
          "in your CVPR paper.")


if __name__ == "__main__":
    main()
