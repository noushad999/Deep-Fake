"""
Feature Visualization for CVPR Paper
======================================
Produces publication-quality figures:
  1. t-SNE of per-stream features (real vs fake clusters)
  2. Stream attention weights distribution (which stream model trusts most)
  3. Inter-stream correlation heatmap
  4. Prediction calibration curve (ECE)

Usage:
    python scripts/visualize_features.py \
        --checkpoint checkpoints/best_model.pth \
        --data-dir data --output-dir logs/feature_viz
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from utils.utils import get_device


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device, max_batches=100):
    model.eval()
    feats = {"spatial": [], "frequency": [], "semantic": [], "fused": []}
    attn_weights_list = []
    labels_all = []
    scores_all = []

    for i, batch in enumerate(tqdm(loader, desc="Extracting features", leave=False)):
        if i >= max_batches:
            break
        imgs = batch["image"].to(device)
        lbls = batch["label"].numpy()

        logits, features = model(imgs, return_features=True)
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        for k in feats:
            feats[k].append(features[k].cpu().numpy())
        labels_all.extend(lbls.tolist())
        scores_all.extend(probs.tolist())

        # Extract attention weights from fusion module
        s = model.fusion.cross_stream_attn.spatial_proj(features["spatial"]).unsqueeze(1)
        f = model.fusion.cross_stream_attn.freq_proj(features["frequency"]).unsqueeze(1)
        t = model.fusion.cross_stream_attn.semantic_proj(features["semantic"]).unsqueeze(1)
        tokens = torch.cat([s, f, t], dim=1)
        _, aw = model.fusion.cross_stream_attn.attn(tokens, tokens, tokens)
        attn_weights_list.append(aw.cpu().numpy())  # [B, 3, 3]

    for k in feats:
        feats[k] = np.concatenate(feats[k], axis=0)
    attn_weights = np.concatenate(attn_weights_list, axis=0)  # [N, 3, 3]
    return feats, attn_weights, np.array(labels_all), np.array(scores_all)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_tsne(feats, labels, output_dir):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("sklearn required: pip install scikit-learn"); return

    stream_names = ["spatial", "frequency", "semantic", "fused"]
    colors = {0: "#2196F3", 1: "#F44336"}
    color_map = [colors[int(l)] for l in labels]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, name in zip(axes, stream_names):
        data = feats[name]
        if data.shape[0] > 2000:
            idx = np.random.choice(data.shape[0], 2000, replace=False)
            data = data[idx]
            cm = [color_map[i] for i in idx]
            lbl = labels[idx]
        else:
            cm = color_map
            lbl = labels

        emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(data)
        ax.scatter(emb[:, 0], emb[:, 1], c=cm, s=8, alpha=0.6)
        ax.set_title(f"{name.capitalize()} Stream", fontsize=11)
        ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2196F3", label="Real"),
                        Patch(color="#F44336", label="Fake")],
               loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("t-SNE Feature Visualization per Stream", fontsize=13, y=1.02)
    plt.tight_layout()

    out = output_dir / "tsne_streams.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_attention_weights(attn_weights, labels, output_dir):
    """Which stream gets most attention for real vs fake images?"""
    stream_names = ["Spatial", "Frequency", "Semantic"]
    # Mean attention received by each stream (column-wise mean of attn matrix)
    # attn_weights: [N, 3, 3], axis 2 = attended-to stream
    mean_attn = attn_weights.mean(axis=1)  # [N, 3]

    real_attn = mean_attn[labels == 0].mean(axis=0)
    fake_attn = mean_attn[labels == 1].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(3)
    width = 0.35
    axes[0].bar(x - width/2, real_attn, width, label="Real", color="#2196F3", alpha=0.8)
    axes[0].bar(x + width/2, fake_attn, width, label="Fake", color="#F44336", alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(stream_names)
    axes[0].set_ylabel("Mean Attention Weight")
    axes[0].set_title("Stream Attention: Real vs Fake")
    axes[0].legend()

    # Heatmap of average attention matrix
    avg_attn_mat = attn_weights.mean(axis=0)  # [3, 3]
    im = axes[1].imshow(avg_attn_mat, cmap="Blues", vmin=0, vmax=avg_attn_mat.max())
    axes[1].set_xticks(range(3)); axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(stream_names); axes[1].set_yticklabels(stream_names)
    axes[1].set_xlabel("Key (attends to)"); axes[1].set_ylabel("Query (from)")
    axes[1].set_title("Average Cross-Stream Attention Matrix")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{avg_attn_mat[i,j]:.3f}", ha="center",
                         va="center", fontsize=10, color="black")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    out = output_dir / "stream_attention.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_calibration(labels, scores, output_dir, n_bins=10):
    """Reliability diagram + Expected Calibration Error (ECE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_accs.append(labels[mask].mean())
        bin_confs.append(scores[mask].mean())
        bin_counts.append(mask.sum())

    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    ece = (np.abs(bin_accs - bin_confs) * bin_counts / bin_counts.sum()).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7,
           color="#4CAF50", label="Model", edgecolor="black")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean Confidence", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title(f"Calibration Curve (ECE = {ece:.4f})", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    out = output_dir / "calibration.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    print(f"  ECE = {ece:.4f} (lower is better, 0=perfect)")
    return float(ece)


def plot_inter_stream_correlation(feats, output_dir):
    """Correlation between stream feature norms — shows orthogonality."""
    stream_names = ["Spatial", "Frequency", "Semantic"]
    keys = ["spatial", "frequency", "semantic"]
    norms = np.stack([np.linalg.norm(feats[k], axis=1) for k in keys], axis=1)

    corr = np.corrcoef(norms.T)  # [3, 3]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(stream_names); ax.set_yticklabels(stream_names)
    ax.set_title("Inter-Stream Feature Correlation\n(Low = diverse, complementary streams)",
                 fontsize=11)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{corr[i,j]:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    out = output_dir / "stream_correlation.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print analysis
    off_diag = [corr[i,j] for i in range(3) for j in range(3) if i != j]
    print(f"  Mean off-diagonal correlation: {np.mean(np.abs(off_diag)):.4f}")
    print("  (Low value proves streams capture orthogonal/diverse features)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="custom", choices=["custom", "ffpp"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--output-dir", default="logs/feature_viz")
    args = parser.parse_args()

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MultiStreamDeepfakeDetector(pretrained_backbones=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device).eval()

    if args.dataset == "ffpp":
        from data.ffpp_dataset import FFPPDataset, get_ffpp_transforms
        from torch.utils.data import DataLoader
        ds = FFPPDataset(args.data_dir, "test", transform=get_ffpp_transforms("test"))
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    else:
        from data.dataset import DeepfakeDataset, get_transforms
        from torch.utils.data import DataLoader
        ds = DeepfakeDataset(args.data_dir, "test", transform=get_transforms("test"))
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    print("Extracting features...")
    feats, attn_weights, labels, scores = extract_features(
        model, loader, device, max_batches=args.max_batches
    )
    print(f"  Samples: {len(labels)} | Real: {(labels==0).sum()} | Fake: {(labels==1).sum()}")

    plot_tsne(feats, labels, output_dir)
    plot_attention_weights(attn_weights, labels, output_dir)
    ece = plot_calibration(labels, scores, output_dir)
    plot_inter_stream_correlation(feats, output_dir)

    print(f"\nAll figures → {output_dir}")


if __name__ == "__main__":
    main()
