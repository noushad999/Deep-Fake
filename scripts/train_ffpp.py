"""
FaceForensics++ Training Script — CVPR Standard Protocol
=========================================================
Trains the Multi-Stream Deepfake Detector on FaceForensics++ (c23)
following the standard protocol used in deepfake detection papers:

  - Compression: c23 (high-quality, most common)
  - Manipulation types: Deepfakes, Face2Face, FaceSwap, NeuralTextures
  - Official train/val/test splits (720/140/140 videos)
  - 32 frames sampled uniformly per video during training
  - All frames during test evaluation
  - Reports: frame-level AUC, video-level AUC (mean aggregation)
  - Per-manipulation-type breakdown

Usage:
    python scripts/train_ffpp.py --data-dir /path/to/FaceForensics++
    python scripts/train_ffpp.py --data-dir /path/to/FF++ --compression c40
    python scripts/train_ffpp.py --data-dir /path/to/FF++ --model xception
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.ffpp_dataset import (
    MANIPULATION_TYPES,
    create_ffpp_dataloaders,
    aggregate_video_predictions,
)
from models.full_model import MultiStreamDeepfakeDetector
from models.baselines import build_baseline
from utils.utils import set_seed, get_device, save_checkpoint


def _orth_loss(*feats):
    """Penalize cosine similarity between stream feature centroids."""
    norms = [nn.functional.normalize(f.mean(0, keepdim=True), dim=-1) for f in feats]
    loss, n = torch.tensor(0.0, device=feats[0].device), len(norms)
    for i in range(n):
        for j in range(i + 1, n):
            loss = loss + (norms[i] * norms[j]).sum().abs()
    return loss / max(n * (n - 1) / 2, 1)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(labels, scores, threshold=0.5):
    preds = (np.array(scores) > threshold).astype(int)
    labels = np.array(labels)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    acc = 100.0 * (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    auc, eer = 0.0, 0.0
    if len(set(labels)) > 1:
        try:
            auc = roc_auc_score(labels, scores) * 100.0
            fpr, tpr, _ = roc_curve(labels, scores)
            eer = brentq(lambda x: 1. - x - float(interp1d(fpr, tpr)(x)),
                         0., 1.) * 100.0
        except Exception:
            pass

    return dict(acc=acc, auc=auc, eer=eer,
                precision=precision * 100, recall=recall * 100,
                f1=f1 * 100, tp=tp, fp=fp, fn=fn, tn=tn)


@torch.no_grad()
def evaluate(model, loader, device, return_video_auc=False):
    model.eval()
    all_labels, all_scores, all_video_ids, all_manip = [], [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(images)
        scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        all_labels.extend(labels.tolist())
        all_scores.extend(scores.tolist())

        if "video_id" in batch:
            all_video_ids.extend(batch["video_id"])
        if "manip_type" in batch:
            all_manip.extend(batch["manip_type"])

    frame_metrics = compute_metrics(all_labels, all_scores)

    video_metrics = None
    if return_video_auc and all_video_ids:
        vid_scores, vid_labels = aggregate_video_predictions(
            all_video_ids, np.array(all_scores), np.array(all_labels),
            strategy="mean",
        )
        video_metrics = compute_metrics(vid_labels, vid_scores)

    # Per-manipulation-type breakdown
    per_manip = {}
    if all_manip:
        for manip in set(all_manip):
            idx = [i for i, m in enumerate(all_manip) if m == manip]
            if idx:
                m_labels = [all_labels[i] for i in idx]
                m_scores = [all_scores[i] for i in idx]
                per_manip[manip] = compute_metrics(m_labels, m_scores)

    return frame_metrics, video_metrics, per_manip


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    all_labels, all_scores = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits, features = model(images, return_features=True)
            loss = criterion(logits.squeeze(-1), labels)
            if features is not None:
                orth = _orth_loss(features['spatial'], features['frequency'], features['semantic'])
                loss = loss + 0.01 * orth

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        scores = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(scores.tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(all_labels, all_scores)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train on FaceForensics++ (CVPR standard protocol)"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to FaceForensics++ root directory")
    parser.add_argument("--compression", default="c23",
                        choices=["c0", "c23", "c40"])
    parser.add_argument("--model", default="multistream",
                        choices=["multistream", "xception", "cnndetect",
                                 "univfd", "f3net"],
                        help="Model to train")
    parser.add_argument("--manipulation-types", nargs="+",
                        default=MANIPULATION_TYPES,
                        help="Manipulation types to include (default: all 4)")
    parser.add_argument("--frames-per-video", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints/ffpp")
    parser.add_argument("--ablation-mode", default=None,
                        choices=[None, "spatial_only", "freq_only",
                                 "semantic_only", "spatial_freq",
                                 "spatial_semantic"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"FF++ TRAINING — {args.compression} | model={args.model}")
    print(f"Manipulation types: {args.manipulation_types}")
    print(f"Device: {device} | Seed: {args.seed}")
    print("=" * 70)

    # Our model uses 256×256 (FFT mask); baselines use 224×224
    img_size = 256 if args.model == "multistream" else 224

    # Data
    train_loader, val_loader, test_loader = create_ffpp_dataloaders(
        root_dir=args.data_dir,
        compression=args.compression,
        manipulation_types=args.manipulation_types,
        frames_per_video=args.frames_per_video,
        batch_size=args.batch_size,
        num_workers=4,
        img_size=img_size,
        seed=args.seed,
    )

    # Model
    if args.model == "multistream":
        model = MultiStreamDeepfakeDetector(
            pretrained_backbones=True,
            ablation_mode=args.ablation_mode,
        ).to(device)
    else:
        model = build_baseline(args.model, pretrained=True).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Loss — FF++ is roughly balanced (720×4 fake vs 720 real per type)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer: no weight decay on bias/norm
    no_decay = {"bias", "norm", "layernorm"}
    grouped_params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n.lower() for nd in no_decay)],
         "weight_decay": 1e-4},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n.lower() for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped_params, lr=args.lr, betas=(0.9, 0.999))

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - 3,
                               eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[3])

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    best_val_auc = 0.0
    patience_counter = 0
    patience = 8

    print("\nTraining started...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        val_frame, val_video, _ = evaluate(
            model, val_loader, device, return_video_auc=True
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:>3}/{args.epochs} | {elapsed:.0f}s | LR={lr:.2e}\n"
            f"  Train  Loss={train_m['loss']:.4f} Acc={train_m['acc']:.1f}% "
            f"AUC={train_m['auc']:.2f}%\n"
            f"  Val    Frame-AUC={val_frame['auc']:.2f}%  "
            f"EER={val_frame['eer']:.2f}%"
            + (f"  Video-AUC={val_video['auc']:.2f}%" if val_video else "")
        )

        val_auc = val_frame["auc"]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_frame.get("loss", 0),
                            str(checkpoint_dir / f"best_{args.model}_{args.compression}.pth"))
            print(f"  >>> Best model saved (AUC={best_val_auc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final test evaluation
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    ckpt_path = checkpoint_dir / f"best_{args.model}_{args.compression}.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint: {ckpt_path}")

    test_frame, test_video, test_per_manip = evaluate(
        model, test_loader, device, return_video_auc=True
    )

    print(f"\nFrame-level:")
    print(f"  AUC:      {test_frame['auc']:.4f}%")
    print(f"  Accuracy: {test_frame['acc']:.2f}%")
    print(f"  EER:      {test_frame['eer']:.4f}%")
    print(f"  F1:       {test_frame['f1']:.2f}%")

    if test_video:
        print(f"\nVideo-level (mean aggregation):")
        print(f"  AUC:      {test_video['auc']:.4f}%")
        print(f"  Accuracy: {test_video['acc']:.2f}%")
        print(f"  EER:      {test_video['eer']:.4f}%")

    if test_per_manip:
        print("\nPer-manipulation-type breakdown (frame-level AUC):")
        header = f"  {'Type':<20} {'AUC':>8} {'Acc':>8} {'EER':>8}"
        print(header)
        print("  " + "-" * 44)
        for manip, m in sorted(test_per_manip.items()):
            if manip == "real":
                continue
            print(f"  {manip:<20} {m['auc']:>7.2f}% {m['acc']:>7.2f}% {m['eer']:>7.2f}%")

    # Save results
    results_path = checkpoint_dir / f"results_{args.model}_{args.compression}_seed{args.seed}.txt"
    with open(results_path, "w") as f:
        f.write(f"Model: {args.model} | Compression: {args.compression} | Seed: {args.seed}\n")
        f.write(f"Manipulation types: {args.manipulation_types}\n\n")
        f.write(f"Frame-level:\n")
        for k, v in test_frame.items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.4f}\n")
        if test_video:
            f.write(f"\nVideo-level:\n")
            for k, v in test_video.items():
                if isinstance(v, float):
                    f.write(f"  {k}: {v:.4f}\n")
        if test_per_manip:
            f.write("\nPer-type breakdown:\n")
            for manip, m in sorted(test_per_manip.items()):
                f.write(f"  {manip}: AUC={m['auc']:.4f} Acc={m['acc']:.2f}\n")

    print(f"\nResults saved to: {results_path}")
    print(f"Best Val AUC: {best_val_auc:.4f}%")


if __name__ == "__main__":
    main()
