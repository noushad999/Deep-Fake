"""
Cross-Dataset Generalization Evaluation — CVPR Standard Protocol
=================================================================
Tests a model trained on FaceForensics++ on held-out datasets:
  - Celeb-DF v2 (Li et al., CVPR 2020)
  - FaceShifter on FF++ (optional)
  - DFDC preview (if available)

This is the MOST IMPORTANT evaluation for CVPR:
"Does the model generalize beyond its training distribution?"

Standard protocol:
  - Train on FF++ (c23, all 4 manipulation types)
  - Test on CelebDF-v2 official test partition
  - Report VIDEO-LEVEL AUC (not frame-level) for CelebDF

Usage:
    python scripts/cross_dataset_eval.py \\
        --checkpoint checkpoints/ffpp/best_multistream_c23.pth \\
        --celebdf-dir /path/to/Celeb-DF-v2 \\
        --ffpp-test-dir /path/to/FaceForensics++

    # Compare multiple models:
    python scripts/cross_dataset_eval.py \\
        --checkpoints multistream.pth xception.pth f3net.pth \\
        --model-names "Ours (3-Stream)" "Xception" "F3Net" \\
        --celebdf-dir /path/to/Celeb-DF-v2
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.celebdf_dataset import create_celebdf_testloader
from data.ffpp_dataset import (
    FFPPDataset, get_ffpp_transforms, aggregate_video_predictions,
    MANIPULATION_TYPES,
)
from models.full_model import MultiStreamDeepfakeDetector
from models.baselines import build_baseline
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_auc_eer(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    if len(set(labels)) < 2:
        return 0.0, 0.0
    try:
        auc = roc_auc_score(labels, scores) * 100.0
        fpr, tpr, _ = roc_curve(labels, scores)
        eer = brentq(lambda x: 1. - x - float(interp1d(fpr, tpr)(x)),
                     0., 1.) * 100.0
        return auc, eer
    except Exception:
        return 0.0, 0.0


def compute_accuracy(labels, scores, threshold=0.5):
    preds = (np.array(scores) > threshold).astype(int)
    return 100.0 * (preds == np.array(labels)).mean()


@torch.no_grad()
def run_inference(model, loader, device):
    """Returns (all_labels, all_scores, all_video_ids)."""
    model.eval()
    all_labels, all_scores, all_video_ids = [], [], []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(images)
        scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        all_labels.extend(batch["label"].numpy().tolist())
        all_scores.extend(scores.tolist())
        if "video_id" in batch:
            all_video_ids.extend(batch["video_id"])

    return (np.array(all_labels), np.array(all_scores),
            all_video_ids if all_video_ids else None)


def load_model(checkpoint_path, model_type, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if model_type == "multistream":
        model = MultiStreamDeepfakeDetector(pretrained_backbones=False)
    else:
        model = build_baseline(model_type, pretrained=False)

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def evaluate_on_dataset(model, loader, device, dataset_name):
    """Evaluate model on a dataset and report frame + video AUC."""
    print(f"\n  Evaluating on {dataset_name}...")
    labels, scores, video_ids = run_inference(model, loader, device)

    # Frame-level
    frame_auc, frame_eer = compute_auc_eer(labels, scores)
    frame_acc = compute_accuracy(labels, scores)

    result = {
        "dataset": dataset_name,
        "frame_auc": frame_auc,
        "frame_eer": frame_eer,
        "frame_acc": frame_acc,
    }

    # Video-level
    if video_ids is not None:
        vid_scores, vid_labels = aggregate_video_predictions(
            video_ids, scores, labels, strategy="mean"
        )
        vid_auc, vid_eer = compute_auc_eer(vid_labels, vid_scores)
        vid_acc = compute_accuracy(vid_labels, vid_scores)
        result.update(dict(video_auc=vid_auc, video_eer=vid_eer,
                           video_acc=vid_acc))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-dataset generalization evaluation"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint path")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Multiple checkpoint paths for comparison table")
    parser.add_argument("--model-type", default="multistream",
                        choices=["multistream", "xception", "cnndetect",
                                 "univfd", "f3net"],
                        help="Model architecture (applies to --checkpoint)")
    parser.add_argument("--model-types", nargs="+", default=None,
                        help="Model types for each checkpoint in --checkpoints")
    parser.add_argument("--model-names", nargs="+", default=None,
                        help="Display names for each model")
    parser.add_argument("--celebdf-dir", type=str, default=None,
                        help="Path to Celeb-DF-v2 root directory")
    parser.add_argument("--ffpp-test-dir", type=str, default=None,
                        help="Path to FF++ root (optional, for in-dist test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", default="logs/cross_dataset")
    args = parser.parse_args()

    # Normalize to list of (checkpoint, model_type, display_name)
    if args.checkpoints:
        ckpt_list = args.checkpoints
        type_list = args.model_types or [args.model_type] * len(ckpt_list)
        name_list = args.model_names or [Path(c).stem for c in ckpt_list]
    elif args.checkpoint:
        ckpt_list = [args.checkpoint]
        type_list = [args.model_type]
        name_list = args.model_names or [Path(args.checkpoint).stem]
    else:
        print("ERROR: Provide --checkpoint or --checkpoints")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build data loaders
    loaders = {}

    if args.celebdf_dir:
        loaders["Celeb-DF-v2"] = create_celebdf_testloader(
            root_dir=args.celebdf_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
        )
        print(f"CelebDF-v2 test loader ready.")

    if args.ffpp_test_dir:
        from data.ffpp_dataset import FFPPDataset, get_ffpp_transforms
        ffpp_test_ds = FFPPDataset(
            root_dir=args.ffpp_test_dir,
            split="test",
            compression="c23",
            frames_per_video=-1,
            transform=get_ffpp_transforms("test", args.img_size),
            return_video_id=True,
        )
        loaders["FF++ (c23, in-dist)"] = DataLoader(
            ffpp_test_ds, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
        print(f"FF++ test loader ready.")

    if not loaders:
        print("ERROR: No evaluation datasets provided. "
              "Use --celebdf-dir and/or --ffpp-test-dir")
        return

    # Evaluate each model
    all_results = {}
    for ckpt_path, model_type, model_name in zip(ckpt_list, type_list, name_list):
        print(f"\n{'='*70}")
        print(f"Model: {model_name} ({model_type})")
        print(f"Checkpoint: {ckpt_path}")
        print("=" * 70)

        model = load_model(ckpt_path, model_type, device)
        all_results[model_name] = {}

        for dataset_name, loader in loaders.items():
            res = evaluate_on_dataset(model, loader, device, dataset_name)
            all_results[model_name][dataset_name] = res

    # Print comparison table
    print("\n" + "=" * 80)
    print("CROSS-DATASET GENERALIZATION RESULTS (Video-level AUC %)")
    print("=" * 80)

    dataset_names = list(loaders.keys())
    header = f"{'Model':<30}" + "".join(f" {d:<20}" for d in dataset_names)
    print(header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        row = f"{model_name:<30}"
        for dn in dataset_names:
            if dn in results:
                res = results[dn]
                auc = res.get("video_auc", res.get("frame_auc", 0.0))
                row += f" {auc:>6.2f}%             "
            else:
                row += " N/A                 "
        print(row)

    # Also print frame-level table
    print("\n" + "=" * 80)
    print("FRAME-LEVEL AUC %")
    print("=" * 80)
    print(header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        row = f"{model_name:<30}"
        for dn in dataset_names:
            if dn in results:
                auc = results[dn]["frame_auc"]
                row += f" {auc:>6.2f}%             "
            else:
                row += " N/A                 "
        print(row)

    # Save to file
    out_path = output_dir / "cross_dataset_results.txt"
    with open(out_path, "w") as f:
        f.write("Cross-Dataset Generalization Results\n")
        f.write("=" * 60 + "\n\n")
        for model_name, results in all_results.items():
            f.write(f"Model: {model_name}\n")
            for dn, res in results.items():
                f.write(f"  Dataset: {dn}\n")
                f.write(f"    Frame AUC: {res['frame_auc']:.4f}%\n")
                f.write(f"    Frame EER: {res['frame_eer']:.4f}%\n")
                f.write(f"    Frame Acc: {res['frame_acc']:.2f}%\n")
                if "video_auc" in res:
                    f.write(f"    Video AUC: {res['video_auc']:.4f}%\n")
                    f.write(f"    Video EER: {res['video_eer']:.4f}%\n")
            f.write("\n")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
