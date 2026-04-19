"""
OOD Evaluation: So-Fake-OOD + Cross-Generator Breakdown
=========================================================
Evaluates an existing checkpoint against So-Fake-OOD (real Reddit images
with openai/DALL-E, Seedream3.0, MidJourney generators — never seen during training).

Usage:
    python scripts/eval_ood.py
    python scripts/eval_ood.py --checkpoint checkpoints/best_model.pth
    python scripts/eval_ood.py --checkpoint checkpoints/cvpr/best_cvpr.pth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from utils.utils import get_device


def compute_auc(labels, scores):
    labels, scores = np.array(labels), np.array(scores)
    if len(set(labels.tolist())) < 2:
        return None
    return roc_auc_score(labels, scores) * 100.0


def compute_acc(labels, scores, t=0.5):
    return 100.0 * ((np.array(scores) > t).astype(int) == np.array(labels)).mean()


@torch.no_grad()
def evaluate_ood(model, loader, device):
    model.eval()
    all_labels, all_scores, all_gens = [], [], []

    for batch in tqdm(loader, desc="  So-Fake-OOD", leave=False):
        imgs = batch["image"].to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(imgs)
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy().tolist()
        all_scores.extend(probs)
        all_labels.extend(batch["label"].numpy().tolist())
        all_gens.extend(batch.get("generator", ["unknown"] * len(probs)))

    overall_auc = compute_auc(all_labels, all_scores)
    overall_acc = compute_acc(all_labels, all_scores)

    # Per-generator breakdown
    gen_results = {}
    for g in sorted(set(all_gens)):
        idx = [i for i, x in enumerate(all_gens) if x == g]
        g_labels = [all_labels[i] for i in idx]
        g_scores = [all_scores[i] for i in idx]
        auc = compute_auc(g_labels, g_scores)
        acc = compute_acc(g_labels, g_scores)
        fake_rate = 100.0 * sum(g_labels) / max(len(g_labels), 1)
        gen_results[g] = {
            "n": len(idx), "auc": auc, "acc": acc,
            "fake_pct": fake_rate,
            "avg_score": float(np.mean(g_scores)),
        }

    return {
        "overall_auc": overall_auc,
        "overall_acc": overall_acc,
        "n_total": len(all_labels),
        "n_fake": int(sum(all_labels)),
        "n_real": int(len(all_labels) - sum(all_labels)),
        "per_generator": gen_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Save results to this .txt file")
    args = parser.parse_args()

    device = get_device()
    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    print("=" * 65)
    print("OOD EVALUATION — So-Fake-OOD")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device:     {device}")
    print("=" * 65)

    # Load model
    model = MultiStreamDeepfakeDetector(pretrained_backbones=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:5]}...")
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    # Build OOD loader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from data.hf_sofake import SoFakeOODDataset

    tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    ood_ds = SoFakeOODDataset(transform=tf, max_samples=args.max_samples, seed=args.seed)
    ood_loader = torch.utils.data.DataLoader(
        ood_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluate
    results = evaluate_ood(model, ood_loader, device)

    # Print results
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Total samples : {results['n_total']:,}")
    print(f"  Real          : {results['n_real']:,}")
    print(f"  Fake          : {results['n_fake']:,}")
    print(f"\n  Overall AUC   : {results['overall_auc']:.2f}%")
    print(f"  Overall Acc   : {results['overall_acc']:.2f}%")
    print()
    print(f"  {'Generator':<22} {'N':>6}  {'Fake%':>6}  {'AUC':>8}  {'Acc':>8}  {'AvgScore':>9}")
    print("  " + "-" * 65)
    for g, v in sorted(results["per_generator"].items()):
        auc_str = f"{v['auc']:.2f}%" if v['auc'] is not None else "  N/A  "
        print(f"  {g:<22} {v['n']:>6}  {v['fake_pct']:>5.1f}%  {auc_str:>8}  {v['acc']:>6.2f}%  {v['avg_score']:>9.4f}")

    # Save
    out_path = args.out or str(ckpt_path.parent / "ood_results.txt")
    lines = [
        f"Checkpoint: {ckpt_path}\n",
        f"Total: {results['n_total']}  Real: {results['n_real']}  Fake: {results['n_fake']}\n",
        f"Overall AUC: {results['overall_auc']:.4f}%\n",
        f"Overall Acc: {results['overall_acc']:.4f}%\n\n",
        "Per-generator:\n",
    ]
    for g, v in sorted(results["per_generator"].items()):
        auc_str = f"{v['auc']:.4f}%" if v['auc'] is not None else "N/A"
        lines.append(f"  {g}: AUC={auc_str}  Acc={v['acc']:.4f}%  N={v['n']}\n")
    with open(out_path, "w") as fh:
        fh.writelines(lines)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
