"""
Multi-Seed Statistical Evaluation — CVPR Rigor
================================================
Trains the model with multiple random seeds and reports mean ± std.
CVPR reviewers expect at least 3 seeds for statistical credibility.

For each seed:
  1. Train the model from scratch
  2. Evaluate on test set
  3. Collect AUC, Accuracy, EER, F1

Finally:
  - Report mean ± std across seeds
  - Paired t-test vs. each baseline (statistical significance)
  - Produce LaTeX-ready table

Usage:
    # Quick 3-seed run on your existing dataset:
    python scripts/multi_seed_eval.py --data-dir data --seeds 3 42 123

    # On FF++ with comparison to baselines:
    python scripts/multi_seed_eval.py --data-dir /path/to/FF++ --dataset ffpp \\
        --seeds 3 42 123 --compare-baselines xception cnndetect
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from models.baselines import build_baseline
from utils.utils import set_seed, get_device, save_checkpoint


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_all_metrics(labels, scores, threshold=0.5):
    labels = np.array(labels)
    scores = np.array(scores)
    preds = (scores > threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    acc = 100.0 * (tp + tn) / max(len(labels), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)

    auc, eer = 0.0, 0.0
    if len(set(labels.tolist())) > 1:
        try:
            auc = roc_auc_score(labels, scores) * 100.0
            fpr, tpr, _ = roc_curve(labels, scores)
            eer = brentq(lambda x: 1. - x - float(interp1d(fpr, tpr)(x)),
                         0., 1.) * 100.0
        except Exception:
            pass

    return dict(auc=auc, acc=acc, eer=eer,
                f1=f1 * 100, precision=prec * 100, recall=rec * 100)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_and_evaluate_one_seed(
    seed: int,
    data_root: str,
    dataset_type: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_dir: Path,
    ffpp_compression: str = "c23",
) -> Dict[str, float]:
    """Train one seed and return test metrics."""
    set_seed(seed)
    device = get_device()

    print(f"\n{'─'*60}")
    print(f"Seed {seed} | Model: {model_name}")
    print(f"{'─'*60}")

    # Data loaders
    if dataset_type == "ffpp":
        from data.ffpp_dataset import create_ffpp_dataloaders
        train_loader, val_loader, test_loader = create_ffpp_dataloaders(
            root_dir=data_root,
            compression=ffpp_compression,
            batch_size=batch_size,
            seed=seed,
        )
    else:
        from data.dataset import create_dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            seed=seed,
        )

    # Model
    if model_name == "multistream":
        model = MultiStreamDeepfakeDetector(pretrained_backbones=True).to(device)
    else:
        model = build_baseline(model_name, pretrained=True).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_auc = 0.0
    best_ckpt = checkpoint_dir / f"seed{seed}_{model_name}_best.pth"
    patience = 8
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for batch in tqdm(train_loader, desc=f"  E{epoch:>2} Train",
                          leave=False):
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(imgs)
                loss = criterion(logits.squeeze(-1), lbls)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        # Validate
        model.eval()
        v_labels, v_scores = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    logits, _ = model(imgs)
                scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                v_labels.extend(batch["label"].numpy().tolist())
                v_scores.extend(scores.tolist())

        val_m = compute_all_metrics(v_labels, v_scores)
        print(f"  Epoch {epoch:>2} | Val AUC={val_m['auc']:.2f}%"
              f" EER={val_m['eer']:.2f}%")

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, 0.0, str(best_ckpt))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test evaluation
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t_labels, t_scores = [], []
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                logits, _ = model(imgs)
            scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            t_labels.extend(batch["label"].numpy().tolist())
            t_scores.extend(scores.tolist())

    test_m = compute_all_metrics(t_labels, t_scores)
    print(f"\n  [Seed {seed}] Test | AUC={test_m['auc']:.4f}%"
          f" Acc={test_m['acc']:.2f}% EER={test_m['eer']:.4f}%")

    return test_m


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return dict(mean=arr.mean(), std=arr.std(ddof=1),
                min=arr.min(), max=arr.max())


def format_mean_std(mean, std, decimals=2) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ± {fmt.format(std)}"


def to_latex_row(model_name: str, stats_dict: Dict) -> str:
    """Format a row for LaTeX paper table."""
    auc = format_mean_std(stats_dict["auc"]["mean"], stats_dict["auc"]["std"])
    acc = format_mean_std(stats_dict["acc"]["mean"], stats_dict["acc"]["std"])
    eer = format_mean_std(stats_dict["eer"]["mean"], stats_dict["eer"]["std"])
    f1 = format_mean_std(stats_dict["f1"]["mean"], stats_dict["f1"]["std"])
    return f"{model_name} & {auc} & {acc} & {eer} & {f1} \\\\"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed statistical evaluation for CVPR"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", default="custom",
                        choices=["custom", "ffpp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3, 42, 123],
                        help="Random seeds to run (default: 3 42 123)")
    parser.add_argument("--model", default="multistream",
                        choices=["multistream", "xception", "cnndetect",
                                 "univfd", "f3net"])
    parser.add_argument("--compare-baselines", nargs="+", default=None,
                        help="Additional baselines to compare (e.g. xception cnndetect)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ffpp-compression", default="c23")
    parser.add_argument("--output-dir", default="logs/multi_seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    models_to_run = [args.model]
    if args.compare_baselines:
        models_to_run.extend(args.compare_baselines)

    all_results: Dict[str, List[Dict]] = {}

    for model_name in models_to_run:
        all_results[model_name] = []
        for seed in args.seeds:
            m = train_and_evaluate_one_seed(
                seed=seed,
                data_root=args.data_dir,
                dataset_type=args.dataset,
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                checkpoint_dir=ckpt_dir,
                ffpp_compression=args.ffpp_compression,
            )
            all_results[model_name].append(m)

    # Aggregate statistics
    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS (mean ± std)")
    print("=" * 70)

    metric_keys = ["auc", "acc", "eer", "f1"]
    header = f"{'Model':<25}" + "".join(f" {k.upper():>14}" for k in metric_keys)
    print(header)
    print("-" * len(header))

    aggregated = {}
    for model_name, seed_results in all_results.items():
        stats = {}
        for k in metric_keys:
            values = [r[k] for r in seed_results]
            stats[k] = compute_stats(values)
        aggregated[model_name] = stats

        row = f"{model_name:<25}"
        for k in metric_keys:
            row += f" {format_mean_std(stats[k]['mean'], stats[k]['std']):>14}"
        print(row)

    # Statistical significance: paired t-test of our model vs. each baseline
    if len(models_to_run) > 1:
        our_name = args.model
        our_aucs = [r["auc"] for r in all_results[our_name]]

        print(f"\nStatistical Significance (paired t-test vs. {our_name}):")
        for baseline_name in args.compare_baselines or []:
            if baseline_name not in all_results:
                continue
            base_aucs = [r["auc"] for r in all_results[baseline_name]]
            if len(our_aucs) == len(base_aucs) and len(our_aucs) >= 2:
                t_stat, p_val = ttest_rel(our_aucs, base_aucs)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01
                      else ("*" if p_val < 0.05 else "n.s."))
                print(f"  vs {baseline_name:<20}: t={t_stat:+.3f}  p={p_val:.4f}  {sig}")

    # LaTeX table
    print("\nLaTeX Table (paste into paper):")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & AUC (\\%) & Accuracy (\\%) & EER (\\%) & F1 (\\%) \\\\")
    print("\\midrule")
    for model_name, stats in aggregated.items():
        print(to_latex_row(model_name, stats))
    print("\\bottomrule")
    print("\\end{tabular}")

    # Save to JSON
    out_path = output_dir / "multi_seed_results.json"
    save_data = {}
    for model_name, seed_results in all_results.items():
        save_data[model_name] = {
            "per_seed": seed_results,
            "seeds": args.seeds,
            "stats": {k: {sk: float(sv) for sk, sv in aggregated[model_name][k].items()}
                      for k in metric_keys},
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
