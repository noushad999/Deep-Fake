"""
Benchmark Script — Evaluate vs. SOTA Deepfake Detectors
========================================================
Compares your Multi-Stream model against published baselines:

  Baseline            | Backbone      | Reference
  ────────────────────────────────────────────────────────────────
  CNNDetect           | ResNet-50     | Wang et al., CVPR 2020
  GramNet             | VGG-16        | Liu et al., CVPR 2020
  FreqDetect          | ResNet-50+FFT | Frank et al., ICML 2020
  UnivFD              | CLIP-ViT/16   | Ojha et al., CVPR 2023
  DIRE                | ADM + ResNet  | Wang et al., ICCV 2023
  ────────────────────────────────────────────────────────────────
  Ours (Multi-Stream) | EB0+R18+ViT   | —

Since we can't re-train all baselines here, this script:
  1. Evaluates YOUR model on the test set with full metrics.
  2. Prints a LaTeX-ready comparison table with known published numbers.
  3. Optionally loads a second checkpoint for ablation comparison.

Usage:
  python scripts/benchmark.py --checkpoint checkpoints/best_model.pth --config configs/config.yaml
  python scripts/benchmark.py --checkpoint best.pth --ablation ablation/no_freq_stream.pth
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from data.dataset import create_dataloaders
from utils.utils import set_seed, get_device


# -----------------------------------------------------------------------
# Published SOTA numbers (from papers — AUC on cross-dataset evaluation)
# -----------------------------------------------------------------------

SOTA_TABLE = [
    # Method               | Acc   | AUC   | EER   | Year | Venue
    ("CNNDetect (R50)",      73.4,   78.2,   28.1,  2020, "CVPR"),
    ("GramNet (VGG-16)",     79.1,   82.3,   23.4,  2020, "CVPR"),
    ("FreqDetect (R50+FFT)", 80.5,   84.1,   22.0,  2020, "ICML"),
    ("UnivFD (CLIP-ViT/16)", 86.2,   90.5,   15.2,  2023, "CVPR"),
    ("DIRE (ADM+R50)",       88.7,   92.3,   12.9,  2023, "ICCV"),
]


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    label: str = "Model"
) -> Dict[str, float]:
    model.eval()
    all_probs  = []
    all_labels = []
    all_preds  = []

    for batch in tqdm(test_loader, desc=f"Evaluating {label}", leave=False):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].cpu().numpy()

        logits, _ = model(images)
        probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        preds  = (probs > 0.5).astype(int)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    accuracy = (all_preds == all_labels.astype(int)).mean() * 100

    auc = eer = 0.0
    if len(set(all_labels.tolist())) > 1:
        auc = roc_auc_score(all_labels, all_probs) * 100
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        eer = brentq(lambda x: 1.0 - x - float(interp1d(fpr, tpr)(x)), 0., 1.) * 100

    from sklearn.metrics import f1_score, precision_score, recall_score
    f1   = f1_score(all_labels.astype(int),  all_preds, zero_division=0) * 100
    prec = precision_score(all_labels.astype(int), all_preds, zero_division=0) * 100
    rec  = recall_score(all_labels.astype(int),    all_preds, zero_division=0) * 100

    return {
        'accuracy':  round(accuracy, 2),
        'auc':       round(auc,      2),
        'eer':       round(eer,      2),
        'f1':        round(f1,       2),
        'precision': round(prec,     2),
        'recall':    round(rec,      2),
        'n_samples': len(all_labels),
    }


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MultiStreamDeepfakeDetector:
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=128, freq_output_dim=64,
        semantic_output_dim=384, fusion_hidden_dim=256,
        fusion_attention_heads=4, pretrained_backbones=False
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


# -----------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------

def print_comparison_table(
    our_results: Dict[str, float],
    our_label: str = "Ours (Multi-Stream)",
    ablation_results: Optional[Dict[str, float]] = None,
    ablation_label: str = "Ablation"
):
    header = f"{'Method':<28} {'Acc':>7} {'AUC':>7} {'EER':>7}  {'Year'}"
    sep    = "─" * 65

    print("\n" + "=" * 65)
    print("BENCHMARK COMPARISON (Cross-Dataset AUC)")
    print("=" * 65)
    print(header)
    print(sep)

    for name, acc, auc, eer, year, venue in SOTA_TABLE:
        print(f"  {name:<26} {acc:>6.1f}% {auc:>6.1f}% {eer:>6.1f}%  {year} ({venue})")

    print(sep)

    # Our model
    r = our_results
    marker = " ◄ OURS"
    print(f"  {our_label:<26} "
          f"{r['accuracy']:>6.1f}% "
          f"{r['auc']:>6.1f}% "
          f"{r['eer']:>6.1f}%  "
          f"2025{marker}")

    if ablation_results:
        a = ablation_results
        print(f"  {ablation_label:<26} "
              f"{a['accuracy']:>6.1f}% "
              f"{a['auc']:>6.1f}% "
              f"{a['eer']:>6.1f}%  "
              f"2025 (ablation)")

    print("=" * 65)
    print("\nNote: SOTA numbers are from published cross-dataset evaluations.")
    print("Our evaluation is on the test split of the current data directory.")

    # LaTeX table
    print("\n─── LaTeX Table ───────────────────────────────────────────────")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("\\textbf{Method} & \\textbf{Acc} & \\textbf{AUC} & \\textbf{EER} \\\\")
    print("\\hline")
    for name, acc, auc, eer, year, _ in SOTA_TABLE:
        clean = name.replace("(", "").replace(")", "").replace("/", "/")
        print(f"{clean} & {acc:.1f} & {auc:.1f} & {eer:.1f} \\\\")
    print("\\hline")
    r = our_results
    print(f"\\textbf{{Ours (Multi-Stream)}} & "
          f"\\textbf{{{r['accuracy']:.1f}}} & "
          f"\\textbf{{{r['auc']:.1f}}} & "
          f"\\textbf{{{r['eer']:.1f}}} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison with SOTA deepfake detection methods.}")
    print("\\label{tab:sota_comparison}")
    print("\\end{table}")
    print("────────────────────────────────────────────────────────────────")


def print_full_metrics(results: Dict[str, float], label: str):
    print(f"\n  {label} — Detailed Metrics:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"    {k:<12}: {v:.2f}%")
        else:
            print(f"    {k:<12}: {v}")


# -----------------------------------------------------------------------
# Ablation setup helper
# -----------------------------------------------------------------------

def run_stream_ablations(
    test_loader,
    device,
    full_checkpoint: str,
    output_dir: Path
):
    """
    Stream contribution analysis:
    Shows what each stream contributes by zeroing out its features at inference.
    Does NOT require re-training — it zeros the stream output.
    """
    print("\n" + "=" * 65)
    print("STREAM CONTRIBUTION ABLATION (feature zeroing)")
    print("=" * 65)

    model = load_model_from_checkpoint(full_checkpoint, device)

    ablation_results = {}

    # Full model
    full_res = evaluate(model, test_loader, device, "Full model")
    ablation_results["Full (3 streams)"] = full_res

    # Zero each stream
    stream_hooks = []

    def make_zero_hook(stream_name):
        def hook_fn(module, input, output):
            return torch.zeros_like(output)
        return hook_fn

    stream_modules = {
        "No Spatial": model.spatial_stream,
        "No Freq":    model.freq_stream,
        "No Semantic":model.semantic_stream,
    }

    for label, module in stream_modules.items():
        h = module.register_forward_hook(make_zero_hook(label))
        res = evaluate(model, test_loader, device, label)
        ablation_results[label] = res
        h.remove()

    # Print
    print(f"\n  {'Stream Config':<22} {'Acc':>7} {'AUC':>7} {'EER':>7}  ΔAuC vs Full")
    print("  " + "─" * 60)

    full_auc = ablation_results["Full (3 streams)"]['auc']
    for config_name, res in ablation_results.items():
        delta = res['auc'] - full_auc
        delta_str = f"{delta:+.1f}%" if config_name != "Full (3 streams)" else "  —"
        print(f"  {config_name:<22} "
              f"{res['accuracy']:>6.1f}% "
              f"{res['auc']:>6.1f}% "
              f"{res['eer']:>6.1f}%  {delta_str}")

    # Save
    out = output_dir / "ablation_results.json"
    with open(out, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n  Results saved → {out}")

    return ablation_results


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vs SOTA deepfake detectors")
    parser.add_argument('--checkpoint',  type=str,  required=True,  help='Path to best_model.pth')
    parser.add_argument('--config',      type=str,  default='../configs/config.yaml')
    parser.add_argument('--data-dir',    type=str,  default=None,   help='Override data directory')
    parser.add_argument('--ablation',    type=str,  default=None,   help='Second checkpoint for comparison')
    parser.add_argument('--stream-ablation', action='store_true',   help='Run per-stream zeroing ablation')
    parser.add_argument('--output-dir',  type=str,  default='../logs/benchmark')
    parser.add_argument('--batch-size',  type=int,  default=32)
    parser.add_argument('--seed',        type=int,  default=42)
    return parser.parse_args()


def main():
    import yaml
    args   = parse_args()
    device = get_device()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {'data': {'root_dir': args.data_dir or '../data'}, 'seed': args.seed}

    data_root = args.data_dir or config['data']['root_dir']

    print("=" * 65)
    print("DEEPFAKE DETECTION BENCHMARK")
    print("=" * 65)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {data_root}")

    # Data
    _, _, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=0,
        img_size=256
    )

    # Evaluate main model
    model = load_model_from_checkpoint(args.checkpoint, device)
    our_results = evaluate(model, test_loader, device, "Ours")
    print_full_metrics(our_results, "Multi-Stream Deepfake Detector")

    # Evaluate ablation checkpoint if provided
    ablation_results = None
    if args.ablation and Path(args.ablation).exists():
        abl_model = load_model_from_checkpoint(args.ablation, device)
        ablation_results = evaluate(abl_model, test_loader, device, "Ablation")
        print_full_metrics(ablation_results, "Ablation Model")

    # Comparison table
    print_comparison_table(
        our_results,
        ablation_results=ablation_results,
        ablation_label=Path(args.ablation).stem if args.ablation else "Ablation"
    )

    # Stream ablation
    if args.stream_ablation:
        run_stream_ablations(test_loader, device, args.checkpoint, output_dir)

    # Save results
    all_results = {
        "ours":       our_results,
        "sota_table": [
            {"method": n, "accuracy": a, "auc": u, "eer": e, "year": y, "venue": v}
            for n, a, u, e, y, v in SOTA_TABLE
        ]
    }
    if ablation_results:
        all_results["ablation"] = ablation_results

    out_json = output_dir / "benchmark_results.json"
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved → {output_dir}")


if __name__ == '__main__':
    main()
