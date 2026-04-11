"""
Production-Grade Evaluation with Compression Robustness Testing
Tests model at multiple compression levels: C0, C23, C40
Cross-dataset generalization testing
Compare against SOTA baselines
"""
import os
import sys
import json
import io
import argparse
import yaml
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from data.dataset import create_dataloaders, DeepfakeDataset
from utils.utils import set_seed, get_device, load_checkpoint


def apply_jpeg_compression(tensor, quality):
    """Apply JPEG compression to a batch of tensors."""
    compressed = []
    for img_tensor in tensor:
        img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_pil = Image.open(buffer)
        compressed_np = np.array(compressed_pil) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        compressed_np = (compressed_np - mean) / std
        compressed_tensor = torch.from_numpy(compressed_np.transpose(2, 0, 1)).float().to(tensor.device)
        compressed.append(compressed_tensor)
    return torch.stack(compressed)


def apply_gaussian_blur(tensor, sigma):
    """Apply Gaussian blur to a batch of tensors."""
    from scipy.ndimage import gaussian_filter
    blurred = []
    for img_tensor in tensor:
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        # Apply blur to each channel separately
        blurred_channels = [gaussian_filter(img_np[:, :, c], sigma=sigma) for c in range(img_np.shape[2])]
        blurred_np = np.stack(blurred_channels, axis=-1)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        blurred_np = (blurred_np - mean) / std
        blurred_tensor = torch.from_numpy(blurred_np.transpose(2, 0, 1)).float().to(tensor.device)
        blurred.append(blurred_tensor)
    return torch.stack(blurred)


def evaluate_at_condition(model, test_loader, device, condition_name, transform_fn=None):
    """Evaluate model under specific conditions."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {condition_name}", leave=False):
            images = batch['image'].to(device)
            labels = batch['label']
            
            if transform_fn:
                images = transform_fn(images)
            
            logits, _ = model(images)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'n_samples': len(all_labels)
    }


def compression_robustness_test(model, test_loader, device, output_dir):
    """Test model at different compression levels (FF++ protocol)."""
    print("\n" + "="*70)
    print("COMPRESSION ROBUSTNESS TEST (FF++ Protocol)")
    print("="*70)
    
    compression_levels = {
        'C0  (No compression, Q=100)': lambda x: x,
        'C23 (Medium compression, Q=75)': lambda x: apply_jpeg_compression(x, quality=75),
        'C40 (Heavy compression, Q=40)': lambda x: apply_jpeg_compression(x, quality=40),
        'C50 (Very heavy, Q=25)': lambda x: apply_jpeg_compression(x, quality=25),
    }
    
    results = {}
    
    for name, transform_fn in compression_levels.items():
        metrics = evaluate_at_condition(model, test_loader, device, name, transform_fn)
        results[name] = metrics
        print(f"  {name:<35} Acc: {metrics['accuracy']:5.1f}%  F1: {metrics['f1_score']:5.1f}%  AUC: {metrics['auc_roc']:5.1f}%")
    
    # Compute degradation
    clean_acc = results['C0  (No compression, Q=100)']['accuracy']
    for name in ['C23 (Medium compression, Q=75)', 'C40 (Heavy compression, Q=40)', 'C50 (Very heavy, Q=25)']:
        if name in results:
            drop = clean_acc - results[name]['accuracy']
            print(f"  {'Degradation '+name.split('(')[0]:<35} Drop: {drop:5.1f}pp")
    
    # Save results
    output_path = Path(output_dir) / 'compression_robustness.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    print(f"\nResults saved to: {output_path}")
    
    # Plot compression robustness curve
    plot_compression_curve(results, output_dir)
    
    return results


def blur_robustness_test(model, test_loader, device, output_dir):
    """Test model under Gaussian blur."""
    print("\n" + "="*70)
    print("BLUR ROBUSTNESS TEST")
    print("="*70)
    
    blur_levels = {
        'No blur (σ=0)': lambda x: x,
        'Light blur (σ=1)': lambda x: apply_gaussian_blur(x, sigma=1),
        'Medium blur (σ=2)': lambda x: apply_gaussian_blur(x, sigma=2),
        'Heavy blur (σ=4)': lambda x: apply_gaussian_blur(x, sigma=4),
    }
    
    results = {}
    
    for name, transform_fn in blur_levels.items():
        metrics = evaluate_at_condition(model, test_loader, device, name, transform_fn)
        results[name] = metrics
        print(f"  {name:<35} Acc: {metrics['accuracy']:5.1f}%  F1: {metrics['f1_score']:5.1f}%  AUC: {metrics['auc_roc']:5.1f}%")
    
    output_path = Path(output_dir) / 'blur_robustness.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    
    return results


def noise_robustness_test(model, test_loader, device, output_dir):
    """Test model under additive noise."""
    print("\n" + "="*70)
    print("NOISE ROBUSTNESS TEST")
    print("="*70)
    
    def add_noise(tensor, noise_level):
        noise = torch.randn_like(tensor) * noise_level
        noisy = tensor + noise
        return noisy
    
    noise_levels = {
        'No noise': lambda x: x,
        'Light noise (σ=0.01)': lambda x: add_noise(x, 0.01),
        'Medium noise (σ=0.05)': lambda x: add_noise(x, 0.05),
        'Heavy noise (σ=0.1)': lambda x: add_noise(x, 0.1),
    }
    
    results = {}
    
    for name, transform_fn in noise_levels.items():
        metrics = evaluate_at_condition(model, test_loader, device, name, transform_fn)
        results[name] = metrics
        print(f"  {name:<35} Acc: {metrics['accuracy']:5.1f}%  F1: {metrics['f1_score']:5.1f}%  AUC: {metrics['auc_roc']:5.1f}%")
    
    output_path = Path(output_dir) / 'noise_robustness.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    
    return results


def plot_compression_curve(results, output_dir):
    """Plot compression robustness curve."""
    qualities = [100, 75, 40, 25]
    names = ['C0 (Q=100)', 'C23 (Q=75)', 'C40 (Q=40)', 'C50 (Q=25)']
    
    accuracies = []
    f1s = []
    aucs = []
    
    for name in names:
        # Find matching key
        for key, val in results.items():
            if name.split('(')[0].strip() in key:
                accuracies.append(val['accuracy'])
                f1s.append(val['f1_score'])
                aucs.append(val['auc_roc'])
                break
    
    if not accuracies:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, metric, color in zip(axes, [accuracies, f1s, aucs], ['#2ecc71', '#3498db', '#e74c3c']):
        ax.plot(qualities, metric, 'o-', linewidth=2, markersize=8, color=color)
        ax.set_xlabel('JPEG Quality')
        ax.set_ylabel(metric == accuracies and 'Accuracy (%)' or metric == f1s and 'F1 Score (%)' or 'AUC (%)')
        ax.set_title(metric == accuracies and 'Accuracy' or metric == f1s and 'F1 Score' or 'AUC')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(qualities)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'compression_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Compression curve saved: {Path(output_dir) / 'compression_curve.png'}")


def generate_comparison_report(v1_results, v2_results, compression_v1, compression_v2, output_dir):
    """Generate a comprehensive comparison report."""
    print("\n" + "="*70)
    print("COMPARISON REPORT: Baseline vs Production")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Production':>12} {'Change':>10}")
    print(f"{'─'*60}")
    
    for key in ['accuracy', 'f1_score', 'auc_roc']:
        v1 = v1_results.get(key, 0)
        v2 = v2_results.get(key, 0)
        change = v2 - v1
        print(f"{key:<25} {v1:>11.1f}% {v2:>11.1f}% {change:>+9.1f}pp")
    
    print(f"\n{'Compression Robustness':<25} {'Baseline':>12} {'Production':>12}")
    print(f"{'─'*60}")
    
    if compression_v1 and compression_v2:
        for name in compression_v1:
            c1 = compression_v1[name]['accuracy']
            # Find matching name in v2
            c2 = None
            for name2 in compression_v2:
                if name.split('(')[0].strip() in name2:
                    c2 = compression_v2[name2]['accuracy']
                    break
            if c2 is not None:
                print(f"  {name:<23} {c1:>11.1f}% {c2:>11.1f}%")
    
    # Save report
    report = {
        'baseline': v1_results,
        'production': v2_results,
        'compression_baseline': compression_v1,
        'compression_production': compression_v2,
    }
    
    with open(Path(output_dir) / 'comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=lambda x: float(x))


def evaluate_model(
    config_path: str = '../configs/config.yaml',
    checkpoint: str = None,
    data_dir: str = None,
    output_dir: str = None,
    batch_size: int = 32
):
    """Full evaluation pipeline."""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if data_dir:
        config['data']['root_dir'] = data_dir
    
    set_seed(config['seed'])
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=config['model']['spatial']['feature_dim'],
        freq_output_dim=config['model']['frequency']['feature_dim'],
        semantic_output_dim=config['model']['semantic']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        fusion_attention_heads=config['model']['fusion']['attention_heads'],
        pretrained_backbones=False
    ).to(device)
    
    if checkpoint and os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        try:
            load_checkpoint(model, torch.optim.AdamW(model.parameters()), checkpoint, device)
        except ValueError:
            print("Loading model weights only (optimizer mismatch)...")
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        print(f"Model loaded successfully")
    
    # Create test loader
    _, _, test_loader = create_dataloaders(
        data_root=config['data']['root_dir'],
        batch_size=batch_size,
        num_workers=0,
        img_size=256
    )
    
    # Standard evaluation
    print("\n" + "="*70)
    print("STANDARD EVALUATION")
    print("="*70)
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['label']
            
            logits, _ = model(images)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except:
        auc = 0.0
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
    }
    
    print(f"\n{'Metric':<20} {'Value':>10}")
    print(f"{'─'*30}")
    for k, v in results.items():
        print(f"{k:<20} {v:>9.2f}%")
    
    # Robustness tests
    compression_results = compression_robustness_test(model, test_loader, device, str(output_dir))
    blur_results = blur_robustness_test(model, test_loader, device, str(output_dir))
    noise_results = noise_robustness_test(model, test_loader, device, str(output_dir))
    
    # Save all results
    all_results = {
        'standard': results,
        'compression': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in compression_results.items()},
        'blur': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in blur_results.items()},
        'noise': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in noise_results.items()},
    }
    
    with open(output_dir / 'evaluation_full.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*70}")
    
    return results, compression_results, blur_results, noise_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(
        config_path=args.config,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
