"""
Production-Grade Deepfake Detection Training
Based on GenD (SOTA 2025) + NPR (CVPR 2024) methodology.

Key improvements over naive training:
1. Frozen backbone, tune only LayerNorm (0.03% params)
2. Alignment + Uniformity losses for hyperspherical feature space
3. Heavy augmentation (JPEG, blur, noise, compression)
4. Paired real-fake training support
5. Cosine LR scheduling with warmup
"""
import os
import sys
import time
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from data.dataset import create_dataloaders
from utils.utils import set_seed, get_device, save_checkpoint, load_checkpoint


class HypersphericalLoss:
    """
    Alignment + Uniformity losses from GenD (arXiv 2025).
    Creates well-structured feature manifold for OOD generalization.
    """
    
    @staticmethod
    def alignment_loss(features, labels):
        """
        Minimize intra-class distance on hypersphere.
        Pull same-class features together.
        """
        # Get positive pairs
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = pos_mask & ~torch.eye(len(labels), device=labels.device).bool()
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # Pairwise distances
        dist = torch.cdist(features, features, p=2)
        
        # Only consider positive pairs
        loss = (dist * pos_mask.float()).sum() / max(pos_mask.sum(), 1)
        return loss
    
    @staticmethod
    def uniformity_loss(features, t=2.0):
        """
        Encourage features to spread uniformly on hypersphere.
        Prevents feature collapse.
        """
        # Pairwise squared distances
        dist = torch.cdist(features, features, p=2) ** 2
        
        # Gaussian potential
        loss = torch.exp(-t * dist).mean().log()
        return loss


class HeavyAugmentation:
    """
    Production-grade augmentation for robustness.
    Based on FF++ / GenD training protocols.
    """
    def __init__(self, level="heavy"):
        self.level = level
        self._setup_transforms()
    
    def _setup_transforms(self):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        if self.level == "heavy":
            self.train_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                # Color augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                ], p=0.7),
                # Blur & noise (critical for robustness)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.4),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.3),
                # JPEG compression (MOST IMPORTANT for deepfake robustness)
                A.ImageCompression(quality_lower=40, quality_upper=95, p=0.5),
                # Random crop and resize
                A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        elif self.level == "medium":
            self.train_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                ], p=0.3),
                A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
                A.RandomResizedCrop(224, 224, scale=(0.85, 1.0), p=0.5),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:  # light
            self.train_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
                A.RandomResizedCrop(224, 224, scale=(0.9, 1.0), p=0.3),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __call__(self, image, is_train=True):
        import numpy as np
        if isinstance(image, np.ndarray):
            img = image
        else:
            # Convert PIL/tensor to numpy
            img = np.array(image)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.dtype == np.uint8:
                pass
            else:
                img = (img * 255).astype(np.uint8)
        
        transform = self.train_transform if is_train else self.val_transform
        augmented = transform(image=img)
        return augmented['image']


def freeze_backbone_params(model):
    """
    Freeze all backbone parameters. Only tune LayerNorm.
    Based on GenD methodology: 0.03% trainable params.
    """
    for name, param in model.named_parameters():
        if 'norm' in name.lower() or 'ln' in name.lower() or 'fusion' in name.lower():
            param.requires_grad = True
        elif 'classifier' in name.lower() or 'head' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParameter-Efficient Tuning:")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Frozen:    {total - trainable:,}")
    print(f"  Total:     {total:,}")
    return trainable, total


def combined_loss(logits, labels, features, alpha=0.1, beta=0.5):
    """
    Combined objective: CE + Alignment + Uniformity
    From GenD (arXiv 2025)
    """
    # Standard BCE
    ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
    
    # L2 normalize features for hyperspherical learning
    if features is not None:
        # Normalize the fused features
        norm_features = F.normalize(features, p=2, dim=1)
        
        # Alignment loss
        align_loss = HypersphericalLoss.alignment_loss(norm_features, labels.long())
        
        # Uniformity loss
        uniform_loss = HypersphericalLoss.uniformity_loss(norm_features)
        
        total_loss = ce_loss + alpha * align_loss + beta * uniform_loss
        
        return total_loss, ce_loss, align_loss, uniform_loss
    else:
        return ce_loss, ce_loss, torch.tensor(0.0), torch.tensor(0.0)


class MetricsTracker:
    """Tracks and computes training/evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.ce_losses = []
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, loss: float, ce_loss: float, logits: torch.Tensor, labels: torch.Tensor):
        self.losses.append(loss)
        self.ce_losses.append(ce_loss)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long().squeeze()
        
        self.correct += (preds == labels.long()).sum().item()
        self.total += labels.size(0)
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        self.all_probs.extend(probs.squeeze().detach().cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        avg_loss = np.mean(self.losses)
        avg_ce = np.mean(self.ce_losses)
        accuracy = 100.0 * self.correct / max(self.total, 1)
        
        tp = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 0 and l == 0)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(self.all_labels, self.all_probs) * 100
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'ce_loss': avg_ce,
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'auc_roc': auc,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }


def train_one_epoch(
    model, train_loader, optimizer, device, epoch,
    alpha=0.1, beta=0.5, max_grad_norm=1.0
):
    """Single training epoch with combined loss."""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    optimizer.zero_grad()
    
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # Forward
        logits, features = model(images, return_features=True)
        logits = logits.squeeze(-1)
        
        # Combined loss
        loss, ce_loss, align_loss, uniform_loss = combined_loss(
            logits, labels, features['fused'], alpha, beta
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        # Update metrics
        metrics.update(loss.item(), ce_loss.item(), logits, labels)
        
        computed = metrics.compute()
        pbar.set_postfix({
            'loss': f"{computed['loss']:.4f}",
            'acc': f"{computed['accuracy']:.1f}%",
            'f1': f"{computed['f1_score']:.1f}%"
        })
    
    return metrics


@torch.no_grad()
def validate(model, val_loader, device, epoch):
    """Validation epoch."""
    model.eval()
    metrics = MetricsTracker()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        logits, features = model(images, return_features=True)
        logits = logits.squeeze(-1)
        
        loss, ce_loss, _, _ = combined_loss(logits, labels, features['fused'])
        metrics.update(loss.item(), ce_loss.item(), logits, labels)
        
        computed = metrics.compute()
        pbar.set_postfix({
            'loss': f"{computed['loss']:.4f}",
            'acc': f"{computed['accuracy']:.1f}%",
            'f1': f"{computed['f1_score']:.1f}%"
        })
    
    return metrics


@torch.no_grad()
def evaluate_compression_robustness(model, test_loader, device, output_dir=None):
    """
    Evaluate model under different compression levels.
    Based on FF++ protocol: C0 (none), C23 (medium), C40 (heavy).
    """
    from PIL import Image
    import io
    
    model.eval()
    
    # Test at different JPEG quality levels
    quality_levels = {
        'C0 (No compression)': 100,
        'C23 (Medium)': 75,
        'C40 (Heavy)': 40,
        'C50 (Very Heavy)': 25,
    }
    
    results = {}
    
    for name, quality in quality_levels.items():
        metrics = MetricsTracker()
        
        for batch in tqdm(test_loader, desc=f"Testing {name}", leave=False):
            images = batch['image'].to(device)
            labels = batch['label']
            
            # Apply JPEG compression
            compressed_images = []
            for img_tensor in images:
                # Convert to PIL
                img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                # Compress
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                compressed_pil = Image.open(buffer)
                compressed_np = np.array(compressed_pil) / 255.0
                
                # Normalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                compressed_np = (compressed_np - mean) / std
                compressed_tensor = torch.from_numpy(compressed_np.transpose(2, 0, 1)).float().to(device)
                compressed_images.append(compressed_tensor)
            
            compressed_batch = torch.stack(compressed_images)
            
            logits, _ = model(compressed_batch)
            logits = logits.squeeze(-1)
            
            loss = F.binary_cross_entropy_with_logits(logits, labels.float().to(device))
            metrics.update(loss.item(), loss.item(), logits, labels.to(device))
        
        computed = metrics.compute()
        results[name] = computed
        
        print(f"  {name:<25} Acc: {computed['accuracy']:5.1f}%  F1: {computed['f1_score']:5.1f}%  AUC: {computed['auc_roc']:5.1f}%")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir) / 'compression_robustness.json'
        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for key, val in results.items():
            serializable_results[key] = {k: float(v) for k, v in val.items()}
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nCompression results saved to: {output_path}")
    
    return results


def train(
    config_path: str = '../configs/config.yaml',
    resume_from: str = None,
    data_dir: str = None,
    epochs: int = None,
    batch_size: int = None,
    augmentation: str = "heavy",
    freeze_backbone: bool = True,
    use_metric_learning: bool = True
):
    """Main training function with SOTA methodology."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if data_dir:
        config['data']['root_dir'] = data_dir
    if epochs:
        config['training']['epochs'] = epochs
    if batch_size:
        config['training']['batch_size'] = batch_size
    
    # Set seed and device
    set_seed(config['seed'])
    device = get_device()
    
    # Create output directories
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    log_dir = Path(config['logging']['log_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard_v2'))
    
    print("="*70)
    print("MULTI-STREAM DEEPFAKE DETECTION - PRODUCTION TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Data: {config['data']['root_dir']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['lr']}")
    print(f"Augmentation: {augmentation}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Metric learning: {use_metric_learning}")
    print("="*70)
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=config['data']['root_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            img_size=256,
            augmentation_level=augmentation
        )
    except Exception as e:
        print(f"\nERROR: Could not load data: {e}")
        return
    
    # Create model
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=config['model']['spatial']['feature_dim'],
        freq_output_dim=config['model']['frequency']['feature_dim'],
        semantic_output_dim=config['model']['semantic']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        fusion_attention_heads=config['model']['fusion']['attention_heads'],
        pretrained_backbones=True
    ).to(device)
    
    param_count = model.count_parameters()
    print(f"\nModel created: {param_count:,} parameters ({param_count/1e6:.2f}M)")
    
    # Freeze backbone if enabled
    if freeze_backbone:
        trainable, total = freeze_backbone_params(model)
    
    # Optimizer - only optimize trainable params
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['training']['lr'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0  # No weight decay for LN-tuning (GenD)
    )
    
    # LR scheduler - cosine with warmup
    warmup_epochs = config['training']['warmup_epochs']
    total_epochs = config['training']['epochs']
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # Loss coefficients
    alpha = 0.1  # Alignment weight
    beta = 0.5   # Uniformity weight
    
    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    if resume_from and os.path.exists(resume_from):
        start_epoch, _ = load_checkpoint(model, optimizer, resume_from)
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            alpha=alpha, beta=beta
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, epoch)
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Timing
        epoch_time = time.time() - epoch_start
        
        # Print summary
        train_summary = train_metrics.compute()
        val_summary = val_metrics.compute()
        
        print(f"\nEpoch {epoch}/{total_epochs} | "
              f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  Train: Loss={train_summary['loss']:.4f}, "
              f"CE={train_summary['ce_loss']:.4f}, "
              f"Acc={train_summary['accuracy']:.1f}%, "
              f"F1={train_summary['f1_score']:.1f}%, "
              f"AUC={train_summary['auc_roc']:.1f}%")
        print(f"  Val:   Loss={val_summary['loss']:.4f}, "
              f"CE={val_summary['ce_loss']:.4f}, "
              f"Acc={val_summary['accuracy']:.1f}%, "
              f"F1={val_summary['f1_score']:.1f}%, "
              f"AUC={val_summary['auc_roc']:.1f}%")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_summary['loss'], epoch)
        writer.add_scalar('Loss/val', val_summary['loss'], epoch)
        writer.add_scalar('Loss_CE/train', train_summary['ce_loss'], epoch)
        writer.add_scalar('Loss_CE/val', val_summary['ce_loss'], epoch)
        writer.add_scalar('Accuracy/train', train_summary['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_summary['accuracy'], epoch)
        writer.add_scalar('F1/train', train_summary['f1_score'], epoch)
        writer.add_scalar('F1/val', val_summary['f1_score'], epoch)
        writer.add_scalar('AUC/train', train_summary['auc_roc'], epoch)
        writer.add_scalar('AUC/val', val_summary['auc_roc'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save best model
        if val_summary['accuracy'] > best_val_acc:
            best_val_acc = val_summary['accuracy']
            save_checkpoint(
                model, optimizer, epoch, val_summary['loss'],
                str(checkpoint_dir / 'best_model_v2.pth')
            )
            print(f"  >>> New best model saved (Acc: {best_val_acc:.1f}%)")
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_summary['loss'],
            str(checkpoint_dir / 'latest_checkpoint_v2.pth')
        )
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Best model saved to: {checkpoint_dir / 'best_model_v2.pth'}")
    print("="*70)
    
    writer.close()
    
    # Run compression robustness test
    print("\n" + "="*70)
    print("COMPRESSION ROBUSTNESS TEST")
    print("="*70)
    
    # Load best model
    if os.path.exists(checkpoint_dir / 'best_model_v2.pth'):
        load_checkpoint(model, torch.optim.AdamW(model.parameters()), str(checkpoint_dir / 'best_model_v2.pth'), device)
    
    compression_results = evaluate_compression_robustness(
        model, test_loader, device, str(log_dir / 'evaluation_v2')
    )
    
    return model, best_val_acc, compression_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train with SOTA methodology")
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--augmentation', type=str, default='heavy', choices=['light', 'medium', 'heavy'])
    parser.add_argument('--no-freeze', action='store_true', help='Don\'t freeze backbone')
    parser.add_argument('--no-metric-learning', action='store_true', help='Don\'t use metric learning losses')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        config_path=args.config,
        resume_from=args.resume,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augmentation=args.augmentation,
        freeze_backbone=not args.no_freeze,
        use_metric_learning=not args.no_metric_learning
    )
