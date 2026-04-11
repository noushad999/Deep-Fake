"""
Training script for Multi-Stream Deepfake Detection.
Includes: early stopping, LR scheduling, TensorBoard logging, checkpointing.
"""
import os
import sys
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
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


class MetricsTracker:
    """Tracks and computes training/evaluation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_labels = []
    
    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        self.losses.append(loss)
        preds = (torch.sigmoid(logits) > 0.5).long().squeeze()
        self.correct += (preds == labels.long()).sum().item()
        self.total += labels.size(0)
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        avg_loss = np.mean(self.losses)
        accuracy = 100.0 * self.correct / max(self.total, 1)
        
        # Compute precision, recall, f1
        tp = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(self.all_preds, self.all_labels) if p == 0 and l == 1)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100
        }


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'max':
            improvement = current_value > self.best_value + self.min_delta
        else:
            improvement = current_value < self.best_value - self.min_delta
        
        if improvement:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    total_epochs: int
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer with warmup and cosine decay."""
    
    # Separate weight decay for bias/layer norm
    optimizer_grouped = [
        {
            'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n.lower()],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n.lower()],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    
    # Warmup + Cosine annealing
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=lr * 0.01)
    
    scheduler = SequentialLR(optimizer, 
                            schedulers=[warmup_scheduler, cosine_scheduler],
                            milestones=[warmup_epochs])
    
    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> MetricsTracker:
    """Single training epoch."""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        # Forward
        logits, _ = model(images)
        logits = logits.squeeze(-1)
        
        # Compute loss (with gradient accumulation support)
        loss = criterion(logits, labels) / gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics (use loss * accumulation_steps for reporting)
        metrics.update(loss.item() * gradient_accumulation_steps, logits, labels)
        
        # Update progress bar
        computed = metrics.compute()
        pbar.set_postfix({
            'loss': f"{computed['loss']:.4f}",
            'acc': f"{computed['accuracy']:.1f}%"
        })
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> MetricsTracker:
    """Validation epoch."""
    model.eval()
    metrics = MetricsTracker()
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        logits, _ = model(images)
        logits = logits.squeeze(-1)
        
        loss = criterion(logits, labels)
        metrics.update(loss.item(), logits, labels)
        
        computed = metrics.compute()
        pbar.set_postfix({
            'loss': f"{computed['loss']:.4f}",
            'acc': f"{computed['accuracy']:.1f}%"
        })
    
    return metrics


def train(
    config_path: str = '../configs/config.yaml',
    resume_from: str = None,
    data_dir: str = None,
    epochs: int = None,
    batch_size: int = None
):
    """Main training function."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args
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
    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
    
    print("="*70)
    print("MULTI-STREAM DEEPFAKE DETECTION - TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Data: {config['data']['root_dir']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['lr']}")
    print("="*70)
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=config['data']['root_dir'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            img_size=256,
            augmentation_level=config['training'].get('augmentation_level', 'medium')
        )
    except Exception as e:
        print(f"\nWARNING: Could not load real data: {e}")
        print("Create dummy data for testing with:")
        print("  python scripts/create_dummy_data.py")
        return
    
    # Create model
    model = MultiStreamDeepfakeDetector(
        spatial_output_dim=config['model']['spatial']['feature_dim'],
        freq_output_dim=config['model']['frequency']['feature_dim'],
        semantic_output_dim=config['model']['semantic']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        fusion_attention_heads=config['model']['fusion']['attention_heads'],
        pretrained_backbones=config['model']['spatial']['pretrained']
    ).to(device)
    
    param_count = model.count_parameters()
    print(f"\nModel created: {param_count:,} parameters ({param_count/1e6:.2f}M)")
    
    # Loss function
    if config['training']['loss']['type'] == 'binary_cross_entropy':
        criterion = nn.BCEWithLogitsLoss()
    elif config['training']['loss']['type'] == 'weighted_bce':
        pos_weight = torch.tensor([1.5]).to(device)  # Adjust based on class balance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        warmup_epochs=config['training']['warmup_epochs'],
        total_epochs=config['training']['epochs']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        mode='max'
    )
    
    # Resume from checkpoint if specified
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
    
    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            gradient_accumulation_steps=config['training'].get('gradient_accumulation', 1),
            max_grad_norm=config['training'].get('max_grad_norm', 1.0)
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Timing
        epoch_time = time.time() - epoch_start
        
        # Print summary
        train_summary = train_metrics.compute()
        val_summary = val_metrics.compute()
        
        print(f"\nEpoch {epoch}/{config['training']['epochs']} | "
              f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  Train: Loss={train_summary['loss']:.4f}, "
              f"Acc={train_summary['accuracy']:.1f}%, "
              f"F1={train_summary['f1_score']:.1f}%")
        print(f"  Val:   Loss={val_summary['loss']:.4f}, "
              f"Acc={val_summary['accuracy']:.1f}%, "
              f"F1={val_summary['f1_score']:.1f}%")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_summary['loss'], epoch)
        writer.add_scalar('Loss/val', val_summary['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_summary['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_summary['accuracy'], epoch)
        writer.add_scalar('F1/train', train_summary['f1_score'], epoch)
        writer.add_scalar('F1/val', val_summary['f1_score'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Save best model
        if val_summary['accuracy'] > best_val_acc:
            best_val_acc = val_summary['accuracy']
            save_checkpoint(
                model, optimizer, epoch, val_summary['loss'],
                str(checkpoint_dir / 'best_model.pth')
            )
            print(f"  >>> New best model saved (Acc: {best_val_acc:.1f}%)")
        
        # Save latest checkpoint (for resume)
        save_checkpoint(
            model, optimizer, epoch, val_summary['loss'],
            str(checkpoint_dir / 'latest_checkpoint.pth')
        )
        
        # Early stopping check
        if early_stopping(val_summary['accuracy']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    print(f"TensorBoard logs: {log_dir / 'tensorboard'}")
    print("="*70)
    
    writer.close()
    
    return model, best_val_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Stream Deepfake Detector")
    parser.add_argument('--config', type=str, default='../configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--data-dir', type=str, default=None, help='Override data directory')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        config_path=args.config,
        resume_from=args.resume,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
