"""
Baseline Training Script — CNNDetect and UnivFD
================================================
Trains comparison baselines on the same dataset as our main model.
Follows Wang et al. 2020 training protocol for CNNDetect.
Follows Ojha et al. 2023 training protocol for UnivFD.

Usage:
  python scripts/train_baseline.py --model cnndetect --config configs/config.yaml
  python scripts/train_baseline.py --model univfd   --config configs/config.yaml
"""
import os
import sys
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baselines import CNNDetect, UnivFD
from data.dataset import create_dataloaders
from utils.utils import set_seed, get_device


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    losses, all_scores, all_labels = [], [], []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()
        use_amp = scaler is not None

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits.squeeze(-1), labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())
        scores = torch.sigmoid(logits.detach()).squeeze(-1).cpu().numpy()
        all_scores.extend(scores.tolist() if scores.ndim > 0 else [float(scores)])
        all_labels.extend(labels.cpu().numpy().tolist())

    auc = roc_auc_score(all_labels, all_scores) * 100 if len(set(all_labels)) > 1 else 0.0
    return float(np.mean(losses)), auc


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    losses, all_scores, all_labels = [], [], []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(images)
            loss = criterion(logits.squeeze(-1), labels)

        losses.append(loss.item())
        scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_scores.extend(scores.tolist() if scores.ndim > 0 else [float(scores)])
        all_labels.extend(labels.cpu().numpy().tolist())

    auc = roc_auc_score(all_labels, all_scores) * 100 if len(set(all_labels)) > 1 else 0.0
    return float(np.mean(losses)), auc


def train(model_name: str, config_path: str, data_dir: str = None, seed: int = 42):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if data_dir:
        config['data']['root_dir'] = data_dir

    set_seed(seed)
    device = get_device()

    ckpt_dir = Path(config['logging']['checkpoint_dir']) / model_name / f'seed{seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = Path(config['logging']['log_dir']) / 'tensorboard' / model_name
    writer   = SummaryWriter(log_dir=str(log_dir))

    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING: {model_name.upper()}  (seed={seed})")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=config['data']['root_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        img_size=256
    )

    # Build model
    if model_name == 'cnndetect':
        model = CNNDetect(pretrained=True).to(device)
        # Wang et al. use SGD with lr=0.0001, momentum=0.9
        optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        epochs = 30
    elif model_name == 'univfd':
        model = UnivFD().to(device)
        # Ojha et al. only train the linear layer, use AdamW
        optimizer = AdamW(model.linear.parameters(), lr=1e-3, weight_decay=1e-4)
        epochs = 20
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Trainable params: {model.count_parameters():,}")

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_auc   = 0.0
    patience   = 8
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_auc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{epochs} | {elapsed:.0f}s | "
              f"Train AUC={tr_auc:.2f}%  Val AUC={vl_auc:.2f}%  Val Loss={vl_loss:.4f}")

        writer.add_scalars('AUC', {'train': tr_auc, 'val': vl_auc}, epoch)

        if vl_auc > best_auc + 0.01:
            best_auc   = vl_auc
            no_improve = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_auc': vl_auc}, ckpt_dir / 'best_model.pth')
            print(f"  >>> Best saved (AUC={best_auc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest Val AUC: {best_auc:.2f}%")
    print(f"Checkpoint: {ckpt_dir / 'best_model.pth'}")
    writer.close()
    return best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    required=True, choices=['cnndetect', 'univfd'])
    parser.add_argument('--config',   default='configs/config.yaml')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    train(args.model, args.config, args.data_dir, args.seed)


if __name__ == '__main__':
    main()
