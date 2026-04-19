"""
CVPR Training Pipeline — HuggingFace Datasets
===============================================
Training + evaluation using freely available HF datasets:

  Train:     Existing dataset (SD v1.x fakes + FFHQ real) OR FakeCOCO
  OOD Test:  So-Fake-OOD — real Reddit images, diverse generators
             (openai/DALL-E, Seedream3.0, MidJourney, etc. — never seen in training)

Evaluation settings for CVPR:
  1. In-distribution AUC (same generator family as training)
  2. OOD AUC (So-Fake-OOD — completely different generator architectures)
  3. Per-generator breakdown (openai vs Seedream vs others)

So-Fake-OOD is the STRONGEST cross-dataset test:
  - Different generators (DALL-E, Seedream vs our SD v1.x training)
  - Different image content (Reddit vs curated faces)
  - Real-world distribution (not curated datasets)

Usage:
    python scripts/train_cvpr.py --data-dir data
    python scripts/train_cvpr.py --data-dir data --use-fakecoco
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from utils.utils import set_seed, get_device, save_checkpoint


def compute_auc(labels, scores):
    labels, scores = np.array(labels), np.array(scores)
    if len(set(labels.tolist())) < 2:
        return 0.0
    try:
        return roc_auc_score(labels, scores) * 100.0
    except Exception:
        return 0.0


def compute_acc(labels, scores, t=0.5):
    return 100.0 * ((np.array(scores) > t).astype(int) == np.array(labels)).mean()


@torch.no_grad()
def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    labels, scores, gens = [], [], []
    for batch in tqdm(loader, desc=f"  {desc}", leave=False):
        imgs = batch["image"].to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(imgs)
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        labels.extend(batch["label"].numpy().tolist())
        scores.extend(probs.tolist())
        gens.extend(batch.get("generator", ["unknown"] * len(probs)))

    auc = compute_auc(labels, scores)
    acc = compute_acc(labels, scores)

    # Per-generator breakdown
    per_gen = {}
    for g in set(gens):
        if g == "real":
            continue
        idx = [i for i, x in enumerate(gens) if x == g]
        if len(idx) > 10:
            g_labels = [labels[i] for i in idx]
            g_scores = [scores[i] for i in idx]
            per_gen[g] = {"auc": compute_auc(g_labels, g_scores),
                          "n": len(idx)}

    return {"auc": auc, "acc": acc, "per_gen": per_gen}


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        imgs = batch["image"].to(device)
        lbls = batch["label"].to(device)
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits, _ = model(imgs)
            loss = criterion(logits.squeeze(-1), lbls)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-mode", default="hf", choices=["hf", "local"])
    parser.add_argument("--data-dir", default="data/cvpr_datasets",
                        help="Used when --data-mode=local")
    parser.add_argument("--train-generators", nargs="+",
                        default=["SD15", "SD21", "PixArt-alpha", "UniDiffuser"])
    parser.add_argument("--crossgen-generators", nargs="+",
                        default=["SDXL", "SD3", "Flux.1", "Playground2.5"])
    parser.add_argument("--max-per-gen", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints/cvpr")
    parser.add_argument("--ablation-mode", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CVPR TRAINING PIPELINE")
    print(f"  Train generators:    {args.train_generators}")
    print(f"  Cross-gen generators:{args.crossgen_generators}")
    print(f"  Device: {device} | Seed: {args.seed}")
    print("=" * 65)

    # Build data loaders
    if args.data_mode == "hf":
        from data.hf_fakecoco import FakeCocoHFDataset
        from data.hf_sofake import SoFakeOODDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        def tf(phase):
            norm = A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            if phase == "train":
                return A.Compose([A.Resize(288,288), A.RandomCrop(256,256),
                                   A.HorizontalFlip(p=0.5), norm, ToTensorV2()])
            return A.Compose([A.Resize(256,256), norm, ToTensorV2()])

        train_ds = FakeCocoHFDataset(
            split="train", generators=args.train_generators,
            transform=tf("train"), max_per_generator=args.max_per_gen,
            seed=args.seed,
        )
        crossgen_ds = FakeCocoHFDataset(
            split="test", generators=args.crossgen_generators,
            transform=tf("test"), max_per_generator=args.max_per_gen,
            seed=args.seed,
        )
        ood_ds = SoFakeOODDataset(transform=tf("test"), seed=args.seed)
    else:
        from data.hf_fakecoco import FakeCocoLocalDataset
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        def tf(phase):
            norm = A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            if phase == "train":
                return A.Compose([A.Resize(288,288), A.RandomCrop(256,256),
                                   A.HorizontalFlip(p=0.5), norm, ToTensorV2()])
            return A.Compose([A.Resize(256,256), norm, ToTensorV2()])

        train_ds = FakeCocoLocalDataset(
            root_dir=str(Path(args.data_dir) / "fakecoco"),
            generators=args.train_generators, split="train",
            transform=tf("train"), max_per_generator=args.max_per_gen,
            seed=args.seed,
        )
        crossgen_ds = FakeCocoLocalDataset(
            root_dir=str(Path(args.data_dir) / "fakecoco"),
            generators=args.crossgen_generators, split="test",
            transform=tf("test"), max_per_generator=args.max_per_gen,
            seed=args.seed,
        )
        from data.hf_sofake import SoFakeOODDataset
        ood_ds = SoFakeOODDataset(transform=tf("test"), seed=args.seed)

    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, drop_last=True, **kw)
    crossgen_loader = torch.utils.data.DataLoader(
        crossgen_ds, shuffle=False, **kw)
    ood_loader = torch.utils.data.DataLoader(
        ood_ds, shuffle=False, **kw)

    # Model
    model = MultiStreamDeepfakeDetector(
        pretrained_backbones=True,
        ablation_mode=args.ablation_mode,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params/1e6:.1f}M params")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_auc = 0.0
    patience = 8
    patience_counter = 0
    best_ckpt = ckpt_dir / "best_cvpr.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        scheduler.step()

        # Evaluate on in-dist (subset of train generators test split)
        indist = evaluate(model, crossgen_loader, device, "Cross-gen")

        print(f"Epoch {epoch:>2}/{args.epochs} | {time.time()-t0:.0f}s | "
              f"loss={loss:.4f} | cross-gen AUC={indist['auc']:.2f}%")

        if indist["auc"] > best_auc:
            best_auc = indist["auc"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, loss, str(best_ckpt))
            print(f"  >>> Best saved (AUC={best_auc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final evaluation
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    crossgen_m = evaluate(model, crossgen_loader, device, "Cross-Generator")
    ood_m = evaluate(model, ood_loader, device, "So-Fake-OOD")

    print(f"\nCross-Generator (unseen: {args.crossgen_generators}):")
    print(f"  AUC: {crossgen_m['auc']:.4f}% | Acc: {crossgen_m['acc']:.2f}%")
    if crossgen_m["per_gen"]:
        print("  Per-generator:")
        for g, v in sorted(crossgen_m["per_gen"].items()):
            print(f"    {g:<20}: AUC={v['auc']:.2f}% (n={v['n']})")

    print(f"\nOOD (So-Fake-OOD, real Reddit):")
    print(f"  AUC: {ood_m['auc']:.4f}% | Acc: {ood_m['acc']:.2f}%")

    # Save results
    out = ckpt_dir / f"cvpr_results_seed{args.seed}.txt"
    with open(out, "w") as f:
        f.write(f"Train generators: {args.train_generators}\n")
        f.write(f"Cross-gen generators: {args.crossgen_generators}\n\n")
        f.write(f"Cross-Generator AUC: {crossgen_m['auc']:.4f}%\n")
        f.write(f"OOD AUC: {ood_m['auc']:.4f}%\n")
        if crossgen_m["per_gen"]:
            f.write("\nPer-generator:\n")
            for g, v in sorted(crossgen_m["per_gen"].items()):
                f.write(f"  {g}: {v['auc']:.4f}%\n")
    print(f"\nResults saved: {out}")


if __name__ == "__main__":
    main()
