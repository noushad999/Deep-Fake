"""
So-Fake Dataset Loaders (saberzl/So-Fake-Set & So-Fake-OOD)
=============================================================
So-Fake-Set:  1M+ images — real, full_synthetic, tampered — with binary masks.
              Designed for social media image forgery detection.
              arxiv: 2505.18660

So-Fake-OOD:  Test-only OOD benchmark from real Reddit content.
              Evaluates generalization to real-world distribution.
              Perfect as held-out benchmark (no training on this).

CVPR Usage:
  Train:  So-Fake-Set (real + full_synthetic classes)
  Test:   So-Fake-OOD (real-world OOD generalization)

HF: https://huggingface.co/datasets/saberzl/So-Fake-Set
    https://huggingface.co/datasets/saberzl/So-Fake-OOD
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# So-Fake label mapping
# String format (So-Fake-Set): real=0, full_synthetic=1, tampered=1
# Numeric format (So-Fake-OOD): 0=real, 1=full_synthetic, 2=tampered
SOFAKE_LABEL_MAP = {
    "real": 0,
    "full_synthetic": 1,
    "tampered": 1,
    0: 0,   # numeric: real
    1: 1,   # numeric: full_synthetic
    2: 1,   # numeric: tampered → fake
}


class SoFakeDataset(Dataset):
    """
    So-Fake-Set from HuggingFace. Parquet format.
    Supports: real, full_synthetic, tampered categories.

    For binary deepfake detection: real=0, (full_synthetic + tampered)=1.
    """

    def __init__(
        self,
        split: str = "train",
        categories: Optional[List[str]] = None,
        transform=None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            split: HF dataset split ('train' or 'test' if available).
            categories: ['real', 'full_synthetic', 'tampered'] or subset.
            max_samples: Cap total samples (balanced real/fake).
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        self.transform = transform
        self.categories = categories or ["real", "full_synthetic"]
        rng = np.random.RandomState(seed)

        print(f"[So-Fake-Set] Loading split='{split}' categories={self.categories}")
        raw = load_dataset("saberzl/So-Fake-Set", split=split, trust_remote_code=True)

        # Filter by category
        raw = raw.filter(lambda x: x.get("label", "") in self.categories)

        if max_samples:
            n_per_class = max_samples // 2
            real_data = raw.filter(lambda x: x.get("label", "") == "real")
            fake_data = raw.filter(lambda x: x.get("label", "") != "real")
            n_real = min(len(real_data), n_per_class)
            n_fake = min(len(fake_data), n_per_class)
            real_idx = rng.choice(len(real_data), n_real, replace=False)
            fake_idx = rng.choice(len(fake_data), n_fake, replace=False)
            from datasets import concatenate_datasets
            raw = concatenate_datasets([real_data.select(real_idx),
                                        fake_data.select(fake_idx)])

        self.data = raw
        real_n = sum(1 for i in range(len(raw)) if raw[i].get("label") == "real")
        fake_n = len(raw) - real_n
        print(f"  Real: {real_n:,} | Fake: {fake_n:,} | Total: {len(raw):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item.get("image")
        raw_label = item.get("label", 1)
        label = SOFAKE_LABEL_MAP.get(raw_label, 1)
        generator = item.get("generator") or str(raw_label)

        if img is None:
            img_np = np.zeros((256, 256, 3), dtype=np.uint8)
        elif isinstance(img, bytes):
            import io
            img_np = np.array(Image.open(io.BytesIO(img)).convert("RGB"))
        elif isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = np.array(img)

        if self.transform:
            img_tensor = self.transform(image=img_np)["image"]
        else:
            tf = A.Compose([A.Resize(256, 256),
                            A.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
            img_tensor = tf(image=img_np)["image"]

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "generator": generator,
            "path": f"sofake_{idx}",
        }


class SoFakeOODDataset(Dataset):
    """
    So-Fake-OOD: real-world out-of-distribution test benchmark.
    Collected from real Reddit posts. DO NOT train on this.
    Use only for final generalization evaluation.
    """

    def __init__(self, transform=None, max_samples: Optional[int] = None,
                 seed: int = 42):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        self.transform = transform
        rng = np.random.RandomState(seed)

        print("[So-Fake-OOD] Loading OOD test benchmark...")
        raw = load_dataset("saberzl/So-Fake-OOD", split="test")

        if max_samples and len(raw) > max_samples:
            idx = rng.choice(len(raw), max_samples, replace=False)
            raw = raw.select(idx)

        self.data = raw
        real_n = sum(1 for i in range(min(len(raw), 1000))
                     if raw[i].get("label", "") == "real")
        print(f"  Total: {len(raw):,} | (estimated real~{real_n/10:.0f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_label = item.get("label", 1)  # numeric: 0=real, 1=synthetic, 2=tampered
        label = SOFAKE_LABEL_MAP.get(raw_label, 1)
        img = item.get("image")

        if img is None:
            img_np = np.zeros((256, 256, 3), dtype=np.uint8)
        elif isinstance(img, bytes):
            import io
            img_np = np.array(Image.open(io.BytesIO(img)).convert("RGB"))
        elif isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = np.array(img)

        if self.transform:
            img_tensor = self.transform(image=img_np)["image"]
        else:
            tf = A.Compose([A.Resize(256, 256),
                            A.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                            ToTensorV2()])
            img_tensor = tf(image=img_np)["image"]

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "generator": item.get("generator", "unknown"),
            "path": f"sofake_ood_{idx}",
        }


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def _get_tf(phase, img_size=256):
    norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if phase == "train":
        return A.Compose([
            A.Resize(img_size + 32, img_size + 32),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, p=0.4),
            A.ImageCompression(quality_range=(75, 100), p=0.3),
            norm, ToTensorV2(),
        ])
    return A.Compose([A.Resize(img_size, img_size), norm, ToTensorV2()])


def create_sofake_loaders(
    train_max: int = 100_000,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    seed: int = 42,
):
    """Train on So-Fake-Set, eval on So-Fake-OOD."""
    train_ds = SoFakeDataset(split="train", max_samples=train_max,
                              transform=_get_tf("train", img_size), seed=seed)
    ood_ds = SoFakeOODDataset(transform=_get_tf("test", img_size), seed=seed)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
            DataLoader(ood_ds, shuffle=False, **kw))


def create_fakecoco_loaders(
    train_generators=None,
    crossgen_generators=None,
    max_per_generator: int = 10_000,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    seed: int = 42,
):
    """
    Train on older generators, test on newer ones.
    Perfect for cross-generator generalization experiments.
    """
    from .hf_fakecoco import (FakeCocoHFDataset,
                               TRAIN_GENERATORS, CROSS_GEN_GENERATORS)

    train_gens = train_generators or TRAIN_GENERATORS
    test_gens = crossgen_generators or CROSS_GEN_GENERATORS

    train_ds = FakeCocoHFDataset(
        split="train", generators=train_gens,
        transform=_get_tf("train", img_size),
        max_per_generator=max_per_generator, seed=seed,
    )
    crossgen_ds = FakeCocoHFDataset(
        split="test", generators=test_gens,
        transform=_get_tf("test", img_size),
        max_per_generator=max_per_generator, seed=seed,
    )

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
            DataLoader(crossgen_ds, shuffle=False, **kw))
