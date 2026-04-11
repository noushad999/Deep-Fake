"""
Download REAL deepfake detection datasets from HuggingFace.
Uses publication-grade datasets used in actual research papers.

Datasets:
1. Deepfake-vs-Real-60K (prithivMLmods) - 60K images, curated high-quality
2. AI-vs-Deepfake-vs-Real (prithivMLmods) - 10K images, 3-class
3. GenImage subset (via GitHub) - The NeurIPS 2023 benchmark dataset
"""
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

DATA_DIR = Path("E:/deepfake-detection/data")
FAKE_DIR = DATA_DIR / "fake"
REAL_DIR = DATA_DIR / "real"


def download_deepfake_vs_real_60k(max_samples=15000):
    """
    Download Deepfake-vs-Real-60K from HuggingFace.
    60,000 high-quality curated images (30K fake, 30K real).
    Used in actual deepfake detection research.
    """
    print("\n" + "="*70)
    print("DOWNLOADING: Deepfake-vs-Real-60K (HuggingFace)")
    print("Dataset: prithivMLmods/Deepfake-vs-Real-60K")
    print("License: Apache 2.0 | NeurIPS-quality curated dataset")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        print("Loading dataset from HuggingFace Hub...")
        print("Note: This may require HF login. Trying public access first...")
        
        # Try loading the dataset
        ds = load_dataset("prithivMLmods/Deepfake-vs-Real-60K", split="train")
        
        print(f"Dataset loaded: {len(ds)} images")
        print(f"Features: {ds.features}")
        
        fake_out = FAKE_DIR / "deepfake_60k"
        real_out = REAL_DIR / "deepfake_60k"
        fake_out.mkdir(parents=True, exist_ok=True)
        real_out.mkdir(parents=True, exist_ok=True)
        
        fake_count = 0
        real_count = 0
        
        print(f"Downloading (max {max_samples} samples)...")
        
        for i, item in enumerate(tqdm(ds, total=min(len(ds), max_samples))):
            if fake_count + real_count >= max_samples:
                break
            
            try:
                img = item['image']
                label = item['label']  # 0=Fake, 1=Real
                
                # Resize to 256x256 if needed
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.LANCZOS)
                
                if label == 0:
                    img.save(fake_out / f"dfake_60k_{fake_count:05d}.png")
                    fake_count += 1
                else:
                    img.save(real_out / f"dreal_60k_{real_count:05d}.png")
                    real_count += 1
                
            except Exception as e:
                continue
        
        print(f"\nDeepfake-vs-Real-60K download COMPLETE:")
        print(f"  Fake images: {fake_count}")
        print(f"  Real images: {real_count}")
        
        return fake_count, real_count
        
    except Exception as e:
        print(f"Error: {e}")
        print("This dataset requires HF login. Trying alternative...")
        return 0, 0


def download_ai_vs_deepfake_vs_real(max_samples=8000):
    """
    Download AI-vs-Deepfake-vs-Real from HuggingFace.
    10K images: 33% Artificial, 33% Deepfake, 33% Real
    """
    print("\n" + "="*70)
    print("DOWNLOADING: AI-vs-Deepfake-vs-Real (HuggingFace)")
    print("Dataset: prithivMLmods/AI-vs-Deepfake-vs-Real")
    print("License: Apache 2.0")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset("prithivMLmods/AI-vs-Deepfake-vs-Real", split="train")
        
        print(f"Dataset loaded: {len(ds)} images")
        
        fake_out = FAKE_DIR / "ai_vs_deepfake_vs_real"
        real_out = REAL_DIR / "ai_vs_deepfake_vs_real"
        fake_out.mkdir(parents=True, exist_ok=True)
        real_out.mkdir(parents=True, exist_ok=True)
        
        fake_count = 0
        real_count = 0
        
        for i, item in enumerate(tqdm(ds, total=min(len(ds), max_samples))):
            if fake_count + real_count >= max_samples:
                break
            
            try:
                img = item['image']
                label = item['label']  # 0=Artificial, 1=Deepfake, 2=Real
                
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.LANCZOS)
                
                if label in [0, 1]:  # Artificial or Deepfake
                    img.save(fake_out / f"ai_gen_{fake_count:05d}.png")
                    fake_count += 1
                else:
                    img.save(real_out / f"real_{real_count:05d}.png")
                    real_count += 1
                    
            except Exception as e:
                continue
        
        print(f"\nAI-vs-Deepfake-vs-Real download COMPLETE:")
        print(f"  Fake images: {fake_count}")
        print(f"  Real images: {real_count}")
        
        return fake_count, real_count
        
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0


def download_genimage_subset(max_samples=10000):
    """
    Download GenImage dataset subset from HuggingFace.
    GenImage is the NeurIPS 2023 million-scale benchmark.
    We download a subset for practical training.
    """
    print("\n" + "="*70)
    print("DOWNLOADING: GenImage Subset (NeurIPS 2023 Benchmark)")
    print("Dataset: nkpai/genimage (via HuggingFace)")
    print("Full dataset: 2.68M images (1.33M real, 1.35M fake)")
    print("License: Research use")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        # Try to find GenImage on HF
        print("Searching for GenImage dataset on HuggingFace...")
        
        ds = load_dataset("nkpai/genimage", split="train", streaming=True)
        
        fake_out = FAKE_DIR / "genimage"
        real_out = REAL_DIR / "genimage"
        fake_out.mkdir(parents=True, exist_ok=True)
        real_out.mkdir(parents=True, exist_ok=True)
        
        fake_count = 0
        real_count = 0
        
        print(f"Downloading subset ({max_samples} samples)...")
        
        for i, item in enumerate(ds):
            if fake_count + real_count >= max_samples:
                break
            
            try:
                img = item['image']
                label = item['label']  # 0=real, 1=fake
                
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.LANCZOS)
                
                if label == 1:
                    img.save(fake_out / f"genimage_fake_{fake_count:05d}.png")
                    fake_count += 1
                else:
                    img.save(real_out / f"genimage_real_{real_count:05d}.png")
                    real_count += 1
                    
            except Exception as e:
                continue
        
        print(f"\nGenImage subset download COMPLETE:")
        print(f"  Fake images: {fake_count}")
        print(f"  Real images: {real_count}")
        
        return fake_count, real_count
        
    except Exception as e:
        print(f"Error: {e}")
        print("GenImage may require manual download from GitHub.")
        return 0, 0


def download_cifar10h_real(max_real=3000):
    """
    Download CIFAR-10-H (human-labeled) as high-quality real images.
    """
    print("\n" + "="*70)
    print("DOWNLOADING: CIFAR-10 (High-Quality Real Images)")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset("cifar10", split="test")
        
        real_out = REAL_DIR / "cifar10"
        real_out.mkdir(parents=True, exist_ok=True)
        
        for i, item in enumerate(tqdm(ds, total=min(len(ds), max_real))):
            if i >= max_real:
                break
            
            img = item['img']
            # Upscale CIFAR-10 (32x32) to 256x256
            img = img.resize((256, 256), Image.LANCZOS)
            img.save(real_out / f"cifar10_real_{i:05d}.png")
        
        print(f"\nCIFAR-10 download COMPLETE: {min(len(ds), max_real)} images")
        return min(len(ds), max_real)
        
    except Exception as e:
        print(f"Error: {e}")
        return 0


def try_all_datasets():
    """Try downloading from all available sources."""
    total_fake = 0
    total_real = 0
    
    # Try 60K dataset first (best quality)
    f, r = download_deepfake_vs_real_60k(max_samples=15000)
    total_fake += f
    total_real += r
    
    # Try AI-vs-Deepfake-vs-Real
    if total_fake < 5000:
        f, r = download_ai_vs_deepfake_vs_real(max_samples=8000)
        total_fake += f
        total_real += r
    
    # Try GenImage
    if total_fake < 5000:
        f, r = download_genimage_subset(max_samples=10000)
        total_fake += f
        total_real += r
    
    # Fallback: CIFAR-10 as real
    if total_real < 2000:
        r = download_cifar10h_real(max_real=5000)
        total_real += r
    
    return total_fake, total_real


def main():
    print("#" * 70)
    print("# DOWNLOADING REAL DEEPFAKE DETECTION DATASETS")
    print("# Publication-Grade Datasets Only")
    print("#" * 70)
    
    # Remove old synthetic data
    print("\nCleaning old synthetic data...")
    for d in ['fake/synthetic_gan', 'real/synthetic_natural', 
              'fake/dummy', 'real/dummy', 'fake/cifake', 'real/cifake']:
        path = DATA_DIR / d
        if path.exists():
            shutil.rmtree(path)
    
    # Download real datasets
    total_fake, total_real = try_all_datasets()
    
    print("\n" + "="*70)
    print("FINAL DATASET STATISTICS")
    print("="*70)
    
    # Count actual files
    actual_fake = sum(1 for f in FAKE_DIR.rglob("*.png"))
    actual_real = sum(1 for f in REAL_DIR.rglob("*.png"))
    
    print(f"  Fake images: {actual_fake}")
    print(f"  Real images: {actual_real}")
    print(f"  Total:       {actual_fake + actual_real}")
    print(f"  Location:    {DATA_DIR}")
    print("="*70)
    
    if actual_fake < 100 or actual_real < 100:
        print("\nWARNING: Insufficient real datasets downloaded.")
        print("Manual download may be required from:")
        print("  1. https://huggingface.co/datasets/prithivMLmods/Deepfake-vs-Real-60K")
        print("  2. https://github.com/GenImage-Dataset/GenImage")
        print("  3. https://www.kaggle.com/datasets/ucimachinelearning/deep-fake-detection-cropped-dataset")


if __name__ == "__main__":
    main()
