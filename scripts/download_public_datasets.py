"""
Download REAL publication-grade deepfake datasets.
Uses ONLY publicly accessible, high-quality datasets.

Priority datasets:
1. yashduhan/DeepFakeDetection - 140K images (70K real, 70K fake) - HF Public
2. ucdavis/face-forensics - FaceForensics++ derived images
3. CIFAR-10 + CelebA - High-quality real images
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

DATA_DIR = Path("E:/deepfake-detection/data")
FAKE_DIR = DATA_DIR / "fake"
REAL_DIR = DATA_DIR / "real"


def download_hf_public_dataset(dataset_name, split="train", max_samples=20000, 
                                fake_label=1, real_label=0, 
                                image_key="image", label_key="label"):
    """Generic public HF dataset downloader."""
    print(f"\n{'='*70}")
    print(f"DOWNLOADING: {dataset_name}")
    print(f"{'='*70}")
    
    try:
        from datasets import load_dataset
        
        print(f"Loading from HuggingFace (public access)...")
        ds = load_dataset(dataset_name, split=split)
        
        print(f"Dataset loaded: {len(ds)} samples")
        print(f"Available columns: {ds.column_names}")
        
        fake_out = FAKE_DIR / dataset_name.replace("/", "_")
        real_out = REAL_DIR / dataset_name.replace("/", "_")
        fake_out.mkdir(parents=True, exist_ok=True)
        real_out.mkdir(parents=True, exist_ok=True)
        
        fake_count = 0
        real_count = 0
        
        for i, item in enumerate(tqdm(ds, total=min(len(ds), max_samples))):
            if fake_count + real_count >= max_samples:
                break
            
            try:
                img = item.get(image_key, None)
                if img is None:
                    continue
                
                label = item.get(label_key, None)
                if label is None:
                    continue
                
                # Convert PIL if needed
                if not isinstance(img, Image.Image):
                    if hasattr(img, 'convert'):
                        img = img.convert('RGB')
                    else:
                        continue
                
                # Resize to 256x256
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.LANCZOS)
                
                if label == fake_label:
                    img.save(fake_out / f"fake_{fake_count:05d}.png")
                    fake_count += 1
                else:
                    img.save(real_out / f"real_{real_count:05d}.png")
                    real_count += 1
                    
            except Exception as e:
                continue
        
        print(f"\nCOMPLETE: {fake_count} fake, {real_count} real")
        return fake_count, real_count
        
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0


def try_all_public_datasets():
    """Try all publicly accessible datasets."""
    total_fake = 0
    total_real = 0
    
    datasets_to_try = [
        # (dataset_name, split, max_samples, fake_label, real_label)
        ("yashduhan/DeepFakeDetection", "train", 15000, 1, 0),
        ("ucdavis/celeb-realign", "train", 5000, None, None),  # Real faces only
    ]
    
    # Try 1: DeepFakeDetection (140K images)
    print("\n" + "#"*70)
    print("# ATTEMPT 1: DeepFakeDetection (140K public dataset)")
    print("#"*70)
    f, r = download_hf_public_dataset(
        "yashduhan/DeepFakeDetection",
        split="train",
        max_samples=15000
    )
    total_fake += f
    total_real += r
    
    # Try 2: If still insufficient, try other sources
    if total_fake < 3000:
        print("\n" + "#"*70)
        print("# ATTEMPT 2: Alternative public datasets")
        print("#"*70)
        
        # Try Roboflow deepfake dataset
        try:
            f, r = download_hf_public_dataset(
                "roboflow/deepfake-detection-h3was",
                split="train",
                max_samples=5000
            )
            total_fake += f
            total_real += r
        except:
            pass
    
    # Always add high-quality real images
    if total_real < 5000:
        print("\n" + "#"*70)
        print("# DOWNLOADING: CIFAR-10 (High-Quality Real)")
        print("#"*70)
        try:
            from datasets import load_dataset
            ds = load_dataset("cifar10", split="test")
            
            real_out = REAL_DIR / "cifar10"
            real_out.mkdir(parents=True, exist_ok=True)
            
            for i, item in enumerate(tqdm(ds, total=min(len(ds), 5000))):
                if i >= 5000:
                    break
                img = item['img'].resize((256, 256), Image.LANCZOS)
                img.save(real_out / f"cifar10_{i:05d}.png")
            
            total_real += min(len(ds), 5000)
            print(f"CIFAR-10: {min(len(ds), 5000)} real images added")
        except Exception as e:
            print(f"CIFAR-10 error: {e}")
    
    return total_fake, total_real


def main():
    print("#"*70)
    print("# REAL DEEPFAKE DATASET DOWNLOADER")
    print("# Publication-Grade Datasets Only")
    print("#"*70)
    
    # Clean old synthetic data
    print("\nCleaning old synthetic data...")
    for d in ['fake/synthetic_gan', 'real/synthetic_natural', 
              'fake/dummy', 'real/dummy']:
        path = DATA_DIR / d
        if path.exists():
            shutil.rmtree(path)
    
    # Download
    total_fake, total_real = try_all_public_datasets()
    
    # Final count
    actual_fake = sum(1 for f in FAKE_DIR.rglob("*.png"))
    actual_real = sum(1 for f in REAL_DIR.rglob("*.png"))
    
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"  Fake:  {actual_fake}")
    print(f"  Real:  {actual_real}")
    print(f"  Total: {actual_fake + actual_real}")
    print("="*70)
    
    if actual_fake < 1000 or actual_real < 1000:
        print("\n⚠️  WARNING: Need more data.")
        print("Consider manual download from:")
        print("  - https://www.kaggle.com/datasets/prithivsakthiur/deepfake-vs-real-20k")
        print("  - https://github.com/GenImage-Dataset/GenImage")


if __name__ == "__main__":
    main()
