"""
Efficient dataset download and preparation script.
Downloads real deepfake detection datasets from HuggingFace.
"""
import os
import sys
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

DATA_DIR = Path("E:/deepfake-detection/data")
FAKE_DIR = DATA_DIR / "fake"
REAL_DIR = DATA_DIR / "real"


def download_cifake_huggingface(max_samples=8000):
    """
    Download CIFake dataset from HuggingFace.
    CIFake contains both real (CIFAR-10) and fake (Stable Diffusion) images.
    """
    print("\n" + "="*70)
    print("DOWNLOADING: CIFake Dataset from HuggingFace")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        print("Loading CIFake dataset (this may take a few minutes)...")
        ds = load_dataset("CIFake", split="train", streaming=False)
        
        fake_out = FAKE_DIR / "cifake"
        real_out = REAL_DIR / "cifake"
        fake_out.mkdir(parents=True, exist_ok=True)
        real_out.mkdir(parents=True, exist_ok=True)
        
        fake_count = 0
        real_count = 0
        
        print(f"Dataset size: {len(ds)} images")
        
        for i, item in enumerate(tqdm(ds, total=len(ds), desc="Downloading")):
            if fake_count + real_count >= max_samples:
                break
            
            img = item['image']
            label = item['label']  # 0 = real (CIFAR-10), 1 = fake (generated)
            
            if label == 1:
                filename = f"cifake_fake_{fake_count:05d}.png"
                img.save(fake_out / filename)
                fake_count += 1
            else:
                filename = f"cifake_real_{real_count:05d}.png"
                img.save(real_out / filename)
                real_count += 1
            
            if (fake_count + real_count) % 1000 == 0:
                print(f"  Progress: {fake_count} fake, {real_count} real")
        
        print(f"\nCIFake download COMPLETE:")
        print(f"  Fake images: {fake_count}")
        print(f"  Real images: {real_count}")
        
        return fake_count, real_count
        
    except Exception as e:
        print(f"Error downloading CIFake: {e}")
        return 0, 0


def download_imagenet_subset(max_real=4000):
    """
    Download real images from ImageNet subset via HuggingFace.
    """
    print("\n" + "="*70)
    print("DOWNLOADING: Real Images (ImageNet Subset)")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        print("Loading ImageNet subset...")
        # Use imagenet-1k validation set (smaller, 50K images)
        ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        
        real_out = REAL_DIR / "imagenet_subset"
        real_out.mkdir(parents=True, exist_ok=True)
        
        saved = 0
        for i, item in enumerate(tqdm(ds, total=max_real, desc="Downloading")):
            if saved >= max_real:
                break
            
            img = item['image']
            # Convert grayscale to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            filename = f"imagenet_real_{saved:05d}.png"
            img.save(real_out / filename)
            saved += 1
            
            if saved % 500 == 0:
                print(f"  Saved {saved}/{max_real} real images")
        
        print(f"\nImageNet subset download COMPLETE: {saved} images")
        return saved
        
    except Exception as e:
        print(f"Error downloading ImageNet: {e}")
        print("Will use alternative source...")
        return 0


def download_alt_real_images(max_real=3000):
    """
    Alternative: Download real images from CIFAR-100 (natural images).
    """
    print("\n" + "="*70)
    print("DOWNLOADING: CIFAR-100 (Real Images)")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset("uoft-cs/cifar100", split="test")
        
        real_out = REAL_DIR / "cifar100"
        real_out.mkdir(parents=True, exist_ok=True)
        
        for i, item in enumerate(tqdm(ds, total=min(len(ds), max_real), desc="Downloading")):
            if i >= max_real:
                break
            
            img = item['img']
            # CIFAR images are small (32x32), upscale to 256x256
            img = img.resize((256, 256), Image.LANCZOS)
            
            filename = f"cifar100_real_{i:05d}.png"
            img.save(real_out / filename)
        
        print(f"\nCIFAR-100 download COMPLETE: {min(len(ds), max_real)} images")
        return min(len(ds), max_real)
        
    except Exception as e:
        print(f"Error: {e}")
        return 0


def download_gendetect_dataset():
    """
    Download a dedicated deepfake detection dataset from HuggingFace.
    """
    print("\n" + "="*70)
    print("DOWNLOADING: GenDetect / Deepfake Detection Dataset")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        # Try multiple dataset options
        datasets_to_try = [
            ("AI-generated/image-prompter-arena", "train"),
            ("shunk0312/image-prompter-arena", "train"),
        ]
        
        for dataset_name, split in datasets_to_try:
            try:
                print(f"Trying: {dataset_name}")
                ds = load_dataset(dataset_name, split=split, streaming=True)
                
                fake_out = FAKE_DIR / dataset_name.replace("/", "_")
                real_out = REAL_DIR / dataset_name.replace("/", "_")
                fake_out.mkdir(parents=True, exist_ok=True)
                real_out.mkdir(parents=True, exist_ok=True)
                
                fake_count = 0
                real_count = 0
                max_per_dataset = 2000
                
                for i, item in enumerate(ds):
                    if fake_count + real_count >= max_per_dataset:
                        break
                    
                    # Try to extract image and label
                    if 'image' in item:
                        img = item['image']
                        # Determine if fake based on available fields
                        label = item.get('label', item.get('is_generated', 1))
                        
                        if label == 1:
                            img.save(fake_out / f"gen_{fake_count:05d}.png")
                            fake_count += 1
                        else:
                            img.save(real_out / f"real_{real_count:05d}.png")
                            real_count += 1
                
                print(f"  Downloaded: {fake_count} fake, {real_count} real")
                
            except Exception as e:
                print(f"  Skipped {dataset_name}: {e}")
                continue
        
    except Exception as e:
        print(f"GenDetect download error: {e}")


def resize_images_to_256(directory: Path, max_count: int = 5000):
    """Resize all images in directory to 256x256."""
    print(f"\nResizing images in {directory} to 256x256...")
    
    count = 0
    for img_path in tqdm(list(directory.glob("*.png")) + list(directory.glob("*.jpg"))):
        if count >= max_count:
            break
        
        try:
            img = Image.open(img_path)
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.LANCZOS)
                img.save(img_path)
            count += 1
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
    
    print(f"  Resized {count} images")


def prepare_final_dataset():
    """Organize all downloaded data into final train-ready format."""
    print("\n" + "="*70)
    print("ORGANIZING FINAL DATASET")
    print("="*70)
    
    # Create organized structure
    final_fake = DATA_DIR / "fake"
    final_real = DATA_DIR / "real"
    
    # Count available images
    fake_count = sum(1 for f in final_fake.rglob("*.png") if f.is_file())
    real_count = sum(1 for f in final_real.rglob("*.png") if f.is_file())
    
    print(f"\nFinal dataset statistics:")
    print(f"  Fake images: {fake_count}")
    print(f"  Real images: {real_count}")
    print(f"  Total:       {fake_count + real_count}")
    print(f"  Location:    {DATA_DIR}")
    
    return fake_count, real_count


def main():
    print("#" * 70)
    print("# DATASET DOWNLOAD AND PREPARATION")
    print("#" * 70)
    
    # Step 1: Download CIFake (main dataset - both real and fake)
    cifake_fake, cifake_real = download_cifake_huggingface(max_samples=8000)
    
    # Step 2: Download additional real images if needed
    if cifake_real < 2000:
        # Try CIFAR-100 as additional real images
        download_alt_real_images(max_real=3000)
    
    # Step 3: Organize and count
    fake_total, real_total = prepare_final_dataset()
    
    print("\n" + "="*70)
    if fake_total > 0 and real_total > 0:
        print("DATASET PREPARATION SUCCESSFUL!")
        print(f"Ready for training with {fake_total} fake + {real_total} real images")
    else:
        print("WARNING: Insufficient data downloaded.")
        print("Creating synthetic fallback data...")
        # Fallback to synthetic data
        os.system("python scripts/create_dummy_data.py --output-dir E:\\deepfake-detection\\data")
    print("="*70)


if __name__ == "__main__":
    main()
