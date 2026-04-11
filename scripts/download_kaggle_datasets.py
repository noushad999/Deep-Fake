"""
Download from Kaggle deepfake detection datasets.
Uses kagglehub to download high-quality curated datasets.
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import zipfile

DATA_DIR = Path("E:/deepfake-detection/data")
FAKE_DIR = DATA_DIR / "fake"
REAL_DIR = DATA_DIR / "real"


def download_kaggle_dataset(owner_slug, dataset_slug, output_dir, max_files=10000):
    """
    Download a Kaggle dataset using kagglehub.
    """
    print(f"\n{'='*70}")
    print(f"DOWNLOADING FROM KAGGLE: {owner_slug}/{dataset_slug}")
    print(f"{'='*70}")
    
    try:
        import kagglehub
        
        print("Downloading from Kaggle...")
        path = kagglehub.dataset_download(f"{owner_slug}/{dataset_slug}")
        print(f"Downloaded to: {path}")
        
        # Process the dataset
        fake_out = FAKE_DIR / dataset_slug
        real_out = REAL_DIR / dataset_slug
        fake_out.mkdir(parents=True, exist_ok=True)
        real_out.mkdir(parents=True, exist_ok=True)
        
        # Find image files
        img_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_files = []
        
        for root, dirs, files in os.walk(path):
            for f in files:
                if Path(f).suffix.lower() in img_extensions:
                    image_files.append(Path(root) / f)
        
        print(f"Found {len(image_files)} images")
        
        fake_count = 0
        real_count = 0
        
        # Try to infer labels from directory structure
        for img_path in tqdm(image_files[:max_files], desc="Processing"):
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Resize to 256x256
                if img.size != (256, 256):
                    img = img.resize((256, 256), Image.LANCZOS)
                
                # Infer label from path
                path_lower = str(img_path).lower()
                if any(x in path_lower for x in ['fake', 'forged', 'manipulated', 'generated', 'ai']):
                    img.save(fake_out / f"fake_{fake_count:05d}.png")
                    fake_count += 1
                elif any(x in path_lower for x in ['real', 'genuine', 'authentic', 'original', 'bonafide']):
                    img.save(real_out / f"real_{real_count:05d}.png")
                    real_count += 1
                else:
                    # Default: put in real (conservative)
                    img.save(real_out / f"real_{real_count:05d}.png")
                    real_count += 1
                    
            except Exception as e:
                continue
        
        print(f"\nCOMPLETE: {fake_count} fake, {real_count} real")
        return fake_count, real_count
        
    except ImportError:
        print("kagglehub not installed. Install: pip install kagglehub")
        return 0, 0
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0


def main():
    print("#"*70)
    print("# KAGGLE DEEPFAKE DATASET DOWNLOADER")
    print("#"*70)
    
    # Clean old synthetic
    for d in ['fake/synthetic_gan', 'real/synthetic_natural', 
              'fake/dummy', 'real/dummy']:
        path = DATA_DIR / d
        if path.exists():
            shutil.rmtree(path)
    
    # Try Kaggle datasets
    datasets_to_try = [
        ("prithivsakthiur", "deepfake-vs-real-20k"),
        ("prithivsakthiur", "deepfake-vs-real-8k"),
        ("uditsharma72", "real-vs-fake-faces"),
    ]
    
    total_fake = 0
    total_real = 0
    
    for owner, slug in datasets_to_try:
        if total_fake >= 5000 and total_real >= 5000:
            break
        
        f, r = download_kaggle_dataset(owner, slug, DATA_DIR, max_files=8000)
        total_fake += f
        total_real += r
    
    # Add CIFAR-10 if needed
    if total_real < 5000:
        print("\nAdding CIFAR-10 real images...")
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
        except Exception as e:
            print(f"CIFAR-10 error: {e}")
    
    # Final stats
    actual_fake = sum(1 for f in FAKE_DIR.rglob("*.png"))
    actual_real = sum(1 for f in REAL_DIR.rglob("*.png"))
    
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"  Fake:  {actual_fake}")
    print(f"  Real:  {actual_real}")
    print(f"  Total: {actual_fake + actual_real}")
    print("="*70)


if __name__ == "__main__":
    main()
