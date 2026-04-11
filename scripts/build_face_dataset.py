"""
DATASET BUILDER: Real Faces + Fake Faces → Training Ready Dataset

Real Source: FFHQ (Flickr-Faces-HQ) — 70K high-quality real faces
Fake Source: StyleGAN/ProGAN generated fake faces

Downloads separately, preprocesses, and merges into:
  data_faces/
    real/
      real_000001.jpg
      real_000002.jpg
      ...
    fake/
      fake_000001.jpg
      fake_000002.jpg
      ...
"""
import os
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

# Configuration
OUTPUT_DIR = Path("/mnt/e/deepfake-detection/data_faces")
REAL_DIR = OUTPUT_DIR / "real"
FAKE_DIR = OUTPUT_DIR / "fake"
IMG_SIZE = 256
MAX_REAL = 5000  # Cap real faces
MAX_FAKE = 5000  # Cap fake faces

random.seed(42)


def download_real_faces():
    """Download REAL face images from FFHQ (HuggingFace)."""
    print("\n" + "=" * 60)
    print("DOWNLOADING REAL FACES — FFHQ (Flickr-Faces-HQ)")
    print("=" * 60)
    print("Source: HF student/FFHQ (70K faces @ 1024x1024)")
    print(f"Target: {MAX_REAL} real faces")
    
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from datasets import load_dataset
        print("Loading FFHQ from HuggingFace (streaming)...")
        ds = load_dataset("student/FFHQ", streaming=True)
        
        count = 0
        skip_count = 0
        
        for item in tqdm(ds['train'], desc="Downloading real faces", total=MAX_REAL):
            if count >= MAX_REAL:
                break
                
            try:
                img = item.get('image')
                if img is None:
                    skip_count += 1
                    continue
                    
                if isinstance(img, Image.Image):
                    # Resize with quality preservation
                    img = img.convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                    img.save(str(REAL_DIR / f"real_{count:06d}.jpg"), quality=95)
                    count += 1
                else:
                    skip_count += 1
                    
            except Exception as e:
                skip_count += 1
                continue
        
        if count > 0:
            print(f"✅ Downloaded {count} REAL faces (skipped {skip_count})")
            print(f"   Saved to: {REAL_DIR}")
            return True
        else:
            print(f"❌ Failed to download real faces (streaming issue?)")
            return False
            
    except Exception as e:
        print(f"❌ FFHQ download failed: {e}")
        return False


def download_fake_faces():
    """Download FAKE face images from StyleGAN (Kaggle)."""
    print("\n" + "=" * 60)
    print("DOWNLOADING FAKE FACES — StyleGAN Generated")
    print("=" * 60)
    print("Source: Kaggle kshitizbhargava/deepfake-face-images")
    print(f"Target: {MAX_FAKE} fake faces")
    
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import kagglehub
        print("Downloading StyleGAN faces from Kaggle...")
        path = kagglehub.dataset_download("kshitizbhargava/deepfake-face-images")
        print(f"  Downloaded to: {path}")
        
        # Explore what's in the downloaded folder
        print(f"  Contents:")
        for item in sorted(Path(path).iterdir()):
            if item.is_dir():
                count = len(list(item.iterdir()))
                print(f"    [DIR]  {item.name} ({count} files)")
            else:
                sz = item.stat().st_size / 1024 / 1024
                print(f"    [FILE] {item.name} ({sz:.1f} MB)")
        
        # Copy and preprocess images
        count = 0
        for root, dirs, files in os.walk(path):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if count >= MAX_FAKE:
                        break
                        
                    src = Path(root) / f
                    try:
                        img = Image.open(src)
                        if img.mode == 'L':
                            img = img.convert('RGB')
                        elif img.mode == 'RGBA':
                            img = img.convert('RGB')
                        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                        img.save(str(FAKE_DIR / f"fake_{count:06d}.jpg"), quality=95)
                        count += 1
                    except Exception as e:
                        continue
            if count >= MAX_FAKE:
                break
        
        if count > 0:
            print(f"✅ Downloaded {count} FAKE faces")
            print(f"   Saved to: {FAKE_DIR}")
            return True
        else:
            print(f"❌ No images found in Kaggle download")
            return False
            
    except Exception as e:
        print(f"❌ StyleGAN download failed: {e}")
        print(f"\nTrying alternative: mayankjha146025/fake-face-images-generated-from-different-gans")
        
        try:
            import kagglehub
            path = kagglehub.dataset_download("mayankjha146025/fake-face-images-generated-from-different-gans")
            print(f"  Downloaded to: {path}")
            
            count = 0
            for root, dirs, files in os.walk(path):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        if count >= MAX_FAKE:
                            break
                        src = Path(root) / f
                        try:
                            img = Image.open(src)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                            img.save(str(FAKE_DIR / f"fake_{count:06d}.jpg"), quality=95)
                            count += 1
                        except:
                            continue
                if count >= MAX_FAKE:
                    break
            
            if count > 0:
                print(f"✅ Downloaded {count} FAKE faces (alternative)")
                print(f"   Saved to: {FAKE_DIR}")
                return True
            else:
                print(f"❌ Alternative also failed")
                return False
                
        except Exception as e2:
            print(f"❌ Alternative also failed: {e2}")
            return False


def verify_dataset():
    """Verify the final dataset."""
    print("\n" + "=" * 60)
    print("VERIFYING DATASET")
    print("=" * 60)
    
    real_images = sorted(list(REAL_DIR.glob("*.jpg")) + list(REAL_DIR.glob("*.png")))
    fake_images = sorted(list(FAKE_DIR.glob("*.jpg")) + list(FAKE_DIR.glob("*.png")))
    
    print(f"\n  Real faces: {len(real_images)}")
    print(f"  Fake faces: {len(fake_images)}")
    print(f"  Total:      {len(real_images) + len(fake_images)}")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("\n❌ DATASET INCOMPLETE!")
        print("   Need both real and fake faces")
        return False
    
    # Check a few samples
    print(f"\n  Sample real images:")
    for img_path in real_images[:3]:
        try:
            img = Image.open(img_path)
            print(f"    {img_path.name}: {img.size}, {img.mode}, {img_path.stat().st_size / 1024:.0f}KB")
        except:
            print(f"    {img_path.name}: CORRUPTED")
    
    print(f"\n  Sample fake images:")
    for img_path in fake_images[:3]:
        try:
            img = Image.open(img_path)
            print(f"    {img_path.name}: {img.size}, {img.mode}, {img_path.stat().st_size / 1024:.0f}KB")
        except:
            print(f"    {img_path.name}: CORRUPTED")
    
    # Check balance
    ratio = len(fake_images) / max(len(real_images), 1)
    if ratio < 0.5 or ratio > 2.0:
        print(f"\n⚠️  Imbalanced dataset! Real:Fake ratio = 1:{ratio:.1f}")
        print("   Ideally should be close to 1:1")
    else:
        print(f"\n✅ Dataset balance OK (Real:Fake = 1:{ratio:.2f})")
    
    total = len(real_images) + len(fake_images)
    if total >= 1000:
        print(f"\n✅ DATASET READY! ({total} images)")
        print(f"\n📁 Location: {OUTPUT_DIR}")
        print(f"\nTo train with this dataset:")
        print(f"  wsl -e bash -c \"source ~/miniconda3/etc/profile.d/conda.sh && conda activate oneclick && \"")
        print(f"  python /mnt/e/deepfake-detection/scripts/train_v2.py \\")
        print(f"    --data-dir /mnt/e/deepfake-detection/data_faces \\")
        print(f"    --epochs 15 --augmentation heavy")
        return True
    else:
        print(f"\n⚠️  Dataset too small ({total} images, need ≥1000)")
        return False


def print_dataset_info():
    """Print info about the datasets we're using."""
    print("\n" + "=" * 60)
    print("DATASET SOURCES")
    print("=" * 60)
    
    print("""
REAL FACES: FFHQ (Flickr-Faces-HQ)
  Source: https://huggingface.co/datasets/student/FFHQ
  Original: 70,000 faces @ 1024×1024
  We'll use: {MAX_REAL} faces @ 256×256
  License: Non-commercial research
  Quality: High (professional photographs)

FAKE FACES: StyleGAN Generated Faces  
  Source: https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images
  Generator: StyleGAN / StyleGAN2
  We'll use: {MAX_FAKE} faces @ 256×256
  Quality: Synthetic (AI-generated faces)

WHY THIS WORKS:
  - Real faces are actual photographs of humans
  - Fake faces are generated by StyleGAN (no real person)
  - Model learns to distinguish real vs synthetic facial features
  - Much better than CIFAKE (which was objects, not faces)
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 FACE DATASET BUILDER")
    print("=" * 60)
    
    print_dataset_info()
    
    print("\nStarting downloads...\n")
    
    real_ok = download_real_faces()
    fake_ok = download_fake_faces()
    
    if real_ok and fake_ok:
        verify_dataset()
    else:
        print("\n⚠️  Some downloads failed. Checking what we have...")
        verify_dataset()
