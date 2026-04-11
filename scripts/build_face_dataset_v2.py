"""
Download and split: Real faces from one source, Fake faces from another.
Then merge into training-ready dataset structure.
"""
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random

random.seed(42)

OUTPUT_DIR = Path("/mnt/e/deepfake-detection/data_faces")
REAL_DIR = OUTPUT_DIR / "real"
FAKE_DIR = OUTPUT_DIR / "fake"
IMG_SIZE = 256
MAX_REAL = 3000
MAX_FAKE = 3000


def download_from_hf():
    """
    Use HF deepfake dataset that has BOTH real and fake faces.
    Split them into separate folders.
    """
    from datasets import load_dataset
    
    print("Loading HF deepfake dataset (parquet format)...")
    print("Source: ash12321/deepfake-v13-dataset")
    print("This has 60K images: 30K real + 30K fake")
    
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try loading in parquet format (faster than streaming)
    try:
        ds = load_dataset("ash12321/deepfake-v13-dataset")
        print(f"  Loaded! Splits: {list(ds.keys())}")
        
        split = list(ds.keys())[0]
        data = ds[split]
        print(f"  Total samples: {len(data)}")
        
        # Check features
        print(f"  Features: {data.features}")
        
        real_count = 0
        fake_count = 0
        
        for item in tqdm(data, desc="Processing faces", total=len(data)):
            img = item.get('image')
            label = item.get('label')
            
            if img is None:
                continue
            
            if isinstance(img, Image.Image):
                img = img.convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                
                if label == 0:
                    if real_count < MAX_REAL:
                        img.save(str(REAL_DIR / f"real_{real_count:06d}.jpg"), quality=95)
                        real_count += 1
                else:
                    if fake_count < MAX_FAKE:
                        img.save(str(FAKE_DIR / f"fake_{fake_count:06d}.jpg"), quality=95)
                        fake_count += 1
                
                if real_count >= MAX_REAL and fake_count >= MAX_FAKE:
                    break
        
        print(f"\n✅ Real faces: {real_count}")
        print(f"✅ Fake faces: {fake_count}")
        print(f"✅ Total: {real_count + fake_count}")
        return True
        
    except Exception as e:
        print(f"  Parquet download failed: {e}")
        print("  Trying streaming mode...")
        
        # Fallback: streaming
        try:
            ds = load_dataset("ash12321/deepfake-v13-dataset", streaming=True)
            
            real_count = 0
            fake_count = 0
            
            for item in tqdm(ds['train'], desc="Streaming faces", total=MAX_REAL + MAX_FAKE):
                img = item.get('image')
                label = item.get('label')
                
                if img is None:
                    continue
                
                if isinstance(img, Image.Image):
                    img = img.convert('RGB')
                    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                    
                    if label == 0:
                        if real_count < MAX_REAL:
                            img.save(str(REAL_DIR / f"real_{real_count:06d}.jpg"), quality=95)
                            real_count += 1
                    else:
                        if fake_count < MAX_FAKE:
                            img.save(str(FAKE_DIR / f"fake_{fake_count:06d}.jpg"), quality=95)
                            fake_count += 1
                    
                    if real_count >= MAX_REAL and fake_count >= MAX_FAKE:
                        break
            
            print(f"\n✅ Real faces: {real_count}")
            print(f"✅ Fake faces: {fake_count}")
            print(f"✅ Total: {real_count + fake_count}")
            return (real_count + fake_count) > 1000
            
        except Exception as e2:
            print(f"  Streaming also failed: {e2}")
            return False


def verify():
    """Verify the dataset."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    real_count = len(list(REAL_DIR.glob("*.jpg")))
    fake_count = len(list(FAKE_DIR.glob("*.jpg")))
    total = real_count + fake_count
    
    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")
    print(f"  Total: {total}")
    
    if total >= 1000:
        print(f"\n✅ Dataset ready!")
        print(f"  Location: {OUTPUT_DIR}")
        
        # Show samples
        print(f"\n  Real samples:")
        for f in sorted(REAL_DIR.glob("*.jpg"))[:3]:
            img = Image.open(f)
            print(f"    {f.name}: {img.size}, {img.mode}")
        
        print(f"\n  Fake samples:")
        for f in sorted(FAKE_DIR.glob("*.jpg"))[:3]:
            img = Image.open(f)
            print(f"    {f.name}: {img.size}, {img.mode}")
        
        return True
    else:
        print(f"\n❌ Dataset too small: {total}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("FACE DATASET BUILDER v2")
    print("=" * 60)
    print("\nDownloading REAL faces (label=0) → data_faces/real/")
    print("Downloading FAKE faces (label=1) → data_faces/fake/")
    print(f"Target: {MAX_REAL} real + {MAX_FAKE} fake = {MAX_REAL + MAX_FAKE} total")
    
    ok = download_from_hf()
    
    if ok:
        verify()
        print(f"\n{'=' * 60}")
        print("NEXT STEP: TRAIN MODEL")
        print(f"{'=' * 60}")
        print(f"\nwsl -e bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate oneclick && '")
        print(f"python /mnt/e/deepfake-detection/scripts/train_v2.py \\")
        print(f"  --data-dir /mnt/e/deepfake-detection/data_faces \\")
        print(f"  --epochs 15 --augmentation heavy")
    else:
        print("\n❌ Dataset build failed")
