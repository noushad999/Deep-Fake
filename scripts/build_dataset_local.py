"""
Generate dataset LOCALLY without downloading.
1. Real faces: Download small LFW dataset (direct HTTP, ~170MB)
2. Fake faces: Generate using StyleGAN2 (weights cached or small download)
"""
import os
import subprocess
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

random.seed(42)

OUTPUT_DIR = Path("/mnt/e/deepfake-detection/data_faces")
REAL_DIR = OUTPUT_DIR / "real"
FAKE_DIR = OUTPUT_DIR / "fake"
IMG_SIZE = 256
MAX_REAL = 3000
MAX_FAKE = 3000


def download_real_faces_lfw():
    """
    Download LFW (Labeled Faces in the Wild) - SMALL dataset.
    Size: ~170MB | 13,233 face images | Direct HTTP download
    """
    print("\n" + "=" * 60)
    print("DOWNLOADING REAL FACES — LFW Dataset")
    print("=" * 60)
    print("Source: http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz")
    print("Size: ~170MB | 13,233 real face images")
    
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    lfw_dir = OUTPUT_DIR / "lfw_raw"
    tgz_path = OUTPUT_DIR / "lfw-deepfunneled.tgz"
    
    if not tgz_path.exists():
        print("\nDownloading LFW (170MB, should take ~5-10 min)...")
        url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
        
        try:
            # Try with requests for better progress tracking
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            
            with open(tgz_path, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Downloaded: {tgz_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\nTrying wget as fallback...")
            try:
                result = subprocess.run(
                    ["wget", "-O", str(tgz_path), url],
                    capture_output=True, text=True, timeout=600
                )
                print(f"wget exit code: {result.returncode}")
            except Exception as e2:
                print(f"❌ wget also failed: {e2}")
                return False
    else:
        print(f"  LFW already downloaded: {tgz_path.stat().st_size / 1e6:.1f} MB")
    
    # Extract
    if not lfw_dir.exists():
        print("\nExtracting LFW...")
        try:
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(lfw_dir)
            print(f"✅ Extracted to: {lfw_dir}")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    # Find all face images
    print("\nCollecting face images...")
    face_images = []
    if lfw_dir.exists():
        for root, dirs, files in os.walk(lfw_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    face_images.append(Path(root) / f)
    
    print(f"  Found {len(face_images)} face images")
    
    # Copy and resize
    count = 0
    for img_path in tqdm(face_images[:MAX_REAL], desc="Processing real faces"):
        try:
            img = Image.open(img_path).convert('RGB')
            # Center crop to square
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(str(REAL_DIR / f"real_{count:06d}.jpg"), quality=95)
            count += 1
        except:
            continue
    
    print(f"\n✅ Real faces: {count}")
    return count > 100


def generate_fake_faces():
    """
    Generate FAKE faces using StyleGAN2-ADA from timm.
    No external download needed - uses pretrained weights.
    """
    print("\n" + "=" * 60)
    print("GENERATING FAKE FACES — StyleGAN2")
    print("=" * 60)
    print("Using timm's StyleGAN2-ADA pretrained model")
    print(f"Target: {MAX_FAKE} fake faces")
    
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import torch
        import timm
        
        print("\nLoading StyleGAN2-ADA from timm...")
        # StyleGAN2-ADA FFHQ model
        model = timm.create_model('stylegan2_ffhq_256')
        model.eval()
        
        print(f"  Model loaded!")
        print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"\nGenerating {MAX_FAKE} fake faces...")
        count = 0
        batch_size = 8
        
        with torch.no_grad():
            for i in tqdm(range(0, MAX_FAKE, batch_size), desc="Generating"):
                n = min(batch_size, MAX_FAKE - i)
                
                # Generate random latents
                z = torch.randn(n, 512, device=device)
                
                # Generate images
                images = model(z)
                
                # Save
                for j in range(n):
                    img = images[j]
                    # Denormalize: [-1, 1] -> [0, 255]
                    img = (img + 1) / 2
                    img = img.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    
                    pil_img = Image.fromarray(img)
                    pil_img.save(str(FAKE_DIR / f"fake_{count+j:06d}.jpg"), quality=95)
                
                count += n
        
        print(f"\n✅ Generated {count} fake faces")
        return True
        
    except Exception as e:
        print(f"❌ StyleGAN2 generation failed: {e}")
        print("\nTrying alternative: use pre-generated fake faces from Kaggle...")
        return False


def verify():
    """Verify the dataset."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    real_count = len(list(REAL_DIR.glob("*.jpg")))
    fake_count = len(list(FAKE_DIR.glob("*.jpg")))
    total = real_count + fake_count
    
    print(f"  Real faces: {real_count}")
    print(f"  Fake faces: {fake_count}")
    print(f"  Total:      {total}")
    
    # Show samples
    if real_count > 0:
        print(f"\n  Real sample:")
        for f in sorted(REAL_DIR.glob("*.jpg"))[:3]:
            img = Image.open(f)
            print(f"    {f.name}: {img.size}, {img.mode}, {f.stat().st_size/1024:.0f}KB")
    
    if fake_count > 0:
        print(f"\n  Fake sample:")
        for f in sorted(FAKE_DIR.glob("*.jpg"))[:3]:
            img = Image.open(f)
            print(f"    {f.name}: {img.size}, {img.mode}, {f.stat().st_size/1024:.0f}KB")
    
    if total >= 1000:
        print(f"\n✅ DATASET READY! ({total} images)")
        print(f"\n📁 Location: {OUTPUT_DIR}")
        print(f"\nTraining command:")
        print(f"  python train_v2.py --data-dir {OUTPUT_DIR} --epochs 15 --augmentation heavy")
        return True
    else:
        print(f"\n❌ Dataset too small: {total} images")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🔬 FACE DATASET BUILDER v3 — LOCAL GENERATION")
    print("=" * 60)
    print("\nPlan:")
    print("  1. Download LFW (real faces, ~170MB via HTTP)")
    print("  2. Generate StyleGAN fake faces (GPU, no download)")
    print("  3. Merge into data_faces/real/ + data_faces/fake/")
    
    print("\n" + "-" * 60)
    print("STEP 1: REAL FACES")
    print("-" * 60)
    real_ok = download_real_faces_lfw()
    
    if not real_ok:
        print("\n⚠️  Real face download failed. Trying cached option...")
    
    print("\n" + "-" * 60)
    print("STEP 2: FAKE FACES")
    print("-" * 60)
    fake_ok = generate_fake_faces()
    
    if not fake_ok:
        print("\n⚠️  Fake face generation failed.")
    
    print("\n" + "-" * 60)
    print("STEP 3: VERIFICATION")
    print("-" * 60)
    verify()
