"""Extract whatever we can from partial Kaggle downloads."""
import os
import tarfile
import zipfile
from pathlib import Path

def try_extract_ffpp():
    """Try to extract the partial FF++ archive."""
    archive = Path("/home/noushad/.cache/kagglehub/datasets/adham7elmy/faceforencispp-extracted-frames/4.archive")
    output = Path("/mnt/e/deepfake-detection/data_ffpp_partial")
    output.mkdir(parents=True, exist_ok=True)
    
    print(f"FF++ archive: {archive.stat().st_size / 1e9:.2f} GB")
    
    # Try tar
    for mode in ['r', 'r:gz', 'r:bz2', 'r:xz', 'r:*']:
        try:
            print(f"\nTrying mode: {mode}")
            with tarfile.open(archive, mode) as tar:
                members = tar.getmembers()
                print(f"  Found {len(members)} entries")
                for m in members[:5]:
                    print(f"    {m.name} ({m.size / 1e6:.1f} MB)")
                
                # Extract all we can
                print("  Extracting...")
                tar.extractall(output)
                print(f"  ✅ Extracted to: {output}")
                return True
        except Exception as e:
            print(f"  Failed: {e}")
    
    # Try as zip
    try:
        print("\nTrying zip...")
        with zipfile.ZipFile(archive, 'r') as zf:
            names = zf.namelist()
            print(f"  Found {len(names)} files")
            for n in names[:5]:
                print(f"    {n}")
            zf.extractall(output)
            print(f"  ✅ Extracted to: {output}")
            return True
    except Exception as e:
        print(f"  Failed: {e}")
    
    # Try as raw image file
    print("\nFile doesn't seem to be a standard archive.")
    print("Checking file signature...")
    with open(archive, 'rb') as f:
        magic = f.read(16)
        print(f"  Magic bytes: {magic.hex()}")
        print(f"  ASCII: {magic}")
    
    return False


def try_extract_stylegan():
    """Try to extract the partial StyleGAN fake faces archive."""
    archive = Path("/home/noushad/.cache/kagglehub/datasets/kshitizbhargava/deepfake-face-images/1.archive")
    output = Path("/mnt/e/deepfake-detection/data_stylegan_partial")
    output.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStyleGAN archive: {archive.stat().st_size / 1e6:.1f} MB")
    
    for mode in ['r', 'r:gz', 'r:bz2', 'r:xz', 'r:*']:
        try:
            print(f"\nTrying mode: {mode}")
            with tarfile.open(archive, mode) as tar:
                members = tar.getmembers()
                print(f"  Found {len(members)} entries")
                for m in members[:10]:
                    print(f"    {m.name} ({m.size / 1024:.0f} KB)")
                
                print("  Extracting...")
                tar.extractall(output)
                print(f"  ✅ Extracted to: {output}")
                
                # Count images
                img_count = len(list(output.rglob("*.jpg")) + list(output.rglob("*.png")))
                print(f"  Found {img_count} images")
                return True
        except Exception as e:
            print(f"  Failed: {e}")
    
    try:
        print("\nTrying zip...")
        with zipfile.ZipFile(archive, 'r') as zf:
            names = zf.namelist()
            print(f"  Found {len(names)} files")
            zf.extractall(output)
            img_count = len(list(output.rglob("*.jpg")) + list(output.rglob("*.png")))
            print(f"  Found {img_count} images")
            return True
    except Exception as e:
        print(f"  Failed: {e}")
    
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("EXTRACTING PARTIAL KAGGLE DOWNLOADS")
    print("=" * 60)
    
    print("\n--- FF++ Archive ---")
    try_extract_ffpp()
    
    print("\n--- StyleGAN Archive ---")
    try_extract_stylegan()
    
    # Check results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for d in [Path("/mnt/e/deepfake-detection/data_ffpp_partial"), 
              Path("/mnt/e/deepfake-detection/data_stylegan_partial")]:
        if d.exists():
            img_count = len(list(d.rglob("*.jpg")) + list(d.rglob("*.png")))
            total_size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            print(f"{d.name}: {img_count} images, {total_size/1e6:.1f} MB")
        else:
            print(f"{d.name}: Not extracted")
