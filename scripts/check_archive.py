"""Check Kaggle archive and extract if ready."""
from pathlib import Path
import subprocess
import tarfile
import zipfile

archive = Path("/home/noushad/.cache/kagglehub/datasets/adham7elmy/faceforencispp-extracted-frames/4.archive")

print(f"Archive exists: {archive.exists()}")
if archive.exists():
    print(f"Size: {archive.stat().st_size / 1e9:.1f} GB")

# Try to detect file type
output_dir = Path("/mnt/e/deepfake-detection/data_ffpp_kaggle")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExtracting to: {output_dir}")

# Try tar
try:
    print("Trying tar extraction...")
    with tarfile.open(archive, 'r:*') as tar:
        members = tar.getmembers()
        print(f"Found {len(members)} files in archive")
        for m in members[:10]:
            print(f"  {m.name}")
        tar.extractall(path=output_dir)
        print("Extraction complete!")
except Exception as e:
    print(f"tar failed: {e}")

# Try zip
try:
    print("Trying zip extraction...")
    with zipfile.ZipFile(archive, 'r') as zf:
        names = zf.namelist()
        print(f"Found {len(names)} files in archive")
        for n in names[:10]:
            print(f"  {n}")
        zf.extractall(path=output_dir)
        print("Extraction complete!")
except Exception as e:
    print(f"zip failed: {e}")

# Check extracted
if output_dir.exists():
    print(f"\nExtracted contents:")
    for item in sorted(output_dir.iterdir())[:20]:
        if item.is_dir():
            count = len(list(item.iterdir()))
            print(f"  [DIR]  {item.name} ({count} items)")
        else:
            print(f"  [FILE] {item.name} ({item.stat().st_size / 1e6:.1f} MB)")
