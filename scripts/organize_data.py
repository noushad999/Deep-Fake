"""Organize and verify face dataset."""
from pathlib import Path
import shutil

base = Path("/mnt/e/deepfake-detection/data_faces")
src_fake = base / "Final Dataset" / "Fake"
dst_fake = base / "fake"
dst_real = base / "real"

# Copy fake faces if source exists
if src_fake.exists():
    print("Copying fake faces...")
    for f in src_fake.glob("*.jpg"):
        shutil.copy2(f, dst_fake / f.name)
    print(f"Copied {len(list(src_fake.glob('*.jpg')))} fake faces")

# Clean up
final_ds = base / "Final Dataset"
if final_ds.exists():
    shutil.rmtree(final_ds)
    print("Cleaned up Final Dataset folder")

# Count
real = list(dst_real.glob("*.jpg"))
fake = list(dst_fake.glob("*.jpg"))
print(f"\n{'='*50}")
print(f"FINAL DATASET")
print(f"{'='*50}")
print(f"Real faces: {len(real)}")
print(f"Fake faces: {len(fake)}")
print(f"Total:      {len(real) + len(fake)}")

if real:
    print(f"\nSample real: {real[0].name}, {real[0].stat().st_size/1024:.0f}KB")
if fake:
    print(f"Sample fake: {fake[0].name}, {fake[0].stat().st_size/1024:.0f}KB")

total = len(real) + len(fake)
if total >= 1000:
    print(f"\n✅ DATASET READY! ({total} images)")
    print(f"\nTraining command:")
    print(f"  python train_v2.py --data-dir {base} --epochs 15 --augmentation heavy")
else:
    print(f"\n❌ Too small: {total}")
