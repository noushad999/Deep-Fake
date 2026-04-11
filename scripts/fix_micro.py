"""
Fix: Create micro subset with proper fake/real split.
"""
import os, shutil, random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("E:/deepfake-detection/data")
MICRO_DIR = Path("E:/deepfake-detection/data_micro")

random.seed(42)

if MICRO_DIR.exists():
    shutil.rmtree(MICRO_DIR)

fake_out = MICRO_DIR / "fake"
real_out = MICRO_DIR / "real"
fake_out.mkdir(parents=True, exist_ok=True)
real_out.mkdir(parents=True, exist_ok=True)

# Get images from both directories
all_fake = list((DATA_DIR / "fake").rglob("*.png"))
all_real = list((DATA_DIR / "real").rglob("*.png"))

print(f"Found: {len(all_fake)} fake, {len(all_real)} real")

fake_sample = random.sample(all_fake, min(400, len(all_fake)))
real_sample = random.sample(all_real, min(600, len(all_real)))

print("Copying fake images...")
for f in tqdm(fake_sample):
    shutil.copy(f, fake_out / f.name)

print("Copying real images...")
for f in tqdm(real_sample):
    shutil.copy(f, real_out / f.name)

print(f"\nMicro subset: {len(fake_sample)} fake + {len(real_sample)} real = {len(fake_sample)+len(real_sample)} total")
print(f"Location: {MICRO_DIR}")

# Verify
vf = sum(1 for _ in fake_out.glob("*.png"))
vr = sum(1 for _ in real_out.glob("*.png"))
print(f"Verification: {vf} fake, {vr} real in output")
