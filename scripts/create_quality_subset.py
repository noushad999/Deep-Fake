"""
Create quality training subset from research-grade data.
"""
import os, shutil, random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("E:/deepfake-detection/data")
SUBSET_DIR = Path("E:/deepfake-detection/data_quality")

random.seed(42)

if SUBSET_DIR.exists():
    shutil.rmtree(SUBSET_DIR)

fake_out = SUBSET_DIR / "fake"
real_out = SUBSET_DIR / "real"
fake_out.mkdir(parents=True, exist_ok=True)
real_out.mkdir(parents=True, exist_ok=True)

# Get quality fake images
all_fake = list((DATA_DIR / "fake" / "research_grade").glob("*.png"))
all_real = list((DATA_DIR / "real").rglob("*.png"))

print(f"Available: {len(all_fake)} fake, {len(all_real)} real")

# Sample: 3000 fake, 4000 real
fake_sample = random.sample(all_fake, min(3000, len(all_fake)))
real_sample = random.sample(all_real, min(4000, len(all_real)))

print("Copying fake images...")
for f in tqdm(fake_sample):
    shutil.copy(f, fake_out / f.name)

print("Copying real images...")
for f in tqdm(real_sample):
    shutil.copy(f, real_out / f.name)

total = len(fake_sample) + len(real_sample)
print(f"\nQuality subset: {len(fake_sample)} fake + {len(real_sample)} real = {total} total")
print(f"Location: {SUBSET_DIR}")
