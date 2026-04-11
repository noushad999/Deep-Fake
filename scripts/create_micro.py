"""
Create micro subset for rapid training completion.
"""
import os, shutil, random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("E:/deepfake-detection/data")
MICRO_DIR = Path("E:/deepfake-detection/data_micro")

random.seed(42)

if MICRO_DIR.exists():
    shutil.rmtree(MICRO_DIR)

fake_dir = MICRO_DIR / "fake"
real_dir = MICRO_DIR / "real"
fake_dir.mkdir(parents=True, exist_ok=True)
real_dir.mkdir(parents=True, exist_ok=True)

all_fake = list(DATA_DIR.glob("fake/**/*.png"))
all_real = list(DATA_DIR.glob("real/**/*.png"))

fake_sample = random.sample(all_fake, min(400, len(all_fake)))
real_sample = random.sample(all_real, min(600, len(all_real)))

print("Copying...")
for f in tqdm(fake_sample + real_sample):
    dst = fake_dir if "fake" in str(f) else real_dir
    shutil.copy(f, dst / f.name)

print(f"Micro subset: {len(fake_sample)} fake + {len(real_sample)} real = {len(fake_sample)+len(real_sample)} total")
