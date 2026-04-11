"""
Create a tiny training subset for rapid CPU training validation.
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("E:/deepfake-detection/data")
TINY_DIR = Path("E:/deepfake-detection/data_tiny")


def create_tiny_subset(n_fake=800, n_real=1200, seed=42):
    """Create a tiny dataset for rapid CPU training."""
    random.seed(seed)
    
    if TINY_DIR.exists():
        shutil.rmtree(TINY_DIR)
    
    fake_dir = TINY_DIR / "fake"
    real_dir = TINY_DIR / "real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    all_fake = list(DATA_DIR.glob("fake/**/*.png"))
    all_real = list(DATA_DIR.glob("real/**/*.png"))
    
    fake_sample = random.sample(all_fake, min(n_fake, len(all_fake)))
    real_sample = random.sample(all_real, min(n_real, len(all_real)))
    
    print("Copying fake images...")
    for f in tqdm(fake_sample):
        shutil.copy(f, fake_dir / f.name)
    
    print("Copying real images...")
    for f in tqdm(real_sample):
        shutil.copy(f, real_dir / f.name)
    
    total = len(fake_sample) + len(real_sample)
    print(f"\nTiny subset: {total} images ({len(fake_sample)} fake, {len(real_sample)} real)")
    print(f"Location: {TINY_DIR}")
    return total


if __name__ == "__main__":
    create_tiny_subset()
