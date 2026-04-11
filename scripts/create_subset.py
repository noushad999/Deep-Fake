"""
Create a smaller training subset for faster CPU training.
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("E:/deepfake-detection/data")
SUBSET_DIR = Path("E:/deepfake-detection/data_subset")


def create_subset(n_fake=2000, n_real=3000, seed=42):
    """Create a smaller dataset for faster CPU training."""
    random.seed(seed)
    
    # Clean old subset
    if SUBSET_DIR.exists():
        shutil.rmtree(SUBSET_DIR)
    
    fake_subset_dir = SUBSET_DIR / "fake"
    real_subset_dir = SUBSET_DIR / "real"
    fake_subset_dir.mkdir(parents=True, exist_ok=True)
    real_subset_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all image paths
    all_fake = list(DATA_DIR.glob("fake/**/*.png"))
    all_real = list(DATA_DIR.glob("real/**/*.png"))
    
    print(f"Available: {len(all_fake)} fake, {len(all_real)} real")
    
    # Sample
    fake_sample = random.sample(all_fake, min(n_fake, len(all_fake)))
    real_sample = random.sample(all_real, min(n_real, len(all_real)))
    
    # Copy
    print(f"Copying {len(fake_sample)} fake images...")
    for f in tqdm(fake_sample):
        shutil.copy(f, fake_subset_dir / f.name)
    
    print(f"Copying {len(real_sample)} real images...")
    for f in tqdm(real_sample):
        shutil.copy(f, real_subset_dir / f.name)
    
    total = len(fake_sample) + len(real_sample)
    print(f"\nSubset created: {total} images ({len(fake_sample)} fake, {len(real_sample)} real)")
    print(f"Location: {SUBSET_DIR}")
    return total


if __name__ == "__main__":
    create_subset()
