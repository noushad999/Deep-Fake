"""Stream FF++ from HuggingFace without full download."""
from datasets import load_dataset
import os
from pathlib import Path

# Set HF cache to a reasonable location
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache_ffpp"

print("Loading FF++ in streaming mode...")
ds = load_dataset("TsienDragon/ffplusplus_c23_frames", streaming=True)

print("\nDataset info:")
print(f"  Splits available: {list(ds.keys())}")

# Check a few samples
print("\nSample from train split:")
for i, item in enumerate(ds['train'].take(5)):
    print(f"  [{i}] label={item['label']}, category={item['category']}, video={item['video']}, frame={item['frame_idx']}, image_size={item['image'].size}")

# Count total samples by iterating
print("\nCounting samples (this will stream all data)...")
counts = {'train': {'real': 0, 'fake': 0}, 'test': {'real': 0, 'fake': 0}}
total = {'train': 0, 'test': 0}

for split in ['train', 'test']:
    for item in ds[split]:
        label = item['label']
        if label == 'real':
            counts[split]['real'] += 1
        else:
            counts[split]['fake'] += 1
        total[split] += 1
        
        if total[split] % 1000 == 0:
            print(f"  {split}: counted {total[split]} samples...")

print(f"\nFinal counts:")
for split in ['train', 'test']:
    print(f"  {split}: {total[split]} total ({counts[split]['real']} real, {counts[split]['fake']} fake)")
    real_pct = 100 * counts[split]['real'] / max(total[split], 1)
    print(f"    Real: {real_pct:.1f}%")
