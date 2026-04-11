"""Quick test: can we stream ONE sample from the HF deepfake dataset?"""
from datasets import load_dataset

print("Loading streaming dataset...")
ds = load_dataset("ash12321/deepfake-v13-dataset", streaming=True)
print(f"Splits: {list(ds.keys())}")

print("Fetching first sample...")
sample = next(iter(ds["train"]))
print(f"Keys: {sample.keys()}")
print(f"Image type: {type(sample['image'])}")
if hasattr(sample['image'], 'size'):
    print(f"Image size: {sample['image'].size}")
    print(f"Image mode: {sample['image'].mode}")
print(f"Label: {sample['label']}")

# Fetch 10 more
print("\nFetching 10 more samples...")
labels = [sample['label']]
for i, s in enumerate(ds["train"]):
    if i >= 10:
        break
    labels.append(s['label'])

from collections import Counter
print(f"First 11 labels: {labels}")
print(f"Real (0): {labels.count(0)}, Fake (1): {labels.count(1)}")
print("\n✅ STREAMING WORKS!")
