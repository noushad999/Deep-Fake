"""Quick test: can we actually stream from HF deepfake dataset?"""
from datasets import load_dataset
import time

print("Testing HF deepfake streaming...")
start = time.time()

try:
    print("Loading metadata...")
    ds = load_dataset("ash12321/deepfake-v13-dataset", streaming=True)
    print(f"  OK in {time.time()-start:.1f}s")
    
    print("Getting first sample...")
    sample = next(iter(ds["train"]))
    elapsed = time.time() - start
    print(f"  Got sample in {elapsed:.1f}s")
    
    img = sample.get("image")
    label = sample.get("label")
    print(f"  Image type: {type(img)}")
    print(f"  Image size: {img.size if img else 'None'}")
    print(f"  Label: {label}")
    
    # Try getting 10 more
    print("Getting 10 more samples...")
    labels = [label]
    for i, s in enumerate(ds["train"]):
        if i >= 10:
            break
        labels.append(s.get("label"))
    
    print(f"  Labels: {labels}")
    print(f"  Real(0): {labels.count(0)}, Fake(1): {labels.count(1)}")
    print(f"\n✅ STREAMING WORKS! Total time: {time.time()-start:.1f}s")
    
except Exception as e:
    print(f"\n❌ FAILED after {time.time()-start:.1f}s")
    print(f"Error: {e}")
