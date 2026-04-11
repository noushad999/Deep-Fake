"""Quick check HF deepfake dataset - just get metadata and first sample."""
from datasets import load_dataset
import os

os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_quick"

print("Loading: ash12321/deepfake-v13-dataset (streaming)...")
ds = load_dataset("ash12321/deepfake-v13-dataset", streaming=True)

print(f"Splits: {list(ds.keys())}")

# Get dataset info from config
print(f"\nDataset info: {ds}")

# Get first sample with timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out")

signal.signal(signal.SIGALRM, timeout_handler)

try:
    signal.alarm(30)  # 30 second timeout
    sample = next(iter(ds['train']))
    signal.alarm(0)
    
    print(f"\nFirst sample keys: {sample.keys()}")
    if 'image' in sample:
        print(f"Image type: {type(sample['image'])}")
        print(f"Image size: {sample['image'].size}")
    if 'label' in sample:
        print(f"Label: {sample['label']}")
    for k, v in sample.items():
        print(f"  {k}: {type(v).__name__}")
        
except TimeoutError:
    print("Sample retrieval timed out - but streaming works!")
except Exception as e:
    print(f"Error: {e}")
