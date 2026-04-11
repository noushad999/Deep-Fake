"""Try loading HF deepfake face datasets in streaming mode."""
from datasets import load_dataset
import os

os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_test"

# Try multiple datasets
sources = [
    "ash12321/deepfake-v13-dataset",
    "KubasadNisha/deepfake-detection-dataset-v3",
    "MBZUAI/DeepfakeJudge-Dataset",
]

for name in sources:
    print(f"\n{'='*60}")
    print(f"Trying: {name}")
    print(f"{'='*60}")
    
    try:
        # Try streaming
        ds = load_dataset(name, streaming=True, trust_remote_code=True)
        print(f"  SUCCESS! Splits: {list(ds.keys())}")
        
        # Check first sample
        for split in list(ds.keys())[:1]:
            sample = next(iter(ds[split]))
            print(f"  Keys: {sample.keys()}")
            if 'image' in sample:
                print(f"  Image size: {sample['image'].size}")
            if 'label' in sample:
                print(f"  Label: {sample['label']}")
            
            # Count first 100 samples
            count = 0
            labels = {}
            for item in ds[split]:
                label = item.get('label', 'unknown')
                labels[label] = labels.get(label, 0) + 1
                count += 1
                if count >= 100:
                    break
            
            print(f"  First 100 labels: {labels}")
            print(f"  (Streaming works! Can process incrementally)")
        
    except Exception as e:
        print(f"  FAILED: {e}")
