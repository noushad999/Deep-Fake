"""Download small face face datasets from Kaggle."""
import kagglehub
import os
from pathlib import Path

# Small datasets to try
datasets = [
    "vbmokin/real-and-fake-face-detection-size-400x400",
]

for name in datasets:
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"{'='*60}")
    try:
        path = kagglehub.dataset_download(name)
        print(f"Downloaded to: {path}")
        
        # Check contents
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB ({file_count} files)")
        
        # List directory structure
        for item in sorted(Path(path).iterdir())[:20]:
            if item.is_dir():
                count = len(list(item.iterdir()))
                print(f"  [DIR] {item.name} ({count} items)")
            else:
                print(f"  [FILE] {item.name}")
                
    except Exception as e:
        print(f"FAILED: {e}")
