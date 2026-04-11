"""Check Kaggle dataset sizes without downloading."""
import requests
import json

# Kaggle API endpoint for dataset metadata
datasets = [
    {"owner": "vbmokin", "dataset": "real-and-fake-face-detection-size-400x400"},
    {"owner": "shreyanshpatel1", "dataset": "130k-real-vs-fake-face"},
    {"owner": "nanduncs", "dataset": "1000-videos-split"},
    {"owner": "xhlulu", "dataset": "140k-real-and-fake-faces"},
    {"owner": "iplceb", "dataset": "face-forensics"},
]

print("Checking dataset metadata from Kaggle API...\n")

for ds in datasets:
    try:
        # Use kagglehub to get metadata
        import kagglehub
        
        # This will try to download but we'll interrupt early
        print(f"Dataset: {ds['owner']}/{ds['dataset']}")
        
        # Alternative: check via kaggle API info
        result = kagglehub.dataset_download(f"{ds['owner']}/{ds['dataset']}", force_download=False)
        print(f"  Path: {result}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    print()
