"""Find and check sizes of small face deepfake datasets on Kaggle."""
import kagglehub
import os

datasets = [
    "vbmokin/real-and-fake-face-detection-size-400x400",
    "shreyanshpatel1/130k-real-vs-fake-face",
    "nanduncs/1000-videos-split",
    "iplceb/face-forensics",
    "janged/face-forensics",
    "xhlulu/140k-real-and-fake-faces",
]

for name in datasets:
    try:
        print(f"\n{'='*60}")
        print(f"Checking: {name}")
        print(f"{'='*60}")
        path = kagglehub.dataset_download(name, force_download=False)
        print(f"  Path: {path}")
        
        # Check size
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        
        size_mb = total_size / (1024 * 1024)
        size_gb = total_size / (1024 * 1024 * 1024)
        
        if size_gb >= 1:
            print(f"  SIZE: {size_gb:.2f} GB")
        else:
            print(f"  SIZE: {size_mb:.1f} MB")
        print(f"  FILES: {file_count}")
        
        # List first few files
        for root, dirs, files in os.walk(path):
            for f in sorted(files)[:10]:
                fp = os.path.join(root, f)
                sz = os.path.getsize(fp) / (1024*1024)
                print(f"    {f} ({sz:.1f} MB)")
            if dirs:
                for d in sorted(dirs)[:10]:
                    print(f"    [DIR] {d}")
            break
            
    except Exception as e:
        print(f"  FAILED: {e}")
