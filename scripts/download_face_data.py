"""Download face deepfake datasets for proper training."""
import os
import sys
from pathlib import Path

def download_celeb_df():
    """Try downloading Celeb-DF from Kaggle."""
    try:
        import kagglehub
        print("Attempting Celeb-DF download from Kaggle...")
        path = kagglehub.dataset_download("dansbecker/celeb-df")
        print(f"Celeb-DF downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Celeb-DF download failed: {e}")
        return None

def download_ffpp_frames():
    """Try downloading FF++ extracted frames from Kaggle."""
    try:
        import kagglehub
        print("Attempting FF++ frames download from Kaggle...")
        path = kagglehub.dataset_download("adham7elmy/faceforencispp-extracted-frames")
        print(f"FF++ frames downloaded to: {path}")
        return path
    except Exception as e:
        print(f"FF++ download failed: {e}")
        return None

def check_existing_datasets():
    """Check if we already have any face deepfake data."""
    base = Path("/mnt/e/deepfake-detection")
    
    datasets = {
        "data_quality": base / "data_quality",
        "data": base / "data",
        "data_face": base / "data_face",
        "data_ffpp": base / "data_ffpp",
    }
    
    # Also check common Kaggle/HF download locations
    kaggle_paths = [
        Path("/home/noushad/.cache/kagglehub"),
        Path("/home/noushad/.local/share/kaggle"),
    ]
    
    print("\n=== Checking existing datasets ===")
    for name, path in datasets.items():
        exists = path.exists()
        if exists:
            count = len(list(path.iterdir())) if path.is_dir() else 0
            print(f"  {name}: EXISTS at {path} ({count} items)")
        else:
            print(f"  {name}: NOT FOUND")
    
    print("\n=== Checking Kaggle cache ===")
    for path in kaggle_paths:
        if path.exists():
            print(f"  {path}: EXISTS")
            for item in path.rglob("*"):
                if item.is_dir():
                    print(f"    {item}")
        else:
            print(f"  {path}: NOT FOUND")

def download_hf_datasets():
    """Try downloading from HuggingFace."""
    try:
        from datasets import load_dataset
        print("\nAttempting FF++ from HuggingFace...")
        # Try the HF mirror
        ds = load_dataset("TsienDragon/ffplusplus_c23_frames", split="train", streaming=True)
        print(f"Dataset info: {ds}")
        return True
    except Exception as e:
        print(f"HF download failed: {e}")
        return False

if __name__ == "__main__":
    check_existing_datasets()
    print("\n" + "="*60)
    
    # Try all sources
    result1 = download_ffpp_frames()
    result2 = download_celeb_df()
    result3 = download_hf_datasets()
    
    print("\n" + "="*60)
    print(f"FF++: {'OK' if result1 else 'FAIL'}")
    print(f"Celeb-DF: {'OK' if result2 else 'FAIL'}")
    print(f"HF: {'OK' if result3 else 'FAIL'}")
