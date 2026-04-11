"""Download and setup FF++ from HuggingFace Parquet format."""
import os
from pathlib import Path

def download_ffpp_hf():
    """Download FF++ from HuggingFace in Parquet format."""
    from datasets import load_dataset
    
    print("Downloading FF++ from HuggingFace (Parquet format)...")
    print("This will download ~16K frame samples with labels and categories.")
    
    # Load all splits
    ds = load_dataset("TsienDragon/ffplusplus_c23_frames")
    
    print(f"\nDataset loaded!")
    print(f"  Train: {len(ds['train'])} samples")
    print(f"  Test:  {len(ds['test'])} samples")
    print(f"  Val:   {len(ds['validation'])} samples" if 'validation' in ds else "  Val:   Not available")
    
    # Check label distribution
    print("\nTrain split label distribution:")
    train_labels = ds['train'].to_pandas()['label'].value_counts()
    print(train_labels)
    
    print("\nTrain split category distribution:")
    train_cats = ds['train'].to_pandas()['category'].value_counts()
    print(train_cats)
    
    # Save to local directory as images
    output_dir = Path("/mnt/e/deepfake-detection/data_ffpp")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test', 'validation'] if 'validation' in ds else ['train', 'test']:
        split_dir = output_dir / split
        real_dir = split_dir / 'real'
        fake_dir = split_dir / 'fake'
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {split} split...")
        split_data = ds[split]
        
        real_count = 0
        fake_count = 0
        
        for i, item in enumerate(split_data):
            label = item['label']
            category = item.get('category', 'unknown')
            video = item.get('video', 'unknown')
            frame_idx = item.get('frame_idx', i)
            
            img = item['image']  # PIL Image
            
            if label == 'real' or category == 'original':
                filename = f"{video}_frame{frame_idx:04d}.jpg"
                img.save(str(real_dir / filename), quality=95)
                real_count += 1
            else:
                filename = f"{category}_{video}_frame{frame_idx:04d}.jpg"
                img.save(str(fake_dir / filename), quality=95)
                fake_count += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(split_data)} images...")
        
        print(f"  Saved: {real_count} real, {fake_count} fake")
    
    print(f"\nFF++ dataset saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    download_ffpp_hf()
