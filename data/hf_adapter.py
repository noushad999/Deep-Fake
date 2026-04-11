"""
Streaming Data Adapter for HuggingFace Datasets.
Allows training directly from HF datasets without full download.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HFStreamingDataset(Dataset):
    """
    Wraps a HuggingFace streaming dataset as a PyTorch Dataset.
    Caches samples in memory as they're streamed.
    """
    
    def __init__(self, dataset_name, split='train', max_samples=None, img_size=224):
        print(f"Loading streaming dataset: {dataset_name} ({split})...")
        
        self.ds = load_dataset(dataset_name, streaming=True, split=split)
        self.img_size = img_size
        self.max_samples = max_samples
        
        # Pre-fetch samples into a list
        self.samples = []
        self._preload_samples()
        
        print(f"  Loaded {len(self.samples)} samples into memory")
        
        # Setup transforms
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _preload_samples(self):
        """Load samples from streaming dataset into memory."""
        for i, sample in enumerate(self.ds):
            if self.max_samples and i >= self.max_samples:
                break
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get image
        img = sample['image']
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = np.array(img)
        
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        
        # Get label
        label = int(sample['label'])
        
        # Apply transform
        augmented = self.transform(image=img_np)
        
        return {
            'image': augmented['image'],
            'label': torch.tensor(label, dtype=torch.float32),
            'path': f"hf_sample_{idx}",
        }


def create_hf_dataloaders(
    dataset_name="ash12321/deepfake-v13-dataset",
    batch_size=32,
    num_workers=0,
    img_size=224,
    train_ratio=0.8,
    max_samples=None
):
    """
    Create train/val/test dataloaders from HF streaming dataset.
    """
    # Load full dataset into memory (streaming one-time)
    print(f"\nStreaming {dataset_name}...")
    
    full_ds = load_dataset(dataset_name, streaming=True)
    
    # Collect all samples
    all_samples = []
    for sample in full_ds['train']:
        all_samples.append(sample)
        if max_samples and len(all_samples) >= max_samples:
            break
        if len(all_samples) % 5000 == 0:
            print(f"  Streamed {len(all_samples)} samples...")
    
    print(f"  Total samples collected: {len(all_samples)}")
    
    # Split
    n = len(all_samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + 0.1))
    
    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:]
    
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")
    
    # Count labels
    for name, samples in [('Train', train_samples), ('Val', val_samples), ('Test', test_samples)]:
        real = sum(1 for s in samples if s['label'] == 0)
        fake = sum(1 for s in samples if s['label'] == 1)
        print(f"  {name}: {real} real, {fake} fake")
    
    # Create datasets
    train_ds = MemoryDataset(train_samples, img_size, is_train=True)
    val_ds = MemoryDataset(val_samples, img_size, is_train=False)
    test_ds = MemoryDataset(test_samples, img_size, is_train=False)
    
    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"\n[HF DataLoaders] Created:")
    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


class MemoryDataset(Dataset):
    """Simple dataset that holds samples in memory."""
    
    def __init__(self, samples, img_size=224, is_train=True):
        self.samples = samples
        self.img_size = img_size
        self.is_train = is_train
        
        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = sample['image']
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = np.array(img)
        
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        
        label = int(sample['label'])
        augmented = self.transform(image=img_np)
        
        return {
            'image': augmented['image'],
            'label': torch.tensor(label, dtype=torch.float32),
            'path': f"hf_{idx}",
        }


if __name__ == "__main__":
    print("="*60)
    print("Testing HF Streaming Dataset Adapter")
    print("="*60)
    
    # Test with small subset first
    train_loader, val_loader, test_loader = create_hf_dataloaders(
        dataset_name="ash12321/deepfake-v13-dataset",
        batch_size=16,
        max_samples=5000  # Just test with 5K samples
    )
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"\nBatch image shape: {batch['image'].shape}")
    print(f"Batch labels: {batch['label'][:10]}")
    print(f"Label distribution: {(batch['label'] > 0.5).sum().item()} fake, {(batch['label'] <= 0.5).sum().item()} real")
