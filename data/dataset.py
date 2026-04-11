"""
Robust PyTorch Dataset/DataLoader for deepfake detection.
Handles ForenSynths, CIFake, Stable Diffusion v2.1, and real image datasets.
"""
import os
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Valid image extensions
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


class DeepfakeDataset(Dataset):
    """
    A robust dataset for deepfake detection that handles multiple data sources.
    
    Directory structure expected:
        data_root/
        ├── fake/
        │   ├── forensynths/
        │   ├── cifake/
        │   └── stable_diffusion/
        └── real/
            ├── imagenet/
            └── coco/
    
    Each image is labeled as fake (1) or real (0) based on its parent directory.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        max_samples_per_class: Optional[int] = None,
        domain_labels: bool = False,
        seed: int = 42
    ):
        """
        Args:
            data_root: Root directory containing 'fake/' and 'real/' subdirs.
            split: One of 'train', 'val', 'test'. Controls data splitting.
            transform: Albumentations transform to apply.
            max_samples_per_class: Limit samples per class (for fast prototyping).
            domain_labels: If True, also return domain label (which generator).
            seed: Random seed for reproducibility.
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class
        self.domain_labels = domain_labels
        self.seed = seed
        
        # Collect all image paths and labels
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.domain_names: List[str] = []
        
        self._collect_data()
        self._split_data()
        
        print(f"[DeepfakeDataset] {split} split: {len(self.image_paths)} images "
              f"(Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})")
    
    def _collect_data(self):
        """Recursively collect all image paths with labels."""
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root not found: {self.data_root}\n"
                f"Please run: python scripts/download_datasets.py"
            )
        
        # Collect fake images
        fake_dir = self.data_root / 'fake'
        if fake_dir.exists():
            self._collect_from_directory(fake_dir, label=1)
        
        # Collect real images
        real_dir = self.data_root / 'real'
        if real_dir.exists():
            self._collect_from_directory(real_dir, label=0)
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No images found in {self.data_root}\n"
                f"Expected structure: data_root/fake/... and data_root/real/..."
            )
    
    def _collect_from_directory(self, directory: Path, label: int):
        """Collect images from a directory tree."""
        collected = 0
        
        for root, dirs, files in os.walk(directory):
            for fname in sorted(files):  # sorted for reproducibility
                ext = Path(fname).suffix.lower()
                if ext in IMG_EXTENSIONS:
                    img_path = Path(root) / fname
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    
                    # Extract domain name (e.g., 'forensynths', 'coco')
                    # Domain is the deepest directory name
                    domain = Path(root).name
                    self.domain_names.append(domain)
                    
                    collected += 1
        
        print(f"  Collected {collected} images from {directory} (label={label})")
    
    def _split_data(self):
        """Split data into train/val/test based on seed."""
        rng = np.random.RandomState(self.seed)
        
        # Get indices for each class
        real_indices = [i for i, l in enumerate(self.labels) if l == 0]
        fake_indices = [i for i, l in enumerate(self.labels) if l == 1]
        
        # Split each class separately to maintain balance
        def split_indices(indices):
            rng.shuffle(indices)
            n = len(indices)
            if self.split == 'train':
                return indices[:int(0.8 * n)]
            elif self.split == 'val':
                return indices[int(0.8 * n):int(0.9 * n)]
            else:  # test
                return indices[int(0.9 * n):]
        
        train_real = split_indices(real_indices.copy())
        train_fake = split_indices(fake_indices.copy())
        
        # Re-split for val/test
        rng_val = np.random.RandomState(self.seed + 1)
        val_real = [i for i, l in enumerate(self.labels) if l == 0 
                    and i not in train_real]
        val_fake = [i for i, l in enumerate(self.labels) if l == 1 
                    and i not in train_fake]
        
        rng_val.shuffle(val_real)
        rng_val.shuffle(val_fake)
        
        if self.split == 'val':
            selected_real = val_real[:len(val_real)//2]
            selected_fake = val_fake[:len(val_fake)//2]
        elif self.split == 'test':
            selected_real = val_real[len(val_real)//2:]
            selected_fake = val_fake[len(val_fake)//2:]
        else:  # train
            selected_real = train_real
            selected_fake = train_fake
        
        # Apply max_samples limit
        if self.max_samples_per_class is not None:
            selected_real = selected_real[:self.max_samples_per_class]
            selected_fake = selected_fake[:self.max_samples_per_class]
        
        # Combine and sort for determinism
        selected = sorted(selected_real + selected_fake)
        
        self.image_paths = [self.image_paths[i] for i in selected]
        self.labels = [self.labels[i] for i in selected]
        self.domain_names = [self.domain_names[i] for i in selected]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dict with 'image', 'label', 'path', and optionally 'domain'.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
        except Exception as e:
            # Fallback: return a black image with warning
            print(f"Warning: Could not load {img_path}: {e}")
            image_np = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            # Default transform
            default_transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            augmented = default_transform(image=image_np)
            image_tensor = augmented['image']
        
        result = {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'path': str(img_path)
        }
        
        if self.domain_labels:
            result['domain'] = self.domain_names[idx]
        
        return result
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        n_real = self.labels.count(0)
        n_fake = self.labels.count(1)
        total = n_real + n_fake
        
        weight_real = total / (2.0 * n_real) if n_real > 0 else 1.0
        weight_fake = total / (2.0 * n_fake) if n_fake > 0 else 1.0
        
        return torch.tensor([weight_real, weight_fake])
    
    def get_domain_list(self) -> List[str]:
        """Get unique domain names."""
        return list(set(self.domain_names))


def get_transforms(
    phase: str = 'train',
    img_size: int = 256,
    augmentation_level: str = 'medium'
) -> A.Compose:
    """
    Get data augmentation transforms.
    
    Args:
        phase: 'train', 'val', or 'test'.
        img_size: Target image size.
        augmentation_level: 'none', 'light', 'medium', 'heavy'.
    """
    # Base normalize (ImageNet stats)
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    to_tensor = ToTensorV2()
    
    if phase == 'train':
        if augmentation_level == 'none':
            return A.Compose([
                A.Resize(img_size, img_size),
                normalize,
                to_tensor
            ])
        elif augmentation_level == 'light':
            return A.Compose([
                A.Resize(img_size + 32, img_size + 32),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                normalize,
                to_tensor
            ])
        elif augmentation_level == 'medium':
            return A.Compose([
                A.Resize(img_size + 32, img_size + 32),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.15, contrast=0.15, 
                    saturation=0.15, hue=0.05, p=0.5
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, 
                    rotate_limit=10, p=0.5
                ),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                normalize,
                to_tensor
            ])
        else:  # heavy
            return A.Compose([
                A.Resize(img_size + 64, img_size + 64),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, 
                    saturation=0.2, hue=0.1, p=0.8
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, 
                    rotate_limit=20, p=0.8
                ),
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
                A.Blur(blur_limit=3, p=0.2),
                A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.3),
                normalize,
                to_tensor
            ])
    else:
        # Val/Test: deterministic
        return A.Compose([
            A.Resize(img_size, img_size),
            normalize,
            to_tensor
        ])


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 256,
    max_samples_per_class: Optional[int] = None,
    augmentation_level: str = 'medium',
    use_weighted_sampling: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test DataLoaders.
    
    Args:
        data_root: Path to data directory.
        batch_size: Batch size.
        num_workers: Number of worker processes.
        img_size: Image size for resizing.
        max_samples_per_class: Limit for prototyping.
        augmentation_level: Level of augmentation.
        use_weighted_sampling: Balance classes via sampling.
        seed: Random seed.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_transform = get_transforms('train', img_size, augmentation_level)
    val_transform = get_transforms('val', img_size)
    test_transform = get_transforms('test', img_size)
    
    train_dataset = DeepfakeDataset(
        data_root=data_root,
        split='train',
        transform=train_transform,
        max_samples_per_class=max_samples_per_class,
        seed=seed
    )
    
    val_dataset = DeepfakeDataset(
        data_root=data_root,
        split='val',
        transform=val_transform,
        max_samples_per_class=max_samples_per_class,
        seed=seed
    )
    
    test_dataset = DeepfakeDataset(
        data_root=data_root,
        split='test',
        transform=test_transform,
        max_samples_per_class=max_samples_per_class,
        seed=seed
    )
    
    # Sampler for imbalanced data
    train_sampler = None
    if use_weighted_sampling:
        class_weights = train_dataset.get_class_weights()
        sample_weights = [
            class_weights[int(label)].item() 
            for label in train_dataset.labels
        ]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Common loader kwargs
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True
    }
    
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **{**loader_kwargs, 'drop_last': False}
    )
    
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **{**loader_kwargs, 'drop_last': False}
    )
    
    print(f"\n[DataLoaders] Created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing DeepfakeDataset...")
    
    # Create dummy directory structure
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal structure
        for split in ['fake/forensynths', 'real/coco']:
            os.makedirs(os.path.join(tmpdir, split), exist_ok=True)
        
        # Create dummy images
        for i in range(10):
            img = Image.new('RGB', (256, 256), color=(i*25, i*25, i*25))
            img.save(os.path.join(tmpdir, 'fake/forensynths', f'fake_{i}.png'))
            img.save(os.path.join(tmpdir, 'real/coco', f'real_{i}.png'))
        
        # Test dataset creation
        dataset = DeepfakeDataset(
            data_root=tmpdir,
            split='train',
            max_samples_per_class=5
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Class weights: {dataset.get_class_weights()}")
        
        # Test data loading
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        
        print("\nDeepfakeDataset test PASSED!")
