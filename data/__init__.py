"""
Data pipeline module for Multi-Stream Deepfake Detection.
Supports: ForenSynths, CIFake, Stable Diffusion v2.1, ImageNet, MS-COCO.
"""
from .dataset import DeepfakeDataset, create_dataloaders, get_transforms
from .download_datasets import DatasetDownloader

__all__ = [
    'DeepfakeDataset',
    'create_dataloaders',
    'get_transforms',
    'DatasetDownloader'
]
