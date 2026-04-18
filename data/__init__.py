"""
Data pipeline for Multi-Stream Deepfake Detection.
Supports: COCO, FFHQ (real) + DiffusionDB, GenImage, ForenSynths (fake).
"""
from .dataset import DeepfakeDataset, create_dataloaders, get_transforms

__all__ = [
    'DeepfakeDataset',
    'create_dataloaders',
    'get_transforms',
]
