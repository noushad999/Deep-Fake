"""
FaceForensics++ Dataset Loader
Standard CVPR protocol: c23 compression, official train/val/test splits.

Directory structure expected:
    FaceForensics++/
    ├── original_sequences/youtube/c23/frames/{video_id}/*.png
    ├── manipulated_sequences/Deepfakes/c23/frames/{vid}_{tgt}/*.png
    ├── manipulated_sequences/Face2Face/c23/frames/{vid}_{tgt}/*.png
    ├── manipulated_sequences/FaceSwap/c23/frames/{vid}_{tgt}/*.png
    ├── manipulated_sequences/NeuralTextures/c23/frames/{vid}_{tgt}/*.png
    └── splits/train.json, val.json, test.json

Official splits JSON format: [["000", "001"], ["002", "003"], ...]
Each pair = [real_video_id, target_video_id used in fake name].

Reference: Rossler et al., FaceForensics++, ICCV 2019.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class FFPPDataset(Dataset):
    """
    FaceForensics++ dataset following standard evaluation protocol.

    Supports:
    - All 4 manipulation types (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
    - Three compression levels: c0 (raw), c23 (HQ), c40 (LQ)
    - Official train/val/test splits
    - Frame-level and video-level evaluation
    - Configurable frames-per-video sampling
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        compression: str = "c23",
        manipulation_types: Optional[List[str]] = None,
        frames_per_video: int = 32,
        transform=None,
        seed: int = 42,
        return_video_id: bool = False,
    ):
        """
        Args:
            root_dir: Path to FaceForensics++ root directory.
            split: 'train', 'val', or 'test'.
            compression: 'c0', 'c23', or 'c40'.
            manipulation_types: List of manipulation types to include.
                                 Defaults to all 4 types.
            frames_per_video: Number of frames to sample per video.
                              Use -1 to load all frames.
            transform: Albumentations transform pipeline.
            seed: Random seed for frame sampling.
            return_video_id: If True, each sample includes 'video_id' key
                             for video-level aggregation.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.compression = compression
        self.manipulation_types = manipulation_types or MANIPULATION_TYPES
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.seed = seed
        self.return_video_id = return_video_id

        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.video_ids: List[str] = []
        self.manip_types: List[str] = []

        self._load_split()

        real_count = self.labels.count(0)
        fake_count = self.labels.count(1)
        print(
            f"[FFPPDataset] {split:5s} | {compression} | "
            f"Real: {real_count:,} | Fake: {fake_count:,} | "
            f"Total: {len(self.image_paths):,}"
        )

    def _load_split(self):
        split_file = self.root_dir / "splits" / f"{self.split}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Download FF++ official splits from: "
                f"https://github.com/ondyari/FaceForensics"
            )

        with open(split_file) as f:
            pairs = json.load(f)

        rng = np.random.RandomState(self.seed)

        for real_id, target_id in pairs:
            # Load real frames
            real_frames_dir = (
                self.root_dir
                / "original_sequences"
                / "youtube"
                / self.compression
                / "frames"
                / real_id
            )
            self._collect_frames(real_frames_dir, label=0,
                                 video_id=f"real_{real_id}",
                                 manip_type="real", rng=rng)

            # Load fake frames for each manipulation type
            for manip in self.manipulation_types:
                fake_id = f"{real_id}_{target_id}"
                fake_frames_dir = (
                    self.root_dir
                    / "manipulated_sequences"
                    / manip
                    / self.compression
                    / "frames"
                    / fake_id
                )
                self._collect_frames(fake_frames_dir, label=1,
                                     video_id=f"fake_{manip}_{fake_id}",
                                     manip_type=manip, rng=rng)

    def _collect_frames(
        self,
        frames_dir: Path,
        label: int,
        video_id: str,
        manip_type: str,
        rng: np.random.RandomState,
    ):
        if not frames_dir.exists():
            return

        frames = sorted([
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        ])

        if len(frames) == 0:
            return

        # Sample frames_per_video frames uniformly
        if self.frames_per_video > 0 and len(frames) > self.frames_per_video:
            indices = np.linspace(0, len(frames) - 1,
                                  self.frames_per_video, dtype=int)
            frames = [frames[i] for i in indices]

        for frame_path in frames:
            self.image_paths.append(frame_path)
            self.labels.append(label)
            self.video_ids.append(video_id)
            self.manip_types.append(manip_type)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            print(f"Warning: Cannot load {img_path}: {e}")
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            image_tensor = self.transform(image=image_np)["image"]
        else:
            default_tf = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            image_tensor = default_tf(image=image_np)["image"]

        result = {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": str(img_path),
            "manip_type": self.manip_types[idx],
        }

        if self.return_video_id:
            result["video_id"] = self.video_ids[idx]

        return result

    def get_video_ids(self) -> List[str]:
        return list(dict.fromkeys(self.video_ids))  # unique, order-preserved

    def get_manipulation_types(self) -> List[str]:
        return list(dict.fromkeys(self.manip_types))


# ---------------------------------------------------------------------------
# Video-level aggregation utilities
# ---------------------------------------------------------------------------

def aggregate_video_predictions(
    video_ids: List[str],
    scores: np.ndarray,
    labels: np.ndarray,
    strategy: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate frame-level predictions to video-level for AUC computation.

    Args:
        video_ids: Per-frame video IDs.
        scores: Per-frame sigmoid probabilities [N].
        labels: Per-frame binary labels [N].
        strategy: 'mean' (recommended) or 'max'.

    Returns:
        video_scores: Video-level scores [V].
        video_labels: Video-level labels [V].
    """
    unique_ids = list(dict.fromkeys(video_ids))
    video_scores = []
    video_labels = []

    for vid_id in unique_ids:
        mask = np.array(video_ids) == vid_id
        vid_scores = scores[mask]
        vid_label = labels[mask][0]  # all frames have same label

        if strategy == "mean":
            video_scores.append(float(vid_scores.mean()))
        elif strategy == "max":
            video_scores.append(float(vid_scores.max()))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        video_labels.append(int(vid_label))

    return np.array(video_scores), np.array(video_labels)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_ffpp_transforms(phase: str = "train", img_size: int = 224) -> A.Compose:
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if phase == "train":
        return A.Compose([
            A.Resize(img_size + 32, img_size + 32),
            A.RandomCrop(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.4),
            A.ImageCompression(quality_range=(75, 100), p=0.3),
            A.GaussNoise(noise_scale_factor=0.08, p=0.2),
            normalize,
            ToTensorV2(),
        ])
    else:
        return A.Compose([A.Resize(img_size, img_size), normalize, ToTensorV2()])


def create_ffpp_dataloaders(
    root_dir: str,
    compression: str = "c23",
    manipulation_types: Optional[List[str]] = None,
    frames_per_video: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create FF++ train/val/test DataLoaders following standard protocol."""

    train_ds = FFPPDataset(
        root_dir=root_dir, split="train", compression=compression,
        manipulation_types=manipulation_types,
        frames_per_video=frames_per_video,
        transform=get_ffpp_transforms("train", img_size), seed=seed,
    )
    val_ds = FFPPDataset(
        root_dir=root_dir, split="val", compression=compression,
        manipulation_types=manipulation_types,
        frames_per_video=frames_per_video,
        transform=get_ffpp_transforms("val", img_size), seed=seed,
        return_video_id=True,
    )
    test_ds = FFPPDataset(
        root_dir=root_dir, split="test", compression=compression,
        manipulation_types=manipulation_types,
        frames_per_video=-1,  # all frames for test
        transform=get_ffpp_transforms("test", img_size), seed=seed,
        return_video_id=True,
    )

    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,
                              drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
