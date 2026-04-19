"""
Celeb-DF v2 Dataset Loader — Test Set Only
Standard cross-dataset evaluation benchmark.

Directory structure expected:
    Celeb-DF-v2/
    ├── Celeb-real/         # real celebrity videos (frames/ subdirectory per video)
    ├── Celeb-synthesis/    # deepfake celebrity videos
    ├── YouTube-real/       # real YouTube videos
    └── List_of_testing_videos.txt  # official test list

List_of_testing_videos.txt format:
    1 YouTube-real/id0_0000.mp4
    0 Celeb-synthesis/id0_id1_0000.mp4
    ...

Label: 1 = real, 0 = fake (inverted from our convention, we fix this below).

Reference: Li et al., Celeb-DF: A Large-Scale Challenging Dataset for DeepFake
           Forensics, CVPR 2020.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .ffpp_dataset import aggregate_video_predictions

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class CelebDFDataset(Dataset):
    """
    Celeb-DF v2 dataset for cross-dataset generalization evaluation.

    Protocol:
    - Only the official test partition is used (List_of_testing_videos.txt).
    - Frames are loaded from pre-extracted frame directories.
    - Supports video-level AUC aggregation.
    """

    def __init__(
        self,
        root_dir: str,
        frames_per_video: int = -1,
        transform=None,
        seed: int = 42,
        return_video_id: bool = True,
    ):
        """
        Args:
            root_dir: Path to Celeb-DF-v2 root directory.
            frames_per_video: Frames to sample per video (-1 = all frames).
            transform: Albumentations transform pipeline.
            seed: Random seed for sampling.
            return_video_id: Include video_id key for aggregation.
        """
        self.root_dir = Path(root_dir)
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.seed = seed
        self.return_video_id = return_video_id

        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.video_ids: List[str] = []

        self._load_test_list()

        real_count = self.labels.count(0)
        fake_count = self.labels.count(1)
        print(
            f"[CelebDFv2] Test | Real: {real_count:,} | "
            f"Fake: {fake_count:,} | Total: {len(self.image_paths):,}"
        )

    def _load_test_list(self):
        test_list_path = self.root_dir / "List_of_testing_videos.txt"
        if not test_list_path.exists():
            raise FileNotFoundError(
                f"Test list not found: {test_list_path}\n"
                f"Download from: https://github.com/yuezunli/celeb-deepfakeforensics"
            )

        rng = np.random.RandomState(self.seed)

        with open(test_list_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "label path/to/video.mp4"
                # label: 1=real, 0=fake (CelebDF convention, we invert)
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                celeb_label = int(parts[0])
                video_rel_path = parts[1].strip()

                # CelebDF: 1=real → our label: 0=real; 0=fake → our label: 1=fake
                our_label = 1 - celeb_label

                # Frames directory: replace video filename with frames/
                video_name = Path(video_rel_path).stem
                video_category = Path(video_rel_path).parent.name
                frames_dir = (
                    self.root_dir / video_category / "frames" / video_name
                )

                if not frames_dir.exists():
                    # Try alternate structure: frames next to video
                    frames_dir = (
                        self.root_dir / video_rel_path.replace(".mp4", "")
                    )

                if not frames_dir.exists():
                    continue

                self._collect_frames(
                    frames_dir=frames_dir,
                    label=our_label,
                    video_id=f"{video_category}/{video_name}",
                    rng=rng,
                )

    def _collect_frames(
        self,
        frames_dir: Path,
        label: int,
        video_id: str,
        rng: np.random.RandomState,
    ):
        frames = sorted([
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS
        ])

        if len(frames) == 0:
            return

        if self.frames_per_video > 0 and len(frames) > self.frames_per_video:
            indices = np.linspace(0, len(frames) - 1,
                                  self.frames_per_video, dtype=int)
            frames = [frames[i] for i in indices]

        for frame_path in frames:
            self.image_paths.append(frame_path)
            self.labels.append(label)
            self.video_ids.append(video_id)

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
        }

        if self.return_video_id:
            result["video_id"] = self.video_ids[idx]

        return result


# ---------------------------------------------------------------------------
# Frame extractor utility (requires opencv-python)
# ---------------------------------------------------------------------------

def extract_frames_from_videos(
    root_dir: str,
    output_dir: Optional[str] = None,
    fps: int = 1,
    max_frames: int = 300,
):
    """
    Extract frames from CelebDF-v2 video files.

    This only needs to be run once before using CelebDFDataset.
    Requires: pip install opencv-python

    Args:
        root_dir: CelebDF-v2 root directory.
        output_dir: Output directory (defaults to root_dir).
        fps: Frames per second to extract.
        max_frames: Maximum frames per video.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required: pip install opencv-python")

    root = Path(root_dir)
    out_root = Path(output_dir) if output_dir else root

    categories = ["Celeb-real", "Celeb-synthesis", "YouTube-real"]
    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.exists():
            continue
        for video_path in sorted(cat_dir.glob("*.mp4")):
            frames_dir = out_root / cat / "frames" / video_path.stem
            if frames_dir.exists() and len(list(frames_dir.iterdir())) > 0:
                continue

            frames_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(video_path))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(video_fps / fps))
            frame_idx = 0
            saved = 0

            while cap.isOpened() and saved < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    out_path = frames_dir / f"{saved:04d}.png"
                    cv2.imwrite(str(out_path), frame)
                    saved += 1
                frame_idx += 1

            cap.release()
            if saved > 0:
                print(f"  Extracted {saved} frames: {video_path.name}")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_celebdf_testloader(
    root_dir: str,
    frames_per_video: int = -1,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 42,
) -> DataLoader:
    """Create CelebDF-v2 test DataLoader for cross-dataset evaluation."""
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    ds = CelebDFDataset(
        root_dir=root_dir,
        frames_per_video=frames_per_video,
        transform=transform,
        seed=seed,
        return_video_id=True,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
