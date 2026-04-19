"""
FaceForensics++ Frame Extraction Utility
=========================================
Extracts frames from FF++ video files into organized frame directories.
Must be run ONCE before using FFPPDataset.

Requires: pip install opencv-python

Usage:
    # Extract c23 frames (most common, ~15GB):
    python scripts/extract_ffpp_frames.py \\
        --ffpp-dir /path/to/FaceForensics++ \\
        --compression c23 \\
        --every-n 10          # 1 frame per 10 frames (~3fps from 30fps video)

    # Extract all compressions:
    python scripts/extract_ffpp_frames.py \\
        --ffpp-dir /path/to/FaceForensics++ \\
        --compression c23 c40 \\
        --every-n 10

After extraction, directory structure will be:
    FaceForensics++/
    ├── original_sequences/youtube/c23/frames/000/0000.png ...
    └── manipulated_sequences/Deepfakes/c23/frames/000_001/0000.png ...
"""
import argparse
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    every_n: int = 10,
    max_frames: int = 300,
    img_size: int = 0,  # 0 = keep original
):
    """Extract frames from a single video file."""
    if output_dir.exists() and len(list(output_dir.glob("*.png"))) > 0:
        return 0  # already extracted

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  Warning: Cannot open {video_path.name}")
        return 0

    frame_idx = 0
    saved = 0

    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            if img_size > 0:
                frame = cv2.resize(frame, (img_size, img_size))
            out_path = output_dir / f"{saved:04d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def extract_ffpp_split(
    ffpp_root: Path,
    compression: str,
    every_n: int,
    max_frames: int,
    manipulation_types: List[str],
    img_size: int,
):
    """Extract all frames for a given compression level."""
    print(f"\n{'='*60}")
    print(f"Extracting FF++ frames — compression: {compression}")
    print(f"Every {every_n}-th frame, max {max_frames} per video")
    print("=" * 60)

    # Real (original) sequences
    real_video_dir = ffpp_root / "original_sequences" / "youtube" / compression / "videos"
    if not real_video_dir.exists():
        print(f"Warning: Real video dir not found: {real_video_dir}")
    else:
        video_files = sorted(real_video_dir.glob("*.mp4"))
        print(f"\nReal sequences: {len(video_files)} videos")
        total_saved = 0
        for vid in tqdm(video_files, desc="  Real videos"):
            out_dir = ffpp_root / "original_sequences" / "youtube" / compression / "frames" / vid.stem
            n = extract_video_frames(vid, out_dir, every_n, max_frames, img_size)
            total_saved += n
        print(f"  Extracted {total_saved:,} real frames")

    # Manipulated sequences
    for manip in manipulation_types:
        fake_video_dir = (
            ffpp_root / "manipulated_sequences" / manip / compression / "videos"
        )
        if not fake_video_dir.exists():
            print(f"Warning: Fake video dir not found: {fake_video_dir}")
            continue

        video_files = sorted(fake_video_dir.glob("*.mp4"))
        print(f"\n{manip}: {len(video_files)} videos")
        total_saved = 0
        for vid in tqdm(video_files, desc=f"  {manip[:12]}"):
            out_dir = (
                ffpp_root / "manipulated_sequences" / manip / compression / "frames" / vid.stem
            )
            n = extract_video_frames(vid, out_dir, every_n, max_frames, img_size)
            total_saved += n
        print(f"  Extracted {total_saved:,} {manip} frames")


def count_extracted_frames(ffpp_root: Path, compression: str) -> dict:
    """Count already-extracted frames."""
    counts = {}

    real_frames_root = ffpp_root / "original_sequences" / "youtube" / compression / "frames"
    if real_frames_root.exists():
        real_count = sum(1 for p in real_frames_root.rglob("*.png"))
        counts["real"] = real_count

    for manip in MANIPULATION_TYPES:
        frames_root = ffpp_root / "manipulated_sequences" / manip / compression / "frames"
        if frames_root.exists():
            count = sum(1 for p in frames_root.rglob("*.png"))
            counts[manip] = count

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from FaceForensics++ videos"
    )
    parser.add_argument("--ffpp-dir", required=True,
                        help="Path to FaceForensics++ root directory")
    parser.add_argument("--compression", nargs="+", default=["c23"],
                        choices=["c0", "c23", "c40"],
                        help="Compression levels to extract (default: c23)")
    parser.add_argument("--every-n", type=int, default=10,
                        help="Sample every N-th frame (default: 10 → ~3fps from 30fps)")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum frames per video (default: 300)")
    parser.add_argument("--manipulation-types", nargs="+",
                        default=MANIPULATION_TYPES)
    parser.add_argument("--img-size", type=int, default=0,
                        help="Resize to img_size×img_size (0 = keep original)")
    parser.add_argument("--count-only", action="store_true",
                        help="Only count already-extracted frames, don't extract")
    args = parser.parse_args()

    if not _CV2_AVAILABLE:
        print("ERROR: opencv-python not installed.")
        print("Install with: pip install opencv-python")
        sys.exit(1)

    ffpp_root = Path(args.ffpp_dir)
    if not ffpp_root.exists():
        print(f"ERROR: FF++ directory not found: {ffpp_root}")
        sys.exit(1)

    if args.count_only:
        print("Counting extracted frames...")
        for comp in args.compression:
            counts = count_extracted_frames(ffpp_root, comp)
            print(f"\n{comp}:")
            total = 0
            for k, v in counts.items():
                print(f"  {k:<20}: {v:>8,} frames")
                total += v
            print(f"  {'TOTAL':<20}: {total:>8,} frames")
        return

    for compression in args.compression:
        extract_ffpp_split(
            ffpp_root=ffpp_root,
            compression=compression,
            every_n=args.every_n,
            max_frames=args.max_frames,
            manipulation_types=args.manipulation_types,
            img_size=args.img_size,
        )

    print("\n" + "=" * 60)
    print("Extraction complete. Summary:")
    for comp in args.compression:
        counts = count_extracted_frames(ffpp_root, comp)
        total = sum(counts.values())
        print(f"  {comp}: {total:,} total frames")
    print("\nNow run: python scripts/train_ffpp.py --data-dir", args.ffpp_dir)


if __name__ == "__main__":
    main()
