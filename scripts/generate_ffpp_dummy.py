"""
Generate dummy FaceForensics++ directory structure for testing.
Creates realistic fake PNG frames + official-format split JSONs.
Run this to test all FF++ scripts without real data.

Usage:
    python scripts/generate_ffpp_dummy.py --output-dir data/FaceForensics++_dummy
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image

MANIP_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
COMPRESSIONS = ["c23"]
N_VIDEOS = 20       # small: 720 real in actual FF++
FRAMES_PER_VIDEO = 8


def make_face_frame(seed: int, is_fake: bool, size: int = 224) -> Image.Image:
    rng = np.random.RandomState(seed)
    img = rng.randint(80, 200, (size, size, 3), dtype=np.uint8)
    if is_fake:
        # Add subtle grid artifact (mimics GAN/diffusion spectral artifact)
        for i in range(0, size, 8):
            img[i, :, :] = np.clip(img[i, :, :].astype(int) + 15, 0, 255)
    return Image.fromarray(img)


def create_structure(root: Path, compression: str, n_videos: int, n_frames: int):
    video_ids = [f"{i:03d}" for i in range(n_videos)]

    # Real frames
    for vid in video_ids:
        d = root / "original_sequences" / "youtube" / compression / "frames" / vid
        d.mkdir(parents=True, exist_ok=True)
        for f in range(n_frames):
            make_face_frame(int(vid) * 1000 + f, False).save(d / f"{f:04d}.png")

    # Fake frames
    for manip in MANIP_TYPES:
        for i, vid in enumerate(video_ids):
            target = video_ids[(i + 1) % n_videos]
            pair = f"{vid}_{target}"
            d = root / "manipulated_sequences" / manip / compression / "frames" / pair
            d.mkdir(parents=True, exist_ok=True)
            for f in range(n_frames):
                make_face_frame(hash(manip + pair) % 9999 + f, True).save(d / f"{f:04d}.png")

    # Official split JSONs
    splits_dir = root / "splits"
    splits_dir.mkdir(exist_ok=True)
    pairs = [[video_ids[i], video_ids[(i + 1) % n_videos]] for i in range(n_videos)]
    random.shuffle(pairs)
    n = n_videos
    splits = {
        "train": pairs[:int(0.72 * n)],
        "val":   pairs[int(0.72 * n):int(0.86 * n)],
        "test":  pairs[int(0.86 * n):],
    }
    for name, data in splits.items():
        with open(splits_dir / f"{name}.json", "w") as f:
            json.dump(data, f)
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/FaceForensics++_dummy")
    parser.add_argument("--n-videos", type=int, default=N_VIDEOS)
    parser.add_argument("--frames", type=int, default=FRAMES_PER_VIDEO)
    args = parser.parse_args()

    root = Path(args.output_dir)
    print(f"Generating dummy FF++ at: {root}")
    for comp in COMPRESSIONS:
        splits = create_structure(root, comp, args.n_videos, args.frames)
        total = args.n_videos * args.frames
        fake_total = args.n_videos * args.frames * len(MANIP_TYPES)
        print(f"  [{comp}] Real frames: {total} | Fake frames: {fake_total}")
        for s, p in splits.items():
            print(f"    {s}: {len(p)} video pairs")

    print(f"\nDone. Now test with:")
    print(f"  python scripts/train_ffpp.py --data-dir {root}")


if __name__ == "__main__":
    main()
