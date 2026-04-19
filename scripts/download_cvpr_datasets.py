"""
Download CVPR Datasets from HuggingFace
=========================================
Downloads all freely available datasets needed for CVPR experiments:

  1. nebula/FakeCOCO     — 1M+ fake images, 10 generators (MIT license)
  2. saberzl/So-Fake-Set — 1M+ images with masks (social media forgery)
  3. saberzl/So-Fake-OOD — OOD benchmark from real Reddit content

Usage:
    # Download all (recommended, ~50GB total):
    python scripts/download_cvpr_datasets.py --output-dir data/cvpr_datasets

    # Download only what you need:
    python scripts/download_cvpr_datasets.py --datasets fakecoco sofake_ood

    # Download FakeCOCO subset (faster, for quick experiments):
    python scripts/download_cvpr_datasets.py --datasets fakecoco --max-per-gen 5000

After download, run:
    python scripts/train_cvpr.py --data-dir data/cvpr_datasets
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def download_fakecoco(output_dir: Path, max_per_gen: int = None):
    """Download FakeCOCO and save images to disk."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        return

    from data.hf_fakecoco import FAKECOCO_GENERATORS

    print("\n" + "="*60)
    print("Downloading FakeCOCO (nebula/FakeCOCO)")
    print("="*60)

    out = output_dir / "fakecoco"

    print("Loading dataset (streaming mode)...")
    ds = load_dataset("nebula/FakeCOCO", split="train",
                      streaming=True, trust_remote_code=True)

    counts = {gen: 0 for gen in FAKECOCO_GENERATORS}
    counts["real"] = 0
    saved = 0

    for item in ds:
        gen = item.get("generator", "unknown")
        img = item.get("image") or item.get("fake_image")
        label = item.get("label", 1)

        if gen not in counts and label != 0:
            continue

        target_dir = out / (gen if label == 1 else "real")
        target_dir.mkdir(parents=True, exist_ok=True)

        cat = gen if label == 1 else "real"
        if max_per_gen and counts.get(cat, 0) >= max_per_gen:
            if all(v >= max_per_gen for v in counts.values()):
                break
            continue

        if img is not None:
            try:
                from PIL import Image as PILImage
                if not isinstance(img, PILImage.Image):
                    import io
                    img = PILImage.open(io.BytesIO(img))
                img_path = target_dir / f"{counts.get(cat, 0):06d}.jpg"
                img.save(img_path, "JPEG", quality=90)
                counts[cat] = counts.get(cat, 0) + 1
                saved += 1
                if saved % 1000 == 0:
                    print(f"  Saved {saved:,} images...")
            except Exception as e:
                pass

    print(f"\nFakeCOCO download complete:")
    for k, v in counts.items():
        if v > 0:
            print(f"  {k:<20}: {v:,} images")
    print(f"  Output: {out}")


def download_sofake(output_dir: Path, dataset_name: str = "So-Fake-Set",
                    max_samples: int = None):
    """Download So-Fake-Set or So-Fake-OOD."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        return

    hf_id = f"saberzl/{dataset_name}"
    print(f"\n{'='*60}")
    print(f"Downloading {hf_id}")
    print("="*60)

    out = output_dir / dataset_name.lower().replace("-", "_")
    split = "test" if "OOD" in dataset_name else "train"
    ds = load_dataset(hf_id, split=split, streaming=True, trust_remote_code=True)

    counts = {"real": 0, "full_synthetic": 0, "tampered": 0}
    saved = 0

    for item in ds:
        label = item.get("label", "full_synthetic")
        img = item.get("image")
        gen = item.get("generator", label)

        target_dir = out / label
        target_dir.mkdir(parents=True, exist_ok=True)

        if max_samples and sum(counts.values()) >= max_samples:
            break

        if img is not None:
            try:
                from PIL import Image as PILImage
                import io
                if not isinstance(img, PILImage.Image):
                    img = PILImage.open(io.BytesIO(img))
                img_path = target_dir / f"{counts.get(label, 0):06d}.jpg"
                img.save(img_path, "JPEG", quality=90)
                counts[label] = counts.get(label, 0) + 1
                saved += 1
                if saved % 1000 == 0:
                    print(f"  Saved {saved:,} images...")
            except Exception:
                pass

    print(f"\n{dataset_name} download complete:")
    for k, v in counts.items():
        if v > 0:
            print(f"  {k:<20}: {v:,} images")
    print(f"  Output: {out}")


def stream_and_cache_hf(output_dir: Path):
    """
    Alternative: just cache HF datasets locally using save_to_disk.
    Faster than image-by-image saving. Use this for training directly.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        return

    datasets_to_cache = [
        ("nebula/FakeCOCO", "train", "fakecoco_train"),
        ("saberzl/So-Fake-OOD", "test", "sofake_ood"),
    ]

    for hf_id, split, name in datasets_to_cache:
        out_path = output_dir / name
        if out_path.exists():
            print(f"Already cached: {out_path}")
            continue
        print(f"\nCaching {hf_id}/{split} → {out_path}")
        ds = load_dataset(hf_id, split=split, trust_remote_code=True)
        ds.save_to_disk(str(out_path))
        print(f"  Saved {len(ds):,} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download CVPR deepfake datasets from HuggingFace"
    )
    parser.add_argument("--output-dir", default="data/cvpr_datasets")
    parser.add_argument("--datasets", nargs="+",
                        default=["fakecoco", "sofake_set", "sofake_ood"],
                        choices=["fakecoco", "sofake_set", "sofake_ood", "all"])
    parser.add_argument("--max-per-gen", type=int, default=None,
                        help="FakeCOCO: max images per generator")
    parser.add_argument("--max-sofake", type=int, default=None,
                        help="So-Fake: max total samples")
    parser.add_argument("--cache-mode", action="store_true",
                        help="Use HF save_to_disk (faster, keeps HF format)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["fakecoco", "sofake_set", "sofake_ood"]

    if args.cache_mode:
        print("Using HF cache mode (save_to_disk)...")
        stream_and_cache_hf(output_dir)
        return

    if "fakecoco" in datasets:
        download_fakecoco(output_dir, max_per_gen=args.max_per_gen)

    if "sofake_set" in datasets:
        download_sofake(output_dir, "So-Fake-Set", max_samples=args.max_sofake)

    if "sofake_ood" in datasets:
        download_sofake(output_dir, "So-Fake-OOD", max_samples=args.max_sofake)

    print("\n" + "="*60)
    print("Download complete!")
    print(f"Data saved to: {output_dir}")
    print("\nNext step:")
    print(f"  python scripts/train_cvpr.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()
