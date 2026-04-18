"""
Dataset Download Script — Faces-Only for Publishable Deepfake Detection
========================================================================

REAL face images:
  1. FFHQ 256px           — up to 20,000 high-quality face photos
     Source: "merkol/ffhq256" on HuggingFace
  2. CelebA-HQ 256px      — celebrity face dataset
     Source: "datasets/celebahq-faces" on HuggingFace (fallback: LFW)

FAKE / AI-generated face images:
  3. DiffusionDB faces    — Stable Diffusion v1.x face prompts
     Source: "poloclub/diffusiondb" (filtered for face content)
  4. GenImage faces       — SD-XL, Midjourney v5, DALL-E 3, ADM face subset
     Source: "Wuvin/GenImage" (face-class images only)
  5. ThisPersonDoesNotExist / PGGAN faces — GAN-era face fakes
     (ForenSynths ProGAN subset — manual download, instructions printed)

Why faces-only?
  Deepfake detection papers target face manipulation. Mixed-content datasets
  (food, animals, scenery) introduce content-type bias: the model learns
  "food photo = real" rather than forgery cues. Reviewers will reject such
  a dataset. All real and fake images must be faces.

Usage:
  python scripts/download_datasets.py --phase 1   # FFHQ + DiffusionDB (~4 GB)
  python scripts/download_datasets.py --phase 2   # Full faces (~15 GB)
  python scripts/download_datasets.py --all        # Everything
  python scripts/download_datasets.py --clean      # Remove non-face images from existing data
"""

import os
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

NON_FACE_DIRS = ["coco", "food101", "naturescenes", "animals", "objects"]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _progress(count, block_size, total_size):
    pct = count * block_size * 100 // max(total_size, 1)
    bar = '#' * (pct // 2) + '-' * (50 - pct // 2)
    print(f"\r  [{bar}] {pct:3d}%", end='', flush=True)


def _download_url(url: str, dest: Path, description: str = ""):
    print(f"  Downloading {description or dest.name} …")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


# -----------------------------------------------------------------------
# Clean non-face images from existing data
# -----------------------------------------------------------------------

def clean_non_face_data(data_dir: Path = DATA_DIR):
    """Remove non-face directories (food101, coco, naturescenes, animals)."""
    print("\n=== Cleaning Non-Face Data ===")
    removed = 0
    for label in ["real", "fake"]:
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        for non_face in NON_FACE_DIRS:
            target = label_dir / non_face
            if target.exists():
                n = len(list(target.rglob("*.jpg"))) + len(list(target.rglob("*.png")))
                shutil.rmtree(target)
                print(f"  Removed {label}/{non_face}/ ({n} images)")
                removed += n
    if removed == 0:
        print("  Nothing to clean — no non-face directories found.")
    else:
        print(f"  Total removed: {removed} non-face images")


# -----------------------------------------------------------------------
# 1. FFHQ 256px (real faces)
# -----------------------------------------------------------------------

def download_ffhq(output_dir: Path, max_images: int = 10000):
    """FFHQ 256px — high-quality human face photos from NVlabs."""
    print(f"\n=== FFHQ 256px (real faces, up to {max_images}) ===")

    real_dir = output_dir / "real" / "ffhq"
    real_dir.mkdir(parents=True, exist_ok=True)

    existing = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    if len(existing) >= max_images:
        print(f"  Already have {len(existing)} FFHQ images — skipping.")
        return

    try:
        from datasets import load_dataset
        print(f"  Streaming FFHQ 256px (need {max_images - len(existing)} more) …")
        ds = load_dataset("merkol/ffhq256", split="train", streaming=True)

        saved = len(existing)
        start_idx = saved
        for i, sample in enumerate(ds):
            if saved >= max_images:
                break
            if i < start_idx:
                continue
            img = sample.get("image") or sample.get("img")
            if img is None:
                continue
            img.save(real_dir / f"ffhq_{i:05d}.png")
            saved += 1
            if saved % 1000 == 0:
                print(f"    {saved}/{max_images} saved …")

        print(f"  Total FFHQ: {saved} images → {real_dir}")

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Fix: pip install datasets")


# -----------------------------------------------------------------------
# 2. CelebA-HQ (real faces)
# -----------------------------------------------------------------------

def download_celebahq(output_dir: Path, max_images: int = 10000):
    """CelebA-HQ 256px — celebrity face photos."""
    print(f"\n=== CelebA-HQ (real faces, up to {max_images}) ===")

    real_dir = output_dir / "real" / "celebahq"
    real_dir.mkdir(parents=True, exist_ok=True)

    existing = list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg"))
    if len(existing) >= max_images:
        print(f"  Already have {len(existing)} CelebA-HQ images — skipping.")
        return

    try:
        from datasets import load_dataset
        print(f"  Streaming CelebA-HQ (up to {max_images}) …")

        # Try multiple known mirrors
        for dataset_name in ["nielsr/CelebA-faces", "datasets/celebahq-faces", "kornia/CelebA-HQ"]:
            try:
                ds = load_dataset(dataset_name, split="train", streaming=True)
                saved = 0
                for i, sample in enumerate(ds):
                    if saved >= max_images:
                        break
                    img = sample.get("image") or sample.get("img")
                    if img is None:
                        continue
                    img.save(real_dir / f"celeba_{i:05d}.png")
                    saved += 1
                    if saved % 1000 == 0:
                        print(f"    {saved}/{max_images} saved …")
                print(f"  Total CelebA-HQ: {saved} images → {real_dir}")
                return
            except Exception:
                continue

        print("  CelebA-HQ not available via HuggingFace auto-download.")
        print("  Manual: download from https://github.com/tkarras/progressive_growing_of_gans")
        print(f"  Place 256x256 face images into: {real_dir}/")

    except ImportError:
        print("  ERROR: pip install datasets")


# -----------------------------------------------------------------------
# 3. DiffusionDB faces (fake — Stable Diffusion)
# -----------------------------------------------------------------------

def download_diffusiondb_faces(output_dir: Path, max_images: int = 5000):
    """
    DiffusionDB 2M — filter for face/portrait prompts only.
    Keywords: portrait, face, person, woman, man, girl, boy, selfie, headshot.
    """
    print(f"\n=== DiffusionDB Face Subset (fake SD faces, up to {max_images}) ===")

    fake_dir = output_dir / "fake" / "diffusiondb_faces"
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg"))
    if len(existing) >= max_images:
        print(f"  Already have {len(existing)} DiffusionDB face images — skipping.")
        return

    FACE_KEYWORDS = {
        "portrait", "face", "person", "woman", "man", "girl", "boy",
        "selfie", "headshot", "close-up", "closeup", "human", "people"
    }

    try:
        from datasets import load_dataset
        print(f"  Streaming DiffusionDB, filtering face prompts …")

        ds = load_dataset(
            "poloclub/diffusiondb",
            "2m_random_5k",
            split="train",
            streaming=True
        )

        saved = 0
        checked = 0
        for i, sample in enumerate(ds):
            if saved >= max_images:
                break
            checked += 1
            prompt = (sample.get("prompt") or "").lower()
            if not any(kw in prompt for kw in FACE_KEYWORDS):
                continue
            img = sample.get("image")
            if img is None:
                continue
            img.save(fake_dir / f"diffdb_face_{i:05d}.png")
            saved += 1
            if saved % 500 == 0:
                print(f"    {saved}/{max_images} (checked {checked}) …")

        if saved < max_images // 2:
            # Not enough face prompts in 5k subset — fall back to all images
            print(f"  Only {saved} face-prompt images in 5k subset.")
            print("  Falling back to larger subset (2m_random_10k) …")
            try:
                ds2 = load_dataset(
                    "poloclub/diffusiondb",
                    "2m_random_10k",
                    split="train",
                    streaming=True
                )
                for i, sample in enumerate(ds2):
                    if saved >= max_images:
                        break
                    prompt = (sample.get("prompt") or "").lower()
                    if not any(kw in prompt for kw in FACE_KEYWORDS):
                        continue
                    img = sample.get("image")
                    if img is None:
                        continue
                    img.save(fake_dir / f"diffdb_face_b_{i:05d}.png")
                    saved += 1
            except Exception:
                pass

        print(f"  Saved {saved} DiffusionDB face images → {fake_dir}")

    except Exception as e:
        print(f"  ERROR: {e}")
        print("  Manual: https://huggingface.co/datasets/poloclub/diffusiondb")


# -----------------------------------------------------------------------
# 4. GenImage face subset (fake — 8 generators)
# -----------------------------------------------------------------------

def download_genimage_faces(output_dir: Path, max_per_generator: int = 1500):
    """
    GenImage — face images from 8 AI generators.
    Filters for face/person images by checking the 'prompt' or 'label_name' field.
    """
    print(f"\n=== GenImage Face Subset (fake, up to {max_per_generator} per generator) ===")

    generators = [
        "stable_diffusion_v_1_4",
        "stable_diffusion_v_1_5",
        "Midjourney",
        "ADM",
        "DALL-E",
        "glide",
    ]

    FACE_KEYWORDS = {
        "portrait", "face", "person", "woman", "man", "girl", "boy",
        "human", "people", "head", "selfie"
    }

    try:
        from datasets import load_dataset

        for gen in generators:
            fake_dir = output_dir / "fake" / "genimage" / gen
            fake_dir.mkdir(parents=True, exist_ok=True)

            existing = list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg"))
            if len(existing) >= max_per_generator:
                print(f"  {gen}: already have {len(existing)} — skipping.")
                continue

            print(f"  Streaming {gen} …")
            try:
                ds = load_dataset(
                    "Wuvin/GenImage",
                    name=gen,
                    split="train",
                    streaming=True
                )

                saved = 0
                for i, sample in enumerate(ds):
                    if saved >= max_per_generator:
                        break
                    if sample.get("label", 1) != 1:
                        continue
                    prompt = (sample.get("prompt") or "").lower()
                    # If prompt available, filter; if not, accept all (GenImage is face-heavy)
                    if prompt and not any(kw in prompt for kw in FACE_KEYWORDS):
                        continue
                    img = sample.get("image")
                    if img is None:
                        continue
                    img.save(fake_dir / f"{gen}_{i:05d}.png")
                    saved += 1

                print(f"    → {saved} face images saved")

            except Exception as e:
                print(f"    WARN: {gen}: {e}")

    except ImportError:
        print("  ERROR: pip install datasets")
    except Exception as e:
        print(f"  ERROR: {e}")


# -----------------------------------------------------------------------
# 5. ForenSynths (manual — GAN-era face fakes)
# -----------------------------------------------------------------------

def print_forensynths_instructions(output_dir: Path):
    """ForenSynths ProGAN face subset — requires manual download."""
    fake_dir = output_dir / "fake" / "forensynths"
    print("\n=== ForenSynths (GAN-era ProGAN/StyleGAN faces) ===")
    print("  ForenSynths requires manual download (license agreement).")
    print("")
    print("  Steps:")
    print("  1. https://github.com/peterwang512/CNNDetection")
    print("  2. Download the Google Drive dataset")
    print("  3. Extract ONLY the face generators (progan, stylegan):")
    print(f"       {fake_dir}/progan/   ← ProGAN face images")
    print(f"       {fake_dir}/stylegan/ ← StyleGAN face images")
    print("")
    print("  NOTE: Skip non-face generators (biggan, cyclegan, etc.) —")
    print("  we are building a faces-only dataset.")
    print("")
    print("  Once placed, this provides the GAN-era baseline for cross-generator eval.")


# -----------------------------------------------------------------------
# Phase runners
# -----------------------------------------------------------------------

def phase1(max_ffhq: int = 10000, max_diffdb: int = 5000):
    """Phase 1 — Quick start (~6 GB): FFHQ + DiffusionDB faces."""
    print("=" * 65)
    print("PHASE 1 — Faces-Only Quick Start")
    print("Real: FFHQ faces | Fake: DiffusionDB SD faces")
    print("=" * 65)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_non_face_data(DATA_DIR)
    download_ffhq(DATA_DIR, max_images=max_ffhq)
    download_diffusiondb_faces(DATA_DIR, max_images=max_diffdb)
    print_forensynths_instructions(DATA_DIR)
    _print_summary()


def phase2(max_ffhq: int = 20000, max_celeba: int = 10000,
           max_diffdb: int = 10000, max_per_gen: int = 1500):
    """Phase 2 — Full faces dataset (~15 GB): all sources."""
    print("=" * 65)
    print("PHASE 2 — Full Faces Dataset")
    print("Real: FFHQ + CelebA-HQ | Fake: DiffusionDB + GenImage × 6")
    print("=" * 65)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean_non_face_data(DATA_DIR)
    download_ffhq(DATA_DIR, max_images=max_ffhq)
    download_celebahq(DATA_DIR, max_images=max_celeba)
    download_diffusiondb_faces(DATA_DIR, max_images=max_diffdb)
    download_genimage_faces(DATA_DIR, max_per_generator=max_per_gen)
    print_forensynths_instructions(DATA_DIR)
    _print_summary()


def _print_summary():
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)

    total_real = total_fake = 0

    for subdir, label in [("real", "REAL"), ("fake", "FAKE")]:
        d = DATA_DIR / subdir
        if not d.exists():
            continue
        for src in sorted(d.rglob("*")):
            if src.is_dir():
                imgs = list(src.glob("*.png")) + list(src.glob("*.jpg"))
                if imgs:
                    relpath = src.relative_to(DATA_DIR)
                    n = len(imgs)
                    print(f"  {label:4s}  {str(relpath):<45}  {n:>6,} images")
                    if label == "REAL":
                        total_real += n
                    else:
                        total_fake += n

    print(f"\n  Total real faces: {total_real:>8,}")
    print(f"  Total fake faces: {total_fake:>8,}")
    print(f"  Grand total:      {total_real + total_fake:>8,}")
    print("=" * 65)
    print(f"\nData directory: {DATA_DIR}")
    print("Next: python scripts/train.py --config configs/config.yaml")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download faces-only deepfake detection datasets"
    )
    parser.add_argument('--phase', type=str, default='1', choices=['1', '2'])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--clean', action='store_true',
                        help='Remove non-face data from existing dataset')
    parser.add_argument('--ffhq-only',   action='store_true')
    parser.add_argument('--celeba-only', action='store_true')
    parser.add_argument('--diffdb-only', action='store_true')
    parser.add_argument('--genimage-only', action='store_true')
    parser.add_argument('--max-images', type=int, default=None)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lim = args.max_images

    if args.clean:
        clean_non_face_data(DATA_DIR)
    elif args.ffhq_only:
        download_ffhq(DATA_DIR, max_images=lim or 10000)
    elif args.celeba_only:
        download_celebahq(DATA_DIR, max_images=lim or 10000)
    elif args.diffdb_only:
        download_diffusiondb_faces(DATA_DIR, max_images=lim or 5000)
    elif args.genimage_only:
        download_genimage_faces(DATA_DIR, max_per_generator=lim or 1500)
    elif args.all or args.phase == '2':
        phase2(
            max_ffhq=lim or 20000,
            max_celeba=lim or 10000,
            max_diffdb=lim or 10000,
            max_per_gen=lim or 1500
        )
    else:
        phase1(
            max_ffhq=lim or 10000,
            max_diffdb=lim or 5000
        )


if __name__ == "__main__":
    main()
