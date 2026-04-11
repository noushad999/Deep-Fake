"""
Dataset Download Script
Downloads and organizes datasets for deepfake detection

Phase 1 (Lightweight ~15GB):
- CIFake (fake images from Latent Diffusion)
- MS-COCO subset (real images)

Phase 2 (Full ~200GB):
- ForenSynths (ProGAN, StyleGAN, etc.)
- Stable Diffusion v2.1
- ImageNet
- Full MS-COCO
"""
import os
import subprocess
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def download_cifake(output_dir, max_samples=None):
    """
    Download CIFake dataset (Latent Diffusion generated images)
    Available via Hugging Face Datasets
    """
    print("\n=== Downloading CIFake Dataset ===")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # Load CIFake from Hugging Face
        print("Loading CIFake dataset from Hugging Face...")
        dataset = load_dataset("CIFake")
        
        # Save fake images
        print("Saving fake images...")
        fake_dir = output_dir / "fake"
        os.makedirs(fake_dir, exist_ok=True)
        
        for i, img in enumerate(dataset['train']['image']):
            if max_samples and i >= max_samples:
                break
            img.save(fake_dir / f"fake_{i:05d}.png")
            if i % 1000 == 0:
                print(f"  Saved {i} fake images...")
        
        print(f"CIFake fake images saved to: {fake_dir}")
        
    except Exception as e:
        print(f"Error downloading CIFake: {e}")
        print("Alternative: Download manually from https://huggingface.co/datasets/CIFake")


def download_coco_subset(output_dir, num_images=5000):
    """
    Download MS-COCO subset (real images)
    Using COCO 2017 val set (smaller, 5000 images)
    """
    print("\n=== Downloading MS-COCO Subset ===")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from datasets import load_dataset
        
        # Load MS-COCO 2017 (val set is ~5000 images)
        print("Loading MS-COCO 2017 validation set...")
        dataset = load_dataset("HuggingFaceM4/COCO", split="validation")
        
        real_dir = output_dir / "real"
        os.makedirs(real_dir, exist_ok=True)
        
        for i, img in enumerate(dataset['image']):
            if i >= num_images:
                break
            img.save(real_dir / f"real_{i:05d}.png")
            if i % 500 == 0:
                print(f"  Saved {i} real images...")
        
        print(f"MS-COCO real images saved to: {real_dir}")
        
    except Exception as e:
        print(f"Error downloading MS-COCO: {e}")
        print("Alternative: Download from https://cocodataset.org/")


def download_stable_diffusion_samples(output_dir, num_samples=2000):
    """
    Generate/download Stable Diffusion v2.1 fake images
    """
    print("\n=== Downloading Stable Diffusion Samples ===")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Note: Generating Stable Diffusion images requires the diffusers library")
    print("For now, you can use pre-generated samples from:")
    print("https://huggingface.co/stabilityai/stable-diffusion-2-1")
    print("\nTo generate your own:")
    print("  pip install diffusers")
    print("  See: scripts/generate_sd_samples.py")


def download_forensynths(output_dir):
    """
    Download ForenSynths dataset (ProGAN, StyleGAN, etc.)
    """
    print("\n=== ForenSynths Dataset ===")
    print("ForenSynths requires manual download from:")
    print("https://github.com/ZhendongWang6/ForenSynths")
    print("\nAfter downloading, place in:", output_dir)


def setup_phase1():
    """Setup Phase 1 datasets (lightweight ~15GB)"""
    print("=" * 60)
    print("PHASE 1 DATASET SETUP (Lightweight ~15GB)")
    print("=" * 60)
    
    cifake_dir = DATA_DIR / "cifake"
    coco_dir = DATA_DIR / "coco"
    
    # Download CIFake (fake)
    download_cifake(cifake_dir, max_samples=10000)
    
    # Download MS-COCO subset (real)
    download_coco_subset(coco_dir, num_images=5000)
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE!")
    print(f"Data location: {DATA_DIR}")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument('--phase', type=str, default='1', choices=['1', '2'],
                       help='Dataset phase (1=lightweight, 2=full)')
    args = parser.parse_args()
    
    if args.phase == '1':
        setup_phase1()
    else:
        print("Phase 2 setup coming soon!")
        print("This includes ForenSynths, full Stable Diffusion, ImageNet, etc.")


if __name__ == "__main__":
    main()
