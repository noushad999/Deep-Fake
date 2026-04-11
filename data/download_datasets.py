"""
Automated dataset download and organization script.
Downloads ForenSynths, CIFake, Stable Diffusion v2.1, ImageNet, MS-COCO.
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class DatasetDownloader:
    """Handles downloading and organizing all datasets."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.fake_dir = self.data_dir / "fake"
        self.real_dir = self.data_dir / "real"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)
        self.real_dir.mkdir(parents=True, exist_ok=True)
    
    def download_cifake(
        self, 
        output_dir: Optional[Path] = None,
        max_samples: int = 20000
    ):
        """
        Download CIFake dataset from Hugging Face.
        CIFake contains images from Latent Diffusion models.
        """
        print("\n" + "="*60)
        print("DOWNLOADING: CIFake Dataset (Latent Diffusion)")
        print("="*60)
        
        output = output_dir or (self.fake_dir / "cifake")
        output.mkdir(parents=True, exist_ok=True)
        
        try:
            from datasets import load_dataset
            
            print("Loading CIFake from Hugging Face Hub...")
            dataset = load_dataset("CIFake")
            
            saved_count = 0
            for split in dataset:
                split_data = dataset[split]
                print(f"Processing split: {split} ({len(split_data)} images)")
                
                for i, img in enumerate(split_data['image']):
                    if saved_count >= max_samples:
                        break
                    
                    filename = f"cifake_fake_{split}_{i:06d}.png"
                    img.save(output / filename)
                    saved_count += 1
                    
                    if saved_count % 2000 == 0:
                        print(f"  Progress: {saved_count}/{max_samples}")
                
                if saved_count >= max_samples:
                    break
            
            print(f"\nCIFake download COMPLETE: {saved_count} images saved to {output}")
            return saved_count
            
        except ImportError:
            print("ERROR: 'datasets' library not found.")
            print("Install: pip install datasets huggingface-hub")
            return 0
        except Exception as e:
            print(f"ERROR downloading CIFake: {e}")
            print("Alternative: Download manually from https://huggingface.co/datasets/CIFake")
            return 0
    
    def download_coco_subset(
        self,
        output_dir: Optional[Path] = None,
        num_images: int = 10000
    ):
        """
        Download MS-COCO subset (real images).
        Uses COCO 2017 validation set (~5000 images).
        """
        print("\n" + "="*60)
        print("DOWNLOADING: MS-COCO Subset (Real Images)")
        print("="*60)
        
        output = output_dir or (self.real_dir / "coco")
        output.mkdir(parents=True, exist_ok=True)
        
        try:
            from datasets import load_dataset
            
            print("Loading MS-COCO 2017 validation set...")
            dataset = load_dataset("HuggingFaceM4/COCO", split="validation")
            
            saved_count = 0
            for i, img in enumerate(dataset['image']):
                if saved_count >= num_images:
                    break
                
                filename = f"coco_real_{i:06d}.png"
                img.save(output / filename)
                saved_count += 1
                
                if saved_count % 1000 == 0:
                    print(f"  Progress: {saved_count}/{num_images}")
            
            print(f"\nMS-COCO download COMPLETE: {saved_count} images saved to {output}")
            return saved_count
            
        except ImportError:
            print("ERROR: 'datasets' library not found.")
            return 0
        except Exception as e:
            print(f"ERROR downloading MS-COCO: {e}")
            print("Alternative: https://cocodataset.org/#download")
            return 0
    
    def download_stable_diffusion_samples(
        self,
        output_dir: Optional[Path] = None,
        num_samples: int = 5000,
        prompts_file: Optional[str] = None
    ):
        """
        Generate fake images using Stable Diffusion v2.1.
        Requires: pip install diffusers accelerate transformers
        """
        print("\n" + "="*60)
        print("GENERATING: Stable Diffusion v2.1 Fake Images")
        print("="*60)
        
        output = output_dir or (self.fake_dir / "stable_diffusion")
        output.mkdir(parents=True, exist_ok=True)
        
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Load pipeline
            print("Loading Stable Diffusion v2.1 pipeline...")
            model_id = "stabilityai/stable-diffusion-2-1"
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            pipe = pipe.to(device)
            pipe.safety_checker = None  # Disable for generation
            
            # Default prompts if none provided
            if prompts_file and os.path.exists(prompts_file):
                with open(prompts_file, 'r') as f:
                    prompts = [line.strip() for line in f.readlines() if line.strip()]
            else:
                prompts = [
                    "a portrait of a person, photorealistic, 4k",
                    "a beautiful landscape with mountains and lake",
                    "a city street at night with neon lights",
                    "a close-up of a cat sitting on a couch",
                    "an aerial view of a beach with waves",
                    "a professional headshot of a business person",
                    "a flower garden in spring with butterflies",
                    "a modern kitchen interior with sunlight",
                    "a vintage car on a country road",
                    "a group of friends at a party, candid photo"
                ]
            
            print(f"Generating {num_samples} images with {len(prompts)} prompts...")
            
            for i in range(num_samples):
                prompt = prompts[i % len(prompts)]
                
                image = pipe(
                    prompt,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images[0]
                
                filename = f"sd_fake_{i:06d}.png"
                image.save(output / filename)
                
                if (i + 1) % 500 == 0:
                    print(f"  Progress: {i + 1}/{num_samples}")
            
            print(f"\nStable Diffusion generation COMPLETE: {num_samples} images")
            return num_samples
            
        except ImportError as e:
            print(f"ERROR: Required libraries not found: {e}")
            print("Install: pip install diffusers accelerate transformers torch")
            return 0
        except Exception as e:
            print(f"ERROR generating SD images: {e}")
            return 0
    
    def download_forensynths_info(self):
        """
        ForenSynths requires manual download due to licensing.
        Provides instructions and creates directory structure.
        """
        print("\n" + "="*60)
        print("FORENSYNTHS DATASET (Manual Download Required)")
        print("="*60)
        
        output = self.fake_dir / "forensynths"
        output.mkdir(parents=True, exist_ok=True)
        
        print("\nForenSynths contains GAN-generated images from:")
        print("  - ProGAN")
        print("  - StyleGAN")
        print("  - StyleGAN2")
        print("  - BigGAN")
        print("  - CycleGAN")
        print("  - StarGAN")
        print("  - GauGAN")
        print("  - Deepfake (face swaps)")
        print("\nDownload from: https://github.com/ZhendongWang6/ForenSynths")
        print(f"\nAfter downloading, organize as: {output}/")
        print("  ├── progan/")
        print("  ├── stylegan/")
        print("  ├── stylegan2/")
        print("  └── ...")
        
        # Create placeholder structure
        for generator in ['progan', 'stylegan', 'stylegan2', 'biggan', 
                         'cyclegan', 'stargan', 'gaugan', 'deepfake']:
            (output / generator).mkdir(exist_ok=True)
        
        print(f"\nDirectory structure created at: {output}")
        print("Place downloaded images in corresponding subdirectories.")
    
    def download_all_phase1(self):
        """Download Phase 1 datasets (lightweight ~15GB)."""
        print("\n" + "#"*60)
        print("# PHASE 1 DATASET DOWNLOAD (Lightweight)")
        print("#"*60)
        
        cifake_count = self.download_cifake(max_samples=10000)
        coco_count = self.download_coco_subset(num_images=5000)
        
        print("\n" + "="*60)
        print("PHASE 1 SUMMARY")
        print("="*60)
        print(f"  Fake images (CIFake): {cifake_count}")
        print(f"  Real images (COCO):   {coco_count}")
        print(f"  Total:                {cifake_count + coco_count}")
        print(f"  Location:             {self.data_dir}")
        print("="*60)
        
        return cifake_count + coco_count
    
    def download_all_phase2(self):
        """Download Phase 2 datasets (full ~200GB)."""
        print("\n" + "#"*60)
        print("# PHASE 2 DATASET DOWNLOAD (Full)")
        print("#"*60)
        
        print("\nPhase 2 includes:")
        print("  1. ForenSynths (manual)")
        print("  2. Stable Diffusion v2.1 (generated)")
        print("  3. ImageNet (registration required)")
        print("  4. Full MS-COCO")
        
        self.download_forensynths_info()
        
        # Ask user if they want to generate SD images
        gen_sd = input("\nGenerate Stable Diffusion samples now? (y/n): ")
        if gen_sd.lower() == 'y':
            num = int(input("Number of samples (default 5000): ") or 5000)
            self.download_stable_diffusion_samples(num_samples=num)
        
        print("\nPhase 2 setup complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download datasets for deepfake detection"
    )
    parser.add_argument(
        '--phase', type=int, default=1, choices=[1, 2],
        help='Dataset phase to download (1=lightweight, 2=full)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='./data',
        help='Root data directory'
    )
    parser.add_argument(
        '--max-fake', type=int, default=10000,
        help='Max fake images to download'
    )
    parser.add_argument(
        '--max-real', type=int, default=5000,
        help='Max real images to download'
    )
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.phase == 1:
        downloader.download_all_phase1()
    else:
        downloader.download_all_phase2()


if __name__ == "__main__":
    main()
