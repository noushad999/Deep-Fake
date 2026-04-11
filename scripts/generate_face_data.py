"""Generate face deepfake dataset locally using pre-trained StyleGAN.
No download needed - generate fake faces, use small real face dataset."""
import torch
import os
from pathlib import Path

def check_stylegan_availability():
    """Check if we can load StyleGAN from torch hub or pip."""
    try:
        # Try loading StyleGAN2 from torch hub
        print("Attempting torch.hub StyleGAN2-ADA...")
        G = torch.hub.load("huggingface/pytorch-image-models", "stylegan2", trust_repo=True)
        print("SUCCESS: StyleGAN loaded from timm!")
        return G
    except Exception as e:
        print(f"torch.hub StyleGAN failed: {e}")
    
    try:
        # Try pretrainedmodels
        import pretrainedmodels
        print("Checking pretrainedmodels...")
    except:
        print("pretrainedmodels not available")
    
    try:
        # Check if stylegan2-pytorch is installed
        import stylegan2
        print("stylegan2 package available")
    except:
        print("stylegan2 not installed")
    
    try:
        # Check diffusers
        from diffusers import StableDiffusionPipeline
        print("diffusers available")
        return "diffusers"
    except Exception as e:
        print(f"diffusers not available: {e}")
    
    return None

if __name__ == "__main__":
    print("Checking available face generation methods...")
    result = check_stylegan_availability()
    print(f"\nAvailable: {result}")
