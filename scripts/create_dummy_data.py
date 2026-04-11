"""
Create dummy dataset for testing the training pipeline without real data.
Generates synthetic real/fake images with simple patterns.
"""
import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def create_dummy_image(img_type: str, idx: int, size: int = 256) -> Image.Image:
    """
    Create a dummy image with distinguishable patterns for real vs fake.
    Fake images have high-frequency noise patterns; real images are smoother.
    """
    rng = np.random.RandomState(idx)
    
    if img_type == 'real':
        # Smooth gradient patterns (simulate natural images)
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        
        r = (0.5 + 0.3 * X + 0.1 * np.sin(2 * np.pi * Y)).astype(np.float32)
        g = (0.4 + 0.2 * Y + 0.15 * np.sin(3 * np.pi * X)).astype(np.float32)
        b = (0.6 + 0.1 * X * Y + 0.1 * np.cos(2 * np.pi * (X + Y))).astype(np.float32)
        
        # Add mild noise
        r = np.clip(r + rng.randn(size, size) * 0.02, 0, 1)
        g = np.clip(g + rng.randn(size, size) * 0.02, 0, 1)
        b = np.clip(b + rng.randn(size, size) * 0.02, 0, 1)
        
    else:  # fake
        # High-frequency noise + grid patterns (simulate GAN artifacts)
        r = rng.rand(size, size).astype(np.float32) * 0.5 + 0.25
        g = rng.rand(size, size).astype(np.float32) * 0.5 + 0.25
        b = rng.rand(size, size).astype(np.float32) * 0.5 + 0.25
        
        # Add grid artifacts (common in GANs)
        for i in range(0, size, 8):
            r[i, :] = np.clip(r[i, :] + 0.3, 0, 1)
            g[:, i] = np.clip(g[:, i] + 0.2, 0, 1)
        
        # Checkerboard pattern
        checker = np.zeros((size, size), dtype=np.float32)
        for i in range(0, size, 16):
            for j in range(0, size, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    checker[i:i+8, j:j+8] = 0.3
        
        r = np.clip(r + checker, 0, 1)
        g = np.clip(g + checker * 0.5, 0, 1)
    
    # Convert to uint8
    img_array = np.stack([r, g, b], axis=-1)
    img_array = np.uint8(img_array * 255)
    
    return Image.fromarray(img_array)


def create_dummy_dataset(
    output_dir: str,
    n_real: int = 2000,
    n_fake: int = 2000,
    img_size: int = 256
):
    """Create a dummy dataset for testing."""
    data_dir = Path(output_dir)
    
    # Create directory structure
    real_dir = data_dir / 'real' / 'dummy'
    fake_dir = data_dir / 'fake' / 'dummy'
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy dataset at: {data_dir}")
    print(f"  Real images: {n_real}")
    print(f"  Fake images: {n_fake}")
    print(f"  Image size:  {img_size}")
    
    # Generate real images
    print("\nGenerating real images...")
    for i in tqdm(range(n_real)):
        img = create_dummy_image('real', i, img_size)
        img.save(real_dir / f'real_{i:05d}.png')
    
    # Generate fake images
    print("\nGenerating fake images...")
    for i in tqdm(range(n_fake)):
        img = create_dummy_image('fake', i + n_real, img_size)
        img.save(fake_dir / f'fake_{i:05d}.png')
    
    total = n_real + n_fake
    print(f"\nDummy dataset complete: {total} images total")
    print(f"Location: {data_dir}")
    
    return total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='../data')
    parser.add_argument('--n-real', type=int, default=2000)
    parser.add_argument('--n-fake', type=int, default=2000)
    parser.add_argument('--img-size', type=int, default=256)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_dummy_dataset(
        output_dir=args.output_dir,
        n_real=args.n_real,
        n_fake=args.n_fake,
        img_size=args.img_size
    )
