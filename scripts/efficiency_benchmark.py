"""
Model Efficiency Benchmark — CVPR Table
=========================================
Reports: Parameters, FLOPs, Inference latency, Throughput (FPS)
Compares all models side-by-side for the paper's efficiency table.

Usage:
    python scripts/efficiency_benchmark.py
    python scripts/efficiency_benchmark.py --batch-sizes 1 8 32
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.full_model import MultiStreamDeepfakeDetector
from models.baselines import CNNDetect, XceptionDetect, F3Net


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, input_size=(1, 3, 224, 224)):
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.zeros(input_size)
        flops = FlopCountAnalysis(model, x).total()
        return flops / 1e9  # GFLOPs
    except ImportError:
        return None  # fvcore not installed


def measure_latency(model, device, input_size=(1, 3, 224, 224), n_runs=100, warmup=10):
    model.eval()
    x = torch.randn(input_size).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    return np.mean(times), np.std(times)


def measure_throughput(model, device, batch_size=32, img_size=224, n_runs=20):
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size).to(device)

    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = (batch_size * n_runs) / elapsed
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-runs", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Our model uses 256×256 (FFT mask size); baselines use 224×224
    models = {
        "CNNDetect (ResNet-50)":  (CNNDetect(pretrained=False),    224),
        "XceptionDetect":         (XceptionDetect(pretrained=False), 224),
        "F3Net":                  (F3Net(pretrained=False),          224),
        "Ours (3-Stream Fusion)": (MultiStreamDeepfakeDetector(pretrained_backbones=False), 256),
    }

    # Parameters table
    print("=" * 65)
    print(f"{'Model':<28} {'Total Params':>14} {'Trainable':>12} {'GFLOPs':>8}")
    print("-" * 65)
    for name, (model, img_sz) in models.items():
        total, trainable = count_params(model)
        flops = measure_flops(model.cpu(), (1, 3, img_sz, img_sz))
        flops_str = f"{flops:.2f}" if flops else "N/A"
        print(f"{name:<28} {total/1e6:>13.2f}M {trainable/1e6:>11.2f}M {flops_str:>8}")

    print("\n" + "=" * 55)
    print(f"{'Model':<28} {'Latency (ms)':>14} {'Std':>8}")
    print("-" * 55)
    for name, (model, img_sz) in models.items():
        model = model.to(device).eval()
        lat, std = measure_latency(model, device,
                                   input_size=(1, 3, img_sz, img_sz),
                                   n_runs=args.n_runs)
        print(f"{name:<28} {lat:>13.2f}  {std:>7.2f}")

    print("\n" + "=" * 55)
    print("Throughput (images/sec) at different batch sizes:")
    header = f"{'Model':<28}" + "".join(f" {'B='+str(b):>9}" for b in args.batch_sizes)
    print(header)
    print("-" * len(header))
    for name, (model, img_sz) in models.items():
        model = model.to(device).eval()
        row = f"{name:<28}"
        for bs in args.batch_sizes:
            fps = measure_throughput(model, device, batch_size=bs, img_size=img_sz)
            row += f" {fps:>9.0f}"
        print(row)

    print("\nNote: Latency measured on", str(device).upper(),
          "| Single-image (batch=1) inference")
    print("FLOPs require: pip install fvcore")


if __name__ == "__main__":
    main()
