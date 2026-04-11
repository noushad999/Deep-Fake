"""
Dimension Verification Test Script
Verifies all stream outputs and full model forward pass.
Run this to validate architecture before training.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.spatial_stream import NPRBranch
from models.freq_stream import FreqBlender
from models.semantic_stream import FATLiteTransformer
from models.fusion import MLAFFusion
from models.full_model import MultiStreamDeepfakeDetector


def test_individual_streams():
    """Test each stream independently."""
    print("\n" + "="*70)
    print("STREAM DIMENSION VERIFICATION")
    print("="*70)
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256)
    
    # Stream 1: NPR
    print("\n[1/4] Testing NPR Branch (Spatial)...")
    npr = NPRBranch(output_dim=128, pretrained=False)
    npr.eval()
    with torch.no_grad():
        out = npr(x)
    assert out.shape == (batch_size, 128), f"NPR FAIL: {out.shape}"
    print(f"       Input: {x.shape} -> Output: {out.shape} [PASS]")
    
    # Stream 2: FreqBlender
    print("[2/4] Testing FreqBlender (Frequency)...")
    freq = FreqBlender(output_dim=64, pretrained=False, img_size=256)
    freq.eval()
    with torch.no_grad():
        out = freq(x)
    assert out.shape == (batch_size, 64), f"FREQ FAIL: {out.shape}"
    print(f"       Input: {x.shape} -> Output: {out.shape} [PASS]")
    
    # Stream 3: FAT-Lite
    print("[3/4] Testing FAT-Lite (Semantic)...")
    fat = FATLiteTransformer(output_dim=384, pretrained=False)
    fat.eval()
    with torch.no_grad():
        out = fat(x)
    assert out.shape == (batch_size, 384), f"FAT FAIL: {out.shape}"
    print(f"       Input: {x.shape} -> Output: {out.shape} [PASS]")
    
    # Fusion
    print("[4/4] Testing MLAF Fusion...")
    mlaf = MLAFFusion(spatial_dim=128, freq_dim=64, semantic_dim=384)
    mlaf.eval()
    with torch.no_grad():
        logits, fused = mlaf(
            torch.randn(batch_size, 128),
            torch.randn(batch_size, 64),
            torch.randn(batch_size, 384)
        )
    assert logits.shape == (batch_size, 1), f"MLAF LOGITS FAIL: {logits.shape}"
    assert fused.shape == (batch_size, 256), f"MLAF FUSED FAIL: {fused.shape}"
    print(f"       Inputs: [B,128]+[B,64]+[B,384] -> Logits: {logits.shape}, Fused: {fused.shape} [PASS]")
    
    print("\n" + "="*70)
    print("ALL STREAM TESTS PASSED!")
    print("="*70)


def test_full_model():
    """Test complete end-to-end model."""
    print("\n" + "="*70)
    print("FULL MODEL VERIFICATION")
    print("="*70)
    
    batch_size = 4
    model = MultiStreamDeepfakeDetector(pretrained_backbones=False)
    model.eval()
    
    x = torch.randn(batch_size, 3, 256, 256)
    
    with torch.no_grad():
        logits, features = model(x, return_features=True)
    
    print(f"\nInput:              {x.shape}")
    print(f"Logits:             {logits.shape}")
    print(f"Spatial features:   {features['spatial'].shape}")
    print(f"Frequency features: {features['frequency'].shape}")
    print(f"Semantic features:  {features['semantic'].shape}")
    print(f"Fused features:     {features['fused'].shape}")
    print(f"Combined features:  {features['combined'].shape}")
    
    # Strict assertions
    assert logits.shape == (batch_size, 1)
    assert features['spatial'].shape == (batch_size, 128)
    assert features['frequency'].shape == (batch_size, 64)
    assert features['semantic'].shape == (batch_size, 384)
    assert features['combined'].shape == (batch_size, 576)
    assert features['fused'].shape == (batch_size, 256)
    
    # Parameter count
    print(f"\n{'='*70}")
    print("PARAMETER ANALYSIS:")
    print(f"{'='*70}")
    per_stream = model.count_parameters_per_stream()
    for name, count in per_stream.items():
        print(f"  {name:20s}: {count:>12,} params")
    
    total_m = per_stream['total'] / 1e6
    print(f"\n  TOTAL: {per_stream['total']:,} ({total_m:.2f}M params)")
    
    # Check if within target
    if 20_000_000 <= per_stream['total'] <= 30_000_000:
        print(f"  STATUS: Within target range (20-30M) [PASS]")
    else:
        print(f"  STATUS: Outside target range (20-30M) - review backbones")
    
    # Test prediction
    preds = model.predict(x)
    probs = model.get_probabilities(x)
    
    print(f"\nPrediction test:  preds={preds.tolist()}, probs={probs.tolist()}")
    
    print("\n" + "="*70)
    print("FULL MODEL TEST PASSED!")
    print("Architecture verified. Ready for training.")
    print("="*70)


if __name__ == "__main__":
    test_individual_streams()
    test_full_model()
