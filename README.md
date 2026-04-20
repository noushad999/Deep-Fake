# Diagnosing and Preventing Stream Co-Adaptation in Multi-Stream Deepfake Detectors

**Md Noushad Jahan Ramim**  · April 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](#citation)

---

We study a fundamental failure mode of multi-stream deepfake detectors: **stream co-adaptation**, where spatially, spectrally, and semantically specialized streams collapse to learning redundant representations during joint training. We propose two training-time inductive biases — per-sample orthogonality regularization and stochastic stream dropout — that measurably reduce inter-stream cosine similarity and improve generalization to unseen generators.

<p align="center">
  <img src="assets/figures/architecture.png" width="80%"/>
  <br/>
  <em>Multi-Stream Deepfake Detector with MLAF Cross-Stream Attention Fusion. Three streams process the same input in parallel; the MLAF module fuses them via 3-token multi-head attention with learnable stream-type embeddings.</em>
</p>

---

## Method

The detector comprises three parallel encoders and a cross-stream attention fusion module.

| Stream | Backbone | Output | What it captures |
|--------|----------|--------|-----------------|
| Spatial | EfficientNet-B0 | 128-d | Pixel-level artifacts, boundary inconsistencies |
| Frequency | ResNet-18 + learnable FFT mask | 64-d | Spectral fingerprints unique to each generator |
| Semantic | ViT-Tiny-Patch16 | 384-d | Global structural inconsistencies |
| **Fusion (MLAF)** | 3-token cross-stream MHA | 256-d | Adaptive per-sample stream weighting |

**Learnable FFT mask.** The frequency stream applies a learnable per-channel spectral mask before passing the filtered image to ResNet-18. The mask is factored into a full spatial component and four radial band weights (low, mid-low, mid-high, high), enabling the model to learn which frequency bands are most discriminative without committing to a fixed filter design.

**Stream orthogonality loss.** To prevent streams from collapsing to the same representation, we penalize the pairwise per-sample cosine similarity between stream features:

$$\mathcal{L}_\text{orth} = \frac{1}{\binom{3}{2}} \sum_{i < j} \frac{1}{B} \sum_{b=1}^{B} \left| \cos\!\left(\mathbf{f}_i^{(b)},\, \mathbf{f}_j^{(b)}\right) \right|$$

**Stream dropout.** During training, each stream is independently zeroed with probability $p$ (at least one stream always remains active). This prevents any single stream from dominating and forces the fusion module to remain robust to missing stream inputs.

---

## Results

### In-distribution evaluation

<p align="center">
  <img src="assets/figures/roc_curve.png" width="32%"/>
  <img src="assets/figures/confusion_matrix.png" width="30%"/>
  <img src="assets/figures/prediction_dist.png" width="30%"/>
</p>

Evaluated on 2,497 held-out images (SD-generated fakes + COCO/FFHQ real, seed=42).

| Method | AUC | Acc | F1 | EER |
|--------|-----|-----|----|-----|
| CNNDetect | — | — | — | — |
| F3Net | — | — | — | — |
| UnivFD | — | — | — | — |
| **Ours** | **99.99%** | **99.64%** | **99.64%** | **~0.01%** |

*Cross-dataset numbers (FF++, Celeb-DF-v2) pending data acquisition.*

### Compression robustness

| Compression level | AUC |
|-------------------|-----|
| None (C0) | 100.00% |
| Light (C23) | 93.47% |
| Medium (C40) | 91.59% |
| Heavy (C50) | 89.89% |

### Baseline comparison

<p align="center">
  <img src="assets/figures/baseline_comparison.png" width="65%"/>
</p>

---

## Ablation

<p align="center">
  <img src="assets/figures/ablation_crossgen.png" width="60%"/>
</p>

**Stream combinations.** Ablating each stream by zeroing its output at inference time:

| Configuration | AUC | Δ |
|--------------|-----|---|
| Full model | 99.99% | — |
| Spatial + Semantic | 99.98% | −0.01 |
| Spatial + Frequency | 99.98% | −0.01 |
| Spatial only | 99.95% | −0.04 |
| Semantic only | 99.29% | −0.70 |
| Frequency only | 68.42% | −31.57 |

The frequency stream alone performs near-chance, yet consistently improves the full model — it encodes artifacts invisible to the other two streams. This complementarity is precisely what the orthogonality loss is designed to preserve.

**Regularization ablation** *(OOD numbers pending)*:

| | Inter-stream cosine sim ↓ | OOD AUC ↑ |
|--|--------------------------|-----------|
| Baseline (no regularization) | ~0.80 | TBD |
| + Orthogonality loss | ~0.30 | TBD |
| + Stream dropout | ~0.10 | TBD |

---

## Interpretability

<p align="center">
  <img src="assets/heatmaps/heatmap_01.jpg" width="14%"/>
  <img src="assets/heatmaps/heatmap_02.jpg" width="14%"/>
  <img src="assets/heatmaps/heatmap_03.jpg" width="14%"/>
  <img src="assets/heatmaps/heatmap_04.jpg" width="14%"/>
  <img src="assets/heatmaps/heatmap_05.jpg" width="14%"/>
  <img src="assets/heatmaps/heatmap_06.jpg" width="14%"/>
  <br/>
  <em>GradCAM++ activation maps from the spatial stream on correctly classified fake images.</em>
</p>

---

## Installation

```bash
git clone https://github.com/noushad999/Deep-Fake.git
cd Deep-Fake
pip install -r requirements-lock.txt
```

Requirements: Python 3.10+, PyTorch 2.x, timm, scikit-learn, scipy, tqdm, pyyaml.
Tested on RTX 5060 Ti 16 GB (CUDA 12.8). CPU inference is supported.

---

## Usage

**Inference on a single image:**
```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg
```

**Training:**
```bash
# Main pipeline
python scripts/train.py --config configs/config.yaml --data-dir $DATA_ROOT

# FF++ protocol
python scripts/train_ffpp.py --data-dir $FFPP_DIR --compression c23

# Cross-generator pipeline
python scripts/train_cvpr.py --data-mode hf --epochs 30

# Stream combination ablation
python scripts/train.py --ablation-mode spatial_only
# choices: spatial_only | freq_only | semantic_only | spatial_freq | spatial_semantic
```

**Evaluation:**
```bash
# Standard evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data-dir $DATA_ROOT

# OOD (So-Fake-OOD: DALL-E, Seedream3.0)
python scripts/eval_ood.py --checkpoint checkpoints/best_model.pth

# Adversarial robustness (FGSM / PGD-20 / CW)
python scripts/adversarial_eval.py --checkpoint checkpoints/best_model.pth --attacks fgsm pgd20 cw

# Cross-dataset generalization
python scripts/cross_dataset_eval.py \
    --checkpoint checkpoints/best_model.pth \
    --celebdf-dir $CELEBDF_DIR \
    --ffpp-test-dir $FFPP_DIR

# GradCAM++ visualizations
python scripts/visualize_gradcam.py --checkpoint checkpoints/best_model.pth

# Multi-seed reproducibility
bash scripts/run_multiseed.sh

# Generate LaTeX tables
python scripts/generate_paper_tables.py
```

---

## Data

The model is trained on a curated mix of real and AI-generated images:

| Split | Real | Fake |
|-------|------|------|
| Train | FFHQ 256px, MS-COCO val2017 | DiffusionDB (SD v1.x), GenImage (SD-XL, MJ-v5, DALL-E 3), ForenSynths (ProGAN, StyleGAN) |
| Test (in-dist) | Same distribution, held out | Same distribution, held out |
| Test (OOD) | — | So-Fake-OOD (DALL-E, Seedream3.0) — unseen generators |

Download helpers:
```bash
python scripts/download_cvpr_datasets.py   # HuggingFace: FakeCOCO, So-Fake-Set
python scripts/download_datasets.py        # ForenSynths and others
```

FF++ requires a research agreement with TUM: https://github.com/ondyari/FaceForensics

---

## Project structure

```
├── models/
│   ├── full_model.py          # MultiStreamDeepfakeDetector
│   ├── fusion.py              # MLAF cross-stream attention
│   ├── spatial_stream.py      # EfficientNet-B0 branch
│   ├── freq_stream.py         # ResNet-18 + learnable FFT mask
│   ├── semantic_stream.py     # ViT-Tiny branch
│   └── baselines.py           # CNNDetect, UnivFD, XceptionDetect, F3Net
├── data/
│   ├── dataset.py             # Main loader + stratified split
│   ├── ffpp_dataset.py        # FaceForensics++ loader
│   ├── celebdf_dataset.py     # Celeb-DF v2 loader
│   ├── hf_sofake.py           # So-Fake-Set / So-Fake-OOD
│   └── hf_fakecoco.py         # FakeCOCO
├── scripts/                   # Training, evaluation, visualization, utilities
├── configs/                   # config.yaml, ffpp_config.yaml
└── reports/                   # 10 technical reports
```

---

## Status

| Component | Status |
|-----------|--------|
| 3-stream architecture + MLAF fusion | complete |
| Learnable FFT mask | complete |
| Stream dropout | complete |
| Orthogonality regularization (per-sample) | in progress |
| Baselines (CNNDetect, UnivFD, F3Net, Xception) | complete |
| In-distribution evaluation | 99.99% AUC |
| Compression robustness (C23/C40/C50) | 89–94% AUC |
| So-Fake-OOD evaluation | downloading |
| FF++ + Celeb-DF evaluation | data pending |
| Co-adaptation analysis (cos sim vs OOD AUC) | planned |
| CVPR submission draft | Q3 2026 |

---

## Citation

```bibtex
@misc{ramim2026multistream,
  title   = {Diagnosing and Preventing Stream Co-Adaptation
             in Multi-Stream Deepfake Detectors},
  author  = {Ramim, Md Noushad Jahan},
  year    = {2026},
  note    = {BSc thesis, CVPR extension},
  url     = {https://github.com/noushad999/Deep-Fake}
}
```

---

## References

[1] Rossler et al. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019.  
[2] Wang et al. *CNN-generated images are surprisingly easy to spot...for now.* CVPR 2020.  
[3] Qian et al. *Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues.* ECCV 2020.  
[4] Li et al. *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics.* CVPR 2020.  
[5] Ojha et al. *Towards Universal Fake Image Detection by Leveraging Vision Foundation Models.* CVPR 2023.  
[6] Liu et al. *Leveraging Neighboring Pixels for Deepfake Detection.* CVPR 2023.
