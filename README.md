<div align="center">

# Multi-Stream Deepfake Face Detection

**A generalizable deepfake detection system using spatial, frequency, and semantic stream fusion with cross-generator evaluation**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)](LICENSE)

*BSc Thesis Project — Group 3*

</div>

---

## Overview

This project implements a **3-stream neural network** for detecting AI-generated face images (deepfakes). Each stream analyzes a different aspect of the image — pixel textures, frequency patterns, and high-level structure — and a cross-attention fusion module combines their outputs into a final prediction.

The key finding: the full 3-stream model achieves **only −1.91% generalization drop** when tested on an unseen generator (SDXL), compared to −10.24% for a frequency-only baseline — demonstrating that multi-stream fusion learns more universal forgery cues.

---

## Architecture

![Architecture Diagram](assets/figures/architecture.png)

| Stream | Backbone | Output Dim | What It Detects |
|--------|----------|-----------|-----------------|
| **Spatial (NPR)** | EfficientNet-B0 | 128 | Pixel-level texture artifacts, boundary inconsistencies |
| **Frequency (FreqBlender)** | ResNet-18 + Learnable FFT | 64 | Spectral artifacts, periodic noise in frequency domain |
| **Semantic (FAT-Lite)** | ViT-Tiny patch16 | 384 | Global structural inconsistencies, face region relationships |
| **Fusion (MLAF)** | Cross-Stream Attention | 256 | Inter-stream relationships, final classification |

**Total Parameters:** ~22.9M &nbsp;|&nbsp; **Best Epoch:** 15 &nbsp;|&nbsp; **Hardware:** RTX 5060 Ti 16GB, CUDA 12.8

---

## Results

### Baseline Comparison

![Baseline Comparison](assets/figures/baseline_comparison.png)

| Model | AUC-ROC | Accuracy | Recall | F1 | EER |
|-------|---------|----------|--------|----|-----|
| CNNDetect (Wang et al., CVPR 2020) | 85.2% | 79.8% | 78.3% | 77.1% | 15.2% |
| UnivFD (Ojha et al., CVPR 2023) | 91.4% | 86.7% | 85.4% | 84.3% | 9.1% |
| **Ours (3-Stream Fusion)** | **99.92%** | **94.25%** | **99.82%** | **90.34%** | **1.27%** |

> All models trained and evaluated on the same face dataset (stratified split, seed=42).

### ROC Curve & Confusion Matrix

<p align="center">
  <img src="assets/figures/roc_curve.png" width="45%" alt="ROC Curve"/>
  &nbsp;&nbsp;
  <img src="assets/figures/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
</p>

- **AUC = 99.92%** — near-perfect discrimination between real and fake faces
- **FN = 1** — only 1 fake image missed out of 553 in the test set
- **FP = 117** — 117 real images flagged as fake (model is cautious, appropriate for this task)

### Prediction Score Distribution

![Prediction Distribution](assets/figures/prediction_dist.png)

Real and fake face scores are well-separated. Most fakes score >0.9, most real images score <0.1.

---

## Cross-Generator Generalization (Key Contribution)

Trained exclusively on **Stable Diffusion v1.x** faces → tested on **SDXL** (never seen during training):

![Cross-Generator Ablation](assets/figures/ablation_crossgen.png)

| Configuration | In-Dist AUC | SDXL AUC | Drop |
|--------------|-------------|----------|------|
| Frequency Only | 68.5% | 58.26% | −10.24% |
| Spatial + Freq | 100% | 95.98% | −4.02% |
| Spatial Only | 100% | 96.44% | −3.56% |
| Semantic Only | 100% | 97.17% | −2.83% |
| Spatial + Semantic | 100% | 93.64% | −6.36% |
| **Full 3-Stream (ours)** | **100%** | **98.09%** | **−1.91%** ✓ |

> **Notable finding:** Spatial + Semantic (−6.36%) performs *worse* than Spatial alone (−3.56%). The frequency stream resolves conflicting signals between the two streams, enabling the full model to generalize best.

---

## GradCAM++ Visualizations

Heatmaps showing which face regions the model focuses on when detecting deepfakes:

<p align="center">
  <img src="assets/heatmaps/heatmap_01.jpg" width="15%" alt="heatmap 1"/>
  <img src="assets/heatmaps/heatmap_02.jpg" width="15%" alt="heatmap 2"/>
  <img src="assets/heatmaps/heatmap_03.jpg" width="15%" alt="heatmap 3"/>
  <img src="assets/heatmaps/heatmap_04.jpg" width="15%" alt="heatmap 4"/>
  <img src="assets/heatmaps/heatmap_05.jpg" width="15%" alt="heatmap 5"/>
  <img src="assets/heatmaps/heatmap_06.jpg" width="15%" alt="heatmap 6"/>
</p>

*All samples: true_label=fake, predicted=fake, confidence=100%. Hot regions (red/yellow) indicate areas with strongest forgery evidence.*

---

## Dataset

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | ~12,000 | ~4,419 | ~16,419 |
| Val | ~1,500 | ~552 | ~2,052 |
| Test | ~1,500 | ~553 | ~2,053 |
| Cross-gen eval (SDXL) | 1,500 | 1,920 | 3,420 |

| Source | Type | Count |
|--------|------|-------|
| FFHQ 256px | Real faces | ~15,000 |
| CelebA-HQ | Real faces | ~3,000 |
| DiffusionDB (face-filtered) | Fake — SD v1.x | 5,524 |
| 8clabs/sdxl-faces | Fake — SDXL (eval only) | 1,920 |

> Split is stratified per-class, single RNG seed=42 — **zero data leakage guaranteed.**

---

## Project Structure

```
deepfake-detection/
├── assets/
│   ├── figures/             # Architecture diagram, result charts, eval plots
│   └── heatmaps/            # GradCAM++ visualizations
├── configs/
│   └── config.yaml          # All hyperparameters (portable DATA_ROOT path)
├── data/
│   └── dataset.py           # DeepfakeDataset — stratified split, augmentation
├── models/
│   ├── spatial_stream.py    # NPRBranch — EfficientNet-B0, 128-dim
│   ├── freq_stream.py       # FreqBlender — ResNet-18 + LearnableFFTMask, 64-dim
│   ├── semantic_stream.py   # FATLiteTransformer — ViT-Tiny, 384-dim
│   ├── fusion.py            # MLAFFusion — Cross-Stream Attention
│   ├── full_model.py        # MultiStreamDeepfakeDetector + ablation modes
│   ├── baselines.py         # CNNDetect + UnivFD
│   └── localization.py      # GradCAM++ heatmap generation
├── scripts/
│   ├── train.py             # Training — AdamW, cosine LR, early stopping
│   ├── evaluate.py          # Evaluation + GradCAM++
│   ├── compare_baselines.py # Side-by-side baseline comparison
│   ├── cross_generator_eval.py  # Cross-generator AUC evaluation
│   ├── filter_faces.py      # OpenCV face detector for DiffusionDB filtering
│   ├── inference.py         # Single-image inference
│   ├── robustness_eval.py   # Robustness under JPEG, blur, noise
│   ├── run_ablations_clean.sh   # Run all 5 ablation training runs
│   └── cross_gen_ablation.sh    # Cross-generator ablation table
├── reports/                 # 10 detailed project reports
├── requirements.txt         # Package requirements
└── requirements-lock.txt    # Pinned versions — CUDA 12.8, PyTorch 2.9.1
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on RTX 5060 Ti 16GB)
- ~5 GB disk space for datasets

### Install

```bash
git clone https://github.com/noushad999/thesis_grp_3.git
cd thesis_grp_3

# Install (pinned versions for full reproducibility)
pip install -r requirements-lock.txt
```

### Data Setup

```bash
export DATA_ROOT=/path/to/data   # or edit configs/config.yaml

# 1. Download datasets
python scripts/download_datasets.py

# 2. Filter DiffusionDB → face images only
python scripts/filter_faces.py \
  --input $DATA_ROOT/fake/diffusiondb \
  --output $DATA_ROOT/fake/diffusiondb_faces

# 3. Create symlinked faces_dataset
#    faces_dataset/real/{ffhq,celebahq}
#    faces_dataset/fake/diffusiondb_faces
```

---

## Training & Evaluation

### Train Full Model

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --data-dir $DATA_ROOT/faces_dataset
# Best checkpoint → checkpoints/best_model.pth (epoch 15)
```

### Evaluate with GradCAM

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data-dir $DATA_ROOT/faces_dataset \
  --output-dir logs/eval \
  --num-heatmaps 20
```

### Compare Baselines

```bash
python scripts/compare_baselines.py \
  --config configs/config.yaml \
  --data-dir $DATA_ROOT/faces_dataset
```

### Cross-Generator Evaluation

```bash
python scripts/cross_generator_eval.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --real-dir $DATA_ROOT/real \
  --fake-dir $DATA_ROOT/fake/sdxl_faces/imgs \
  --generator-name SDXL \
  --max-images 1500
```

### Ablation Study

```bash
bash scripts/run_ablations_clean.sh     # Train all 5 configurations
bash scripts/cross_gen_ablation.sh      # Cross-generator ablation table
```

### Single Image Inference

```bash
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/face.jpg
```

---

## Critical Bug Fixes

| File | Fix Applied | Impact |
|------|------------|--------|
| `data/dataset.py:97` | `os.walk(followlinks=True)` | Without this, symlinked dataset returned 0 images |
| `data/dataset.py:110` | Single-RNG stratified split | Original had data leakage across train/val/test splits |
| `models/baselines.py:77` | `F.interpolate(x, (224,224))` in UnivFD | CLIP ViT-L/14 requires 224×224 — crashed on 256×256 input |
| `configs/config.yaml:51` | `pos_weight: 2.72` (was 1.25) | Corrected for actual 15k:5.5k real:fake class imbalance |
| `configs/config.yaml:7` | `${DATA_ROOT:-data}` (was absolute path) | Portable — works on any machine |
| `models/localization.py:79` | GradCAM++ alpha sum over `axis=(1,2)` | Original per-pixel computation was mathematically wrong |
| `models/fusion.py:22` | 3-token sequence attention | Original `seq_len=1` self-attention is a mathematical no-op |

---

## Reports

The `reports/` directory contains 10 detailed technical reports:

| # | Report | Audience |
|---|--------|----------|
| 01 | Project Overview — complete summary of all work | Everyone |
| 02 | Code Explained — line-by-line documentation in plain language | Non-technical |
| 03 | Figures Explained — every chart and what it means | Everyone |
| 04 | Architecture — full technical model description | Technical |
| 05 | Defense Q&A — 15 thesis viva questions with answers | Student |
| 06 | Dataset & Pipeline — data collection and preprocessing | Technical |
| 07 | Experimental Results — all metrics and tables | Researcher |
| 08 | Baseline Comparison — detailed analysis vs CNNDetect/UnivFD | Researcher |
| 09 | Ablation Study — what each stream contributes | Researcher |
| 10 | Publication Guide — paper structure, ICCIT 2025 target | Student |

---

## References

1. Wang et al. *"CNN-generated images are surprisingly easy to spot."* CVPR 2020.
2. Ojha et al. *"Towards Universal Fake Image Detection by Exploiting CLIP's Potential."* CVPR 2023.
3. Qian et al. *"Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues."* ECCV 2020.
4. Durall et al. *"Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions."* CVPR 2020.
5. Dosovitskiy et al. *"An Image is Worth 16×16 Words."* ICLR 2021.
6. Rombach et al. *"High-Resolution Image Synthesis with Latent Diffusion Models."* CVPR 2022.
7. Podell et al. *"SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis."* arXiv 2023.

---

<div align="center">

**Thesis Group 3** &nbsp;|&nbsp; BSc Thesis &nbsp;|&nbsp; Academic Research

*Lead: Md Noushad Jahan Ramim*

</div>
