# Multi-Stream Deepfake Detection System

A lightweight and generalizable deepfake detection system using multi-stream fusion approach with spatial, frequency, and semantic analysis.

## Overview

This project implements a novel 3-stream neural network architecture for detecting AI-generated synthetic media (deepfakes). The system combines multiple forensic analysis techniques to achieve robust detection across different types of generative models.

## Architecture

```
                    Input Image (256×256)
                           ↓
    ┌──────────────────────┼──────────────────────┐
    ↓                      ↓                      ↓
Stream 1:              Stream 2:              Stream 3:
Spatial (NPR)          Frequency              Semantic (FAT-Lite)
EfficientNet-B0        (FreqBlender)          ViT-Tiny
↓ 128-dim              ResNet-18              ↓ 384-dim
                       ↓ 64-dim
    └──────────────────────┼──────────────────────┘
                           ↓
                    MLAF Fusion Module
                    (Attention-Based)
                           ↓
                    Final Prediction: Real or Fake
```

### Model Components

| Stream | Purpose | Backbone | Output Dim |
|--------|---------|----------|------------|
| **Spatial (NPR)** | GAN boundary artifacts | EfficientNet-B0 | 128 |
| **Frequency (FreqBlender)** | FFT-based anomalies | ResNet-18 | 64 |
| **Semantic (FAT-Lite)** | Global inconsistencies | ViT-Tiny | 384 |

**Total Parameters:** ~23M  
**Model Size:** ~90 MB

## Project Structure

```
deepfake-detection/
├── configs/              # Configuration files
├── data/                 # Dataset loading utilities
├── data_faces/           # Face dataset (to be populated)
├── data_ffpp_kaggle/     # FaceForensics++ Kaggle version
├── data_micro/           # Micro subset for testing
├── data_quality/         # Quality assessment data
├── data_subset/          # Subset of main dataset
├── data_tiny/            # Tiny subset for quick testing
├── logs/                 # Training logs and visualizations
├── models/               # Model architectures
│   ├── spatial_stream.py    # NPR Branch
│   ├── freq_stream.py       # FreqBlender
│   ├── semantic_stream.py   # FAT-Lite
│   ├── fusion.py            # MLAF Fusion
│   ├── localization.py      # GradCAM++
│   └── full_model.py        # Complete model
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Training and evaluation scripts
│   ├── checkpoints/       # Saved model weights
│   └── logs/              # Training logs
└── utils/                # Utility functions
```

## Key Features

- ✅ **Multi-Stream Fusion**: Combines spatial, frequency, and semantic features
- ✅ **Lightweight Design**: ~23M parameters, suitable for deployment
- ✅ **Robust Evaluation**: Comprehensive testing under various distortions
- ✅ **Interpretable**: GradCAM++ heatmap generation for visual explanations
- ✅ **Modular Architecture**: Easy to extend and modify

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/noushad25/thesis-group-3.git
cd thesis-group-3

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Download Datasets** (requires Kaggle API credentials)
   ```bash
   python scripts/download_datasets.py
   ```

2. **Train the Model**
   ```bash
   python scripts/train.py
   ```

3. **Evaluate the Model**
   ```bash
   python scripts/evaluate.py
   ```

### Advanced Training (Version 2)

Includes frozen backbone, metric learning, and heavy augmentation:

```bash
python scripts/train_v2.py
```

### Robustness Testing

Test model performance under various image distortions:

```bash
python scripts/evaluate_v2.py
```

## Results

### Phase 1 Results (CIFAKE Dataset)

| Metric | Clean | JPEG Q=75 | Blur σ=4 |
|--------|-------|-----------|----------|
| **Accuracy** | 100.0% | 67.8% | 83.0% |
| **F1-Score** | 100.0% | 72.7% | 79.6% |
| **AUC-ROC** | 100.0% | 93.5% | 91.8% |

**Note:** CIFAKE dataset results show 100% accuracy on clean data but significant degradation under compression, indicating dataset-specific learning rather than generalizable forensic features. This is an expected limitation of the initial benchmark dataset.

### Key Findings

1. **Architecture is sound**: 3-stream design mirrors SOTA approaches
2. **Dataset quality matters**: CIFAKE insufficient for real deepfake detection
3. **Robustness testing critical**: Clean accuracy ≠ real-world performance
4. **Need face datasets**: FF++, Celeb-DF, DFDC required for proper evaluation

## Research Context

This project is part of ongoing research into generalizable deepfake detection. Key insights from our literature review:

- Standard benchmarks often overestimate real-world performance (USENIX Security 2024)
- Cross-dataset generalization is the true test of robustness
- Paired real-fake training significantly improves generalization
- Frozen backbone tuning (GenD approach) prevents overfitting

### References

1. Tan et al. "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection." CVPR 2024.
2. GenD Authors. "Deepfake Detection that Generalizes Across Benchmarks." arXiv 2025.
3. Layton et al. "SoK: The Good, The Bad, and The Unbalanced: Measuring Structural Limitations of Deepfake Media Datasets." USENIX Security 2024.

## Current Status

- ✅ Core architecture implemented
- ✅ Initial training complete (CIFAKE)
- ✅ Comprehensive evaluation pipeline
- ✅ Robustness testing framework
- ⚠️ Face dataset download in progress
- ❌ Cross-dataset evaluation pending

## Next Steps

1. Download FaceForensics++ dataset
2. Implement paired real-fake training
3. Cross-dataset generalization testing
4. GradCAM++ visualization generation
5. Model optimization and ablation studies

## Team

**Thesis Group 3**

- Research & Development
- Model Architecture
- Data Pipeline
- Evaluation & Testing

## License

This project is for academic research purposes.

## Acknowledgments

We thank the authors of NPR (CVPR 2024), GenD (arXiv 2025), and other foundational works in deepfake detection research for open-sourcing their methods and inspiring this implementation.

---

**For detailed research findings and honest evaluation, see `Research_Report.md`**
