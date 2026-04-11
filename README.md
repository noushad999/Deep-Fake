# Multi-Stream Deepfake Detection

A lightweight and generalizable deepfake detection system using multi-stream fusion approach.

## Project Structure
```
deepfake-detection/
├── configs/          # Configuration files
├── data/             # Dataset loading utilities
├── models/           # Model architectures
│   ├── spatial_stream.py    # NPR Branch (128-dim)
│   ├── freq_stream.py       # FreqBlender (64-dim)
│   ├── semantic_stream.py   # FAT-Lite (384-dim)
│   ├── fusion.py            # MLAF Fusion
│   ├── localization.py      # GradCAM++
│   └── full_model.py        # Complete model
├── utils/            # Utility functions
├── scripts/          # Training and evaluation scripts
├── checkpoints/      # Saved models
├── logs/             # Training logs and heatmaps
└── notebooks/        # Jupyter notebooks
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup (Phase 1 - Start Small)
We'll start with **CIFake** (~10GB) + **MS-COCO subset** (~5GB):

```bash
# Run dataset download script (to be created)
python scripts/download_datasets.py
```

### 3. Train
```bash
python scripts/train.py
```

## Model Architecture

| Stream | Purpose | Output Dim |
|--------|---------|------------|
| **Spatial (NPR)** | GAN boundary artifacts | 128 |
| **Frequency (FreqBlender)** | FFT-based anomalies | 64 |
| **Semantic (FAT-Lite)** | Global inconsistencies | 384 |

**Total Parameters:** ~24-25M  
**Model Size:** ~95 MB

## Next Steps
- [ ] Download datasets
- [ ] Run training
- [ ] Evaluate on multiple domains
- [ ] Generate GradCAM++ heatmaps
