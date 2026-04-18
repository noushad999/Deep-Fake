# Multi-Stream Deepfake Face Detection

A deepfake face detection system using multi-stream fusion of spatial, frequency, and semantic analysis — trained on Stable Diffusion v1.x faces and evaluated for cross-generator generalization on unseen SDXL-generated images.

**BSc Thesis Project | Thesis Group 3**

---

## Results at a Glance

| Metric | Our Model | CNNDetect (Wang CVPR'20) | UnivFD (Ojha CVPR'23) |
|--------|-----------|--------------------------|------------------------|
| AUC-ROC | **99.92%** | ~85.2% | ~91.4% |
| Accuracy | **94.25%** | ~79.8% | ~86.7% |
| Recall (fake) | **99.82%** | ~78.3% | ~85.4% |
| F1 Score | **90.34%** | ~77.1% | ~84.3% |
| EER | **1.27%** | ~15.2% | ~9.1% |

### Cross-Generator Generalization (Key Finding)
Trained on SD v1.x → Tested on **unseen SDXL** generator:

| Configuration | SDXL AUC | Generalization Drop |
|--------------|----------|---------------------|
| Frequency Only | 58.26% | −10.24% |
| Spatial + Freq | 95.98% | −4.02% |
| Spatial Only | 96.44% | −3.56% |
| Semantic Only | 97.17% | −2.83% |
| Spatial + Semantic | 93.64% | −6.36% |
| **Full 3-Stream (ours)** | **98.09%** | **−1.91% (best)** |

> The full 3-stream model achieves the smallest generalization drop, demonstrating that multi-stream fusion learns more universal forgery cues than any single stream or two-stream combination.

---

## Architecture

```
Input Face Image (3 × 256 × 256)
         │
    ┌────┴──────────────────────────────┐
    │                                   │
    ▼              ▼                    ▼
Stream 1        Stream 2            Stream 3
(Spatial)      (Frequency)         (Semantic)
EfficientNet-B0  ResNet-18          ViT-Tiny
+ Learnable FFT  patch16_224
↓ 128-dim        ↓ 64-dim           ↓ 384-dim
    │                │                   │
    └────────┬────────┘                   │
             │   Project all → 256-dim    │
             └──────────────┬─────────────┘
                            │
               Cross-Stream Multi-Head Attention
               (4 heads, 256-dim hidden, MLAF)
                            │
                   Linear(256→128→1)
                            │
                    Sigmoid → [0, 1]
                            │
                       Real / Fake
```

### Stream Details

| Stream | Backbone | Output | What it detects |
|--------|----------|--------|-----------------|
| **Spatial (NPR)** | EfficientNet-B0 | 128-dim | Pixel-level texture artifacts, boundary inconsistencies |
| **Frequency (FreqBlender)** | ResNet-18 + Learnable FFT | 64-dim | Spectral artifacts, periodic noise patterns |
| **Semantic (FAT-Lite)** | ViT-Tiny patch16 | 384-dim | Global structural inconsistencies, face region relationships |

**Total Parameters:** ~22.9M | **Best Epoch:** 15 | **Hardware:** RTX 5060 Ti 16GB

---

## Project Structure

```
deepfake-detection/
├── configs/
│   └── config.yaml              # All hyperparameters and settings
├── data/
│   ├── dataset.py               # DeepfakeDataset with stratified split (no leakage)
│   └── __init__.py
├── models/
│   ├── spatial_stream.py        # NPRBranch (EfficientNet-B0)
│   ├── freq_stream.py           # FreqBlender (ResNet-18 + LearnableFFTMask)
│   ├── semantic_stream.py       # FATLiteTransformer (ViT-Tiny)
│   ├── fusion.py                # MLAFFusion (Cross-Stream Attention)
│   ├── full_model.py            # MultiStreamDeepfakeDetector
│   ├── baselines.py             # CNNDetect + UnivFD baselines
│   └── localization.py          # GradCAM++ heatmap generation
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation + GradCAM
│   ├── compare_baselines.py     # Baseline comparison
│   ├── cross_generator_eval.py  # Cross-generator generalization test
│   ├── filter_faces.py          # Face detection filter for DiffusionDB
│   ├── inference.py             # Single image inference
│   ├── robustness_eval.py       # Robustness under distortions
│   ├── run_ablations_clean.sh   # Run all 5 ablation configurations
│   └── cross_gen_ablation.sh    # Cross-generator ablation study
├── reports/                     # 10 professional project reports
│   ├── report_01_project_overview.md
│   ├── report_02_code_explained.md
│   ├── report_03_figures_explained.md
│   ├── report_04_architecture.md
│   ├── report_05_defense_qa.md
│   ├── report_06_dataset_pipeline.md
│   ├── report_07_experimental_results.md
│   ├── report_08_baseline_comparison.md
│   ├── report_09_ablation_study.md
│   └── report_10_publication_guide.md
├── utils/
│   └── utils.py                 # Seeding, checkpointing, device helpers
├── requirements.txt             # Package requirements
└── requirements-lock.txt        # Pinned versions (CUDA 12.8, PyTorch 2.9.1)
```

---

## Dataset

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | ~12,000 | ~4,419 | ~16,419 |
| Val | ~1,500 | ~552 | ~2,052 |
| Test | ~1,500 | ~553 | ~2,053 |
| SDXL eval (cross-gen) | 1,500 | 1,920 | 3,420 |

**Real images:** FFHQ 256px + CelebA-HQ (HuggingFace)
**Fake images:** DiffusionDB (SD v1.x) — face-filtered using OpenCV ResNet-SSD detector
**Cross-generator test:** 8clabs/sdxl-faces (HuggingFace)

> Dataset split is stratified (per-class), single RNG seed=42. No data leakage.

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (tested: RTX 5060 Ti 16GB, CUDA 12.8)
- ~5GB disk space for datasets

### Installation

```bash
git clone https://github.com/noushad999/thesis_grp_3.git
cd thesis_grp_3

pip install -r requirements.txt
# For exact reproducibility:
pip install -r requirements-lock.txt
```

### Environment

```bash
export DATA_ROOT=/path/to/your/data   # or edit configs/config.yaml
```

---

## Usage

### 1. Prepare Dataset

```bash
# Download datasets
python scripts/download_datasets.py

# Filter DiffusionDB to face images only
python scripts/filter_faces.py \
  --input data/fake/diffusiondb \
  --output data/fake/diffusiondb_faces

# Build symlinked faces_dataset
# Structure: faces_dataset/real/{ffhq,celebahq} and faces_dataset/fake/diffusiondb_faces
```

### 2. Train

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --data-dir data/faces_dataset
```

### 3. Evaluate with GradCAM

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data-dir data/faces_dataset \
  --output-dir logs/eval \
  --num-heatmaps 20
```

### 4. Compare Baselines

```bash
python scripts/compare_baselines.py \
  --config configs/config.yaml \
  --data-dir data/faces_dataset \
  --output logs/comparison.log
```

### 5. Cross-Generator Evaluation

```bash
python scripts/cross_generator_eval.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --real-dir data/real \
  --fake-dir data/fake/sdxl_faces/imgs \
  --generator-name SDXL \
  --max-images 1500
```

### 6. Run Ablation Study

```bash
bash scripts/run_ablations_clean.sh        # Train all 5 ablation configs
bash scripts/cross_gen_ablation.sh         # Cross-generator ablation table
```

### 7. Single Image Inference

```bash
python scripts/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --image path/to/face.jpg
```

---

## Key Engineering Decisions & Fixes

| File | Fix | Why |
|------|-----|-----|
| `data/dataset.py` | `os.walk(followlinks=True)` | Symlinked dataset returned 0 images without this |
| `data/dataset.py` | Stratified split with single RNG | Original code had data leakage across splits |
| `models/baselines.py` | `F.interpolate(x, (224,224))` in UnivFD | CLIP ViT-L/14 positional embeddings require 224×224 |
| `configs/config.yaml` | `pos_weight: 2.72` | Corrected for 15k:5.5k real:fake ratio |
| `configs/config.yaml` | `${DATA_ROOT:-data}` | Portable path — works on any machine |
| `models/localization.py` | GradCAM++ spatial sum over `axis=(1,2)` | Original per-pixel computation was mathematically wrong |
| `models/fusion.py` | 3-token sequence attention | Original `seq_len=1` attention is a no-op |

---

## Reports

The `reports/` directory contains 10 detailed project reports:

| Report | Contents |
|--------|----------|
| `report_01_project_overview.md` | Complete project summary |
| `report_02_code_explained.md` | Line-by-line code documentation |
| `report_03_figures_explained.md` | All charts and visualizations explained |
| `report_04_architecture.md` | Full technical architecture |
| `report_05_defense_qa.md` | 15 thesis defense Q&A |
| `report_06_dataset_pipeline.md` | Data collection and preprocessing |
| `report_07_experimental_results.md` | All metrics and tables |
| `report_08_baseline_comparison.md` | Detailed baseline analysis |
| `report_09_ablation_study.md` | Stream contribution analysis |
| `report_10_publication_guide.md` | Paper writing guide + ICCIT 2025 target |

---

## References

1. Wang et al. "CNN-generated images are surprisingly easy to spot." CVPR 2020.
2. Ojha et al. "Towards Universal Fake Image Detection by Exploiting CLIP's Potential." CVPR 2023.
3. Qian et al. "Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues." ECCV 2020.
4. Durall et al. "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions." CVPR 2020.
5. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
6. Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.

---

## Team

**Thesis Group 3** | BSc Thesis Project

**Lead:** Md Noushad Jahan Ramim

---

## License

Academic research use only.
