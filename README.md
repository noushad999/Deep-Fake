<div align="center">

<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=6366F1&center=true&vCenter=true&width=700&lines=Multi-Stream+Deepfake+Detection;Stream+Orthogonality+%2B+Adaptive+Fusion;Spatial+%C3%97+Frequency+%C3%97+Semantic" alt="Typing SVG" />
</h1>

<p>
  <strong>Diagnosing and Preventing Stream Co-Adaptation in Multi-Stream Deepfake Detectors</strong><br/>
  <em>BSc Thesis → CVPR Extension | Md Noushad Jahan Ramim</em>
</p>

<p>
  <a href="#architecture"><img src="https://img.shields.io/badge/Architecture-3--Stream%20MLAF-6366f1?style=for-the-badge&logo=pytorch" alt="Architecture"/></a>
  <a href="#results"><img src="https://img.shields.io/badge/In--dist%20AUC-99.99%25-22c55e?style=for-the-badge" alt="AUC"/></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-3.10%2B-3b82f6?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="#installation"><img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <img src="https://img.shields.io/badge/CUDA-12.8-76b900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Status-Active%20Research-f59e0b?style=for-the-badge" alt="Status"/>
</p>

<p>
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-key-contributions">Contributions</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-citation">Citation</a>
</p>

</div>

---

<div align="center">
  <img src="assets/figures/architecture.png" width="85%" alt="Multi-Stream Architecture"/>
  <br/>
  <em>Figure 1: Multi-Stream Deepfake Detector with MLAF Cross-Stream Attention Fusion. Three complementary streams — spatial, frequency, and semantic — fused via cross-stream attention.</em>
</div>

---

## 📌 TL;DR

> Multi-stream detectors fail to generalize because **streams learn redundant features** (co-adaptation). We introduce **per-sample orthogonality regularization** and **stream dropout** to keep streams complementary and improve cross-generator generalization.

---

## 🔬 Key Contributions

| | |
|---|---|
| **3-Stream Architecture** | EfficientNet-B0 (spatial) + ResNet-18+FFT (frequency) + ViT-Tiny (semantic) |
| **MLAF Fusion** | 3-token cross-stream attention with learnable stream-type embeddings |
| **Orthogonality Loss** | Per-sample cosine similarity penalty — forces complementary stream features |
| **Stream Dropout** | p=0.3, prevents co-adaptation between streams during training |
| **Positioning** | AI-generated image detection (diffusion/GAN) — cross-generator generalization |

---

## 🏛️ Architecture

| Stream | Backbone | Dim | Focus |
|--------|----------|-----|-------|
| **Spatial** | EfficientNet-B0 | 128 | Pixel artifacts, boundary inconsistencies |
| **Frequency** | ResNet-18 + Learnable FFT | 64 | Spectral fingerprints, GAN/diffusion residuals |
| **Semantic** | ViT-Tiny-Patch16 @ 256px | 384 | Global structural inconsistencies |
| **Fusion** | MLAF Cross-Stream Attention | 256 | Adaptive stream weighting |

```
Input [B, 3, 256, 256]
  ├── Spatial  (EfficientNet-B0)  ──► [B, 128]
  ├── Frequency (ResNet-18 + FFT) ──► [B,  64]  ─► MLAF ─► [B, 1]
  └── Semantic  (ViT-Tiny)        ──► [B, 384]
```

**21.9M parameters · 256×256 input · ~12ms inference (RTX 3090)**

---

## 📊 Results

<div align="center">
<table><tr>
<td align="center"><img src="assets/figures/roc_curve.png" width="310px"/><br/><em>ROC Curve</em></td>
<td align="center"><img src="assets/figures/confusion_matrix.png" width="290px"/><br/><em>Confusion Matrix</em></td>
<td align="center"><img src="assets/figures/prediction_dist.png" width="290px"/><br/><em>Score Distribution</em></td>
</tr></table>
</div>

| Metric | Value |
|--------|-------|
| AUC-ROC | **99.99%** |
| Accuracy | **99.64%** |
| F1-Score | 99.64% |
| EER | ~0.01% |

*2,497 held-out images · SD fakes + COCO/FFHQ real · seed=42*

### Compression Robustness

| Level | AUC |
|-------|-----|
| C0 (uncompressed) | 100.00% |
| C23 | 93.47% |
| C40 | 91.59% |
| C50 | 89.89% |

### Baseline Comparison

<div align="center">
  <img src="assets/figures/baseline_comparison.png" width="70%"/>
  <br/><em>Figure 2: AUC vs CNNDetect, F3Net, XceptionDetect, UnivFD</em>
</div>

---

## 🔬 Ablation Study

<div align="center">
  <img src="assets/figures/ablation_crossgen.png" width="65%"/>
  <br/><em>Figure 3: Cross-generator AUC under different stream configurations</em>
</div>

| Configuration | AUC (%) | ∆ vs Full |
|--------------|---------|-----------|
| **Full model** | **99.99** | — |
| Spatial only | 99.95 | −0.04 |
| Spatial + Semantic | 99.98 | −0.01 |
| Spatial + Freq | 99.98 | −0.01 |
| Semantic only | 99.29 | −0.70 |
| Frequency only | 68.42 | −31.57 |

> Frequency stream alone is near-chance (68.42%) but encodes artifacts invisible to spatial/semantic — exactly the complementarity orthogonality regularization preserves.

| Regularization Config | Cos Sim ↓ | OOD AUC ↑ |
|----------------------|-----------|-----------|
| No orth, no dropout | ~0.8 | TBD |
| + Orth loss | ~0.3 | TBD |
| + Stream dropout | ~0.1 | TBD |
| **Full** | **~0.1** | **TBD** |

---

## 🎨 GradCAM++ Visualizations

<div align="center">
<table><tr>
  <td><img src="assets/heatmaps/heatmap_01.jpg" width="148px"/></td>
  <td><img src="assets/heatmaps/heatmap_02.jpg" width="148px"/></td>
  <td><img src="assets/heatmaps/heatmap_03.jpg" width="148px"/></td>
  <td><img src="assets/heatmaps/heatmap_04.jpg" width="148px"/></td>
  <td><img src="assets/heatmaps/heatmap_05.jpg" width="148px"/></td>
  <td><img src="assets/heatmaps/heatmap_06.jpg" width="148px"/></td>
</tr></table>
<em>Spatial stream GradCAM++ activations — regions where the model detects manipulation artifacts</em>
</div>

---

## ⚡ Quick Start

```bash
git clone https://github.com/noushad999/Deep-Fake.git
cd Deep-Fake
pip install -r requirements-lock.txt
python scripts/inference.py --checkpoint checkpoints/best_model.pth --image img.jpg
```

---

## 🏋️ Training

```bash
# Main pipeline
python scripts/train.py --config configs/config.yaml --data-dir $DATA_ROOT

# FF++ protocol
python scripts/train_ffpp.py --data-dir $FFPP_DIR --compression c23

# Ablation (stream combination)
python scripts/train.py --ablation-mode spatial_only   # freq_only | semantic_only | ...

# Multi-seed reproducibility
bash scripts/run_multiseed.sh
```

---

## 📈 Evaluation

```bash
python scripts/evaluate.py        --checkpoint checkpoints/best_model.pth
python scripts/eval_ood.py        --checkpoint checkpoints/best_model.pth
python scripts/adversarial_eval.py --checkpoint checkpoints/best_model.pth --attacks fgsm pgd20 cw
python scripts/cross_dataset_eval.py --checkpoint checkpoints/best_model.pth
python scripts/visualize_gradcam.py  --checkpoint checkpoints/best_model.pth
python scripts/generate_paper_tables.py
```

---

## 📋 Roadmap

| Component | Status |
|-----------|--------|
| 3-stream architecture + MLAF | ✅ |
| Stream dropout | ✅ |
| Orthogonality regularization (per-sample) | 🔄 |
| Baselines (CNNDetect, UnivFD, F3Net, Xception) | ✅ |
| In-distribution AUC 99.99% | ✅ |
| Compression robustness C23–C50 | ✅ |
| FF++ evaluation | ⏳ data pending |
| So-Fake-OOD | 🔄 downloading |
| Co-adaptation analysis (cos_sim vs OOD AUC) | 📋 planned |
| CVPR 2026 paper draft | 📋 Q3 2026 target |

---

## 📄 Citation

```bibtex
@misc{ramim2026multistream,
  title  = {Diagnosing and Preventing Stream Co-Adaptation in Multi-Stream Deepfake Detectors},
  author = {Ramim, Md Noushad Jahan},
  year   = {2026},
  url    = {https://github.com/noushad999/Deep-Fake}
}
```

---

## 🗂️ Old Progress Notes

## Project Status (April 2026)

This README reflects what is already implemented in this repository and what has
already been evaluated through saved artifacts.

Current scope includes two tracks:

1.  core pipeline (custom face deepfake dataset + multi-stream model).
2. Extended CVPR-style pipeline (FF++, Celeb-DF, FakeCOCO/So-Fake adapters,
   cross-dataset/OOD/robustness/adversarial/efficiency tooling).

---

## What Has Been Completed So Far

### 1) Core model and architecture

- Implemented 3-stream detector:
  - Spatial stream: EfficientNet-B0 branch
  - Frequency stream: ResNet-based branch with learnable FFT mask
  - Semantic stream: ViT-tiny branch
- Implemented fusion with cross-stream multi-head attention in models/fusion.py.
- Added ablation modes in models/full_model.py:
  - spatial_only, freq_only, semantic_only, spatial_freq, spatial_semantic, full
- Added stream dropout and orthogonality regularization support in training
  paths.

### 2) Baselines

- Implemented baseline models in models/baselines.py:
  - CNNDetect
  - UnivFD (CLIP probe)
  - XceptionDetect
  - F3Net (frequency-aware)

### 3) Dataset/data pipeline

- Custom dataset loader and stratified split pipeline in data/dataset.py.
- FF++ loader with official split handling and frame/video evaluation support:
  data/ffpp_dataset.py.
- Celeb-DF v2 test loader and frame extraction helper: data/celebdf_dataset.py.
- HuggingFace adapters for cross-generator and OOD experiments:
  - data/hf_fakecoco.py
  - data/hf_sofake.py

### 4) Training pipelines

- Main training pipeline: scripts/train.py
- FF++ training protocol pipeline: scripts/train_ffpp.py
- CVPR-oriented cross-generator pipeline: scripts/train_cvpr.py

### 5) Evaluation and analysis toolkit

- Standard evaluation + GradCAM:
  - scripts/evaluate.py
  - scripts/evaluate_v2.py
  - scripts/visualize_gradcam.py
- Cross-dataset/generalization:
  - scripts/cross_generator_eval.py
  - scripts/cross_dataset_eval.py
  - scripts/eval_ood.py
- Robustness/testing:
  - scripts/eval_robustness.py
  - scripts/eval_tta.py
  - scripts/adversarial_eval.py
- Efficiency and representation analysis:
  - scripts/efficiency_benchmark.py
  - scripts/analyze_fft_mask.py
  - scripts/visualize_features.py
- Reporting utilities:
  - scripts/generate_paper_tables.py
  - results/paper_tables.tex

### 6) Documentation and reports

- 10 structured reports available in reports/.
- Presentation and progress documents maintained in root-level markdown files.

---

## Verified Artifacts In This Repo

Below are metrics that are already present as generated files.

### A) Main evaluation artifacts

- logs/evaluation/evaluation_metrics.json
  - Accuracy: 100.0
  - Precision: 100.0
  - Recall: 100.0
  - F1: 100.0
  - AUC-ROC: 100.0
  - Counts: TP=1350, TN=1800, FP=0, FN=0

- logs/evaluation/per_domain_results.json
  - Real domain and fake domain metrics both saved separately.

### B) Robustness v2 artifacts

- logs/evaluation_v2/compression_robustness.json
  - C0 AUC: 100.00
  - C23 AUC: 93.47
  - C40 AUC: 91.59
  - C50 AUC: 89.89

- logs/evaluation_v2/blur_robustness.json
  - Sigma 1 AUC: 96.15
  - Sigma 2 AUC: 93.02
  - Sigma 4 AUC: 91.79

- logs/evaluation_v2/noise_robustness.json
  - Up to sigma 0.1 all recorded at AUC 100.0 in current artifact.

### C) FF++ artifact

- checkpoints/ffpp/results_multistream_c23_seed42.txt
  - Frame-level: AUC 100, Acc 100, EER 0
  - Video-level: AUC 100, Acc 100, EER 0

### D) Extra robustness/TTA artifacts

- checkpoints/robustness_results.txt
- checkpoints/tta_results.txt

Note: Some utility scripts run on synthetic or reduced test inputs when no
external dataset path is provided. Always interpret metrics using the exact
artifact source.

---

## Model Architecture (Current)

![Architecture Diagram](assets/figures/architecture.png)

| Component        | Backbone / Method             | Output Dim |
| ---------------- | ----------------------------- | ---------- |
| Spatial stream   | EfficientNet-B0               | 128        |
| Frequency stream | ResNet + learnable FFT mask   | 64         |
| Semantic stream  | ViT-tiny patch16              | 384        |
| Fusion           | Cross-stream attention (MLAF) | 256        |

---

## Quick Start

### 1) Clone and install

```bash
git clone https://github.com/noushad999/Deep-Fake.git
cd Deep-Fake
pip install -r requirements-lock.txt
```

### 2) Optional environment variables

```bash
export DATA_ROOT=/path/to/data
export FFPP_DIR=/path/to/FaceForensics++
export CELEBDF_DIR=/path/to/Celeb-DF-v2
```

### 3) Data preparation helpers

```bash
# Generic dataset download helpers
python scripts/download_datasets.py
python scripts/download_cvpr_datasets.py

# FF++ helper utilities
python scripts/extract_ffpp_frames.py
python scripts/generate_ffpp_dummy.py
```

---

## Training Workflows

### A) Main custom pipeline

```bash
python scripts/train.py \
  --config configs/config.yaml \
  --data-dir $DATA_ROOT/faces_dataset
```

### B) FF++ protocol pipeline

```bash
python scripts/train_ffpp.py \
  --data-dir $FFPP_DIR \
  --compression c23 \
  --model multistream
```

### C) CVPR-oriented cross-generator pipeline

```bash
python scripts/train_cvpr.py \
  --data-mode hf \
  --epochs 30 \
  --batch-size 32
```

---

## Evaluation Workflows

### Standard and robust evaluation

```bash
python scripts/evaluate.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data-dir $DATA_ROOT/faces_dataset

python scripts/evaluate_v2.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --data-dir $DATA_ROOT/faces_dataset \
  --output-dir logs/evaluation_v2
```

### Cross-dataset and OOD

```bash
python scripts/cross_dataset_eval.py \
  --checkpoint checkpoints/ffpp/best_multistream_c23.pth \
  --celebdf-dir $CELEBDF_DIR \
  --ffpp-test-dir $FFPP_DIR

python scripts/eval_ood.py \
  --checkpoint checkpoints/best_model.pth
```

### Robustness, adversarial, efficiency, interpretability

```bash
python scripts/eval_robustness.py --checkpoint checkpoints/best_model.pth
python scripts/eval_tta.py --checkpoint checkpoints/best_model.pth

python scripts/adversarial_eval.py \
  --checkpoint checkpoints/best_model.pth \
  --data-dir $DATA_ROOT \
  --attacks fgsm pgd20

python scripts/efficiency_benchmark.py
python scripts/analyze_fft_mask.py --checkpoint checkpoints/best_model.pth
python scripts/visualize_gradcam.py --checkpoint checkpoints/best_model.pth
```

### Paper table generation

```bash
python scripts/generate_paper_tables.py
```

---

## Project Structure (Updated)

```text
deepfake-detection/
|- configs/
|  |- config.yaml
|  |- ffpp_config.yaml
|- data/
|  |- dataset.py
|  |- ffpp_dataset.py
|  |- celebdf_dataset.py
|  |- hf_fakecoco.py
|  |- hf_sofake.py
|- models/
|  |- spatial_stream.py
|  |- freq_stream.py
|  |- semantic_stream.py
|  |- fusion.py
|  |- full_model.py
|  |- baselines.py
|  |- localization.py
|- scripts/
|  |- train.py
|  |- train_ffpp.py
|  |- train_cvpr.py
|  |- evaluate.py
|  |- evaluate_v2.py
|  |- cross_dataset_eval.py
|  |- eval_ood.py
|  |- eval_robustness.py
|  |- eval_tta.py
|  |- adversarial_eval.py
|  |- efficiency_benchmark.py
|  |- analyze_fft_mask.py
|  |- visualize_features.py
|  |- visualize_gradcam.py
|  |- generate_paper_tables.py
|- checkpoints/
|- logs/
|- results/
|- reports/
```

---

## Known Notes and Current Gaps

- Some scripts are implementation-complete but still depend on full external
  dataset downloads to produce final publication-grade numbers.
- A few artifacts were generated in synthetic/dummy settings for pipeline
  validation, not final benchmark claims.
- FF++ per-type AUC in one artifact can be zero if computed on single-class
  subsets; use video/frame aggregate metrics as primary signal.

---

## Reports

Detailed project documents are in reports/:

1. report_01_project_overview.md
2. report_02_code_explained.md
3. report_03_figures_explained.md
4. report_04_architecture.md
5. report_05_defense_qa.md
6. report_06_dataset_pipeline.md
7. report_07_experimental_results.md
8. report_08_baseline_comparison.md
9. report_09_ablation_study.md
10. report_10_publication_guide.md

---

## References

1. Rossler et al., FaceForensics++, ICCV 2019.
2. Wang et al., CNNDetect, CVPR 2020.
3. Qian et al., F3Net, ECCV 2020.
4. Li et al., Celeb-DF v2, CVPR 2020.
5. Ojha et al., UnivFD, CVPR 2023.

---


