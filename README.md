# SEMICON KLA 2026: AI Semiconductor Image Restoration System

An AI-based multi-head restoration framework designed for degraded semiconductor wafer inspection images (SEM and optical inspection).

---

## 🏛️ Architecture Overview

The system utilizes a locked multi-head NAFNet-style gated CNN architecture:
- **Shared Backbone**: NAFNet Gated CNN Encoder-Decoder.
- **Head 1**: Restored Image Output ($\hat{I}_{restored}$).
- **Head 2**: Degradation Parameter & Strength Estimation ($\hat{\theta}_{deg}$).
- **Head 3**: Per-Pixel Confidence/Uncertainty Map ($\hat{U}_{pixel}$).

```
Input Image ---> [ NAFNet Backbone ] ---> Head 1: Restored Image
                                    ---> Head 2: Degradation Estimation
                                    ---> Head 3: Uncertainty Map
```

---

## 📊 Dataset Preparation & Pipeline Setup

### 1. Dataset Directory Layout
Place or configure your `.npy` or image files according to the structure:
```
data/
├── train/
│   ├── GT/         # High-Resolution clean images (256x256 .npy)
│   └── NoisyLR/    # Degraded Low-Resolution images (128x128 .npy)
└── test/
    └── NoisyLR/    # Test degraded images (128x128 .npy)
```

Configure local dataset paths in `configs/default_config.yaml`:
```yaml
data:
  mode: "paired"
  train_gt_dir: "/path/to/train/GT"
  train_noisy_dir: "/path/to/train/NoisyLR"
  test_noisy_dir: "/path/to/test/NoisyLR"
  val_ratio: 0.2
```

### 2. Validate Dataset Integrity
Run the dataset validation tool to verify shapes, min/max/mean/std ranges, split sizes, and check for corrupted files:
```bash
py scripts/validate_dataset.py --config configs/default_config.yaml
```

---

## 🔬 Baseline Reproduction

Phase 2 establishes two reproducible baselines evaluated on the deterministic 640-sample validation split (`seed: 42`):
1. **Baseline 1 — Bicubic**: 2x Bicubic Interpolation
2. **Baseline 2 — Simple DnCNN**: 7-Layer Conv+BatchNorm+ReLU Residual Denoising CNN (186,177 parameters)

### 1. Train DnCNN Baseline from Scratch
Train the simple DnCNN baseline on the real paired dataset:
```bash
py -u baseline/train_dncnn.py --epochs 3 --batch_size 16 --save_path checkpoints/dncnn_baseline.pth --seed 42
```

### 2. Evaluate & Benchmark Baselines
Evaluate both Bicubic and DnCNN models on the 640 validation samples, measuring PSNR, SSIM, LPIPS, and inference runtime:
```bash
py -u baseline/evaluate_baseline.py --config configs/default_config.yaml --dncnn_weights checkpoints/dncnn_baseline.pth --output_json results/baseline_comparison.json --output_img_dir results/baseline
```

### 3. Run Baseline Unit Tests
Execute the baseline unit test suite:
```bash
py -m unittest tests/test_baselines.py
```

---

## 🚀 Timed Inference

The mandatory evaluation script is `inference.py`. It accepts input and output directories and operates device-agnostically:

```bash
py inference.py --input_dir ./path/to/inputs --output_dir ./path/to/outputs
```

---

## 📄 Documentation & Experiment Records

- **Architecture Specification**: [docs/architecture.md](docs/architecture.md)
- **Implementation Status & TODOs**: [docs/implementation_status.md](docs/implementation_status.md)
- **Baseline Experiment Records**: [docs/experiments.md](docs/experiments.md)
