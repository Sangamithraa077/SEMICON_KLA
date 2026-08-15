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
│   ├── GT/         # High-Resolution clean images (e.g. 256x256 .npy)
│   └── NoisyLR/    # Degraded Low-Resolution images (e.g. 128x128 .npy)
└── test/
    └── NoisyLR/    # Test degraded images (e.g. 128x128 .npy)
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

### 3. Run Dataset Unit Tests
Execute the dataset test suite to verify determinism, transforms, and DataLoader batching:
```bash
py -m unittest tests/test_dataset.py
```

### 4. Visualize Degradation Sample
Generate a side-by-side comparison of a clean sample vs synthetic Poisson-Gaussian degradation:
```bash
py scripts/visualize_sample.py --config configs/default_config.yaml
```

---

## 🚀 Timed Inference

The mandatory evaluation script is `inference.py`. It accepts input and output directories and operates device-agnostically:

```bash
py inference.py --input_dir ./path/to/inputs --output_dir ./path/to/outputs
```

---

## 📄 Documentation

- **Architecture Specification**: [docs/architecture.md](docs/architecture.md)
- **Implementation Status & TODOs**: [docs/implementation_status.md](docs/implementation_status.md)
