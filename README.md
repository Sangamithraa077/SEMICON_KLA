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

## 🚀 Quick Start & Timed Inference

The mandatory evaluation script is `inference.py`. It accepts input and output directories and operates device-agnostically:

```bash
python inference.py --input_dir ./path/to/inputs --output_dir ./path/to/outputs
```

### Options:
- `--input_dir`: Path to folder containing input degraded images (Required).
- `--output_dir`: Path to folder where restored images will be saved (Required).
- `--config`: Path to custom YAML configuration file (Default: `configs/default_config.yaml`).
- `--weights`: Path to trained model weights checkpoint.
- `--device`: Target compute device (`auto`, `cuda`, `cpu`, `mps`).

---

## 📄 Documentation

- **Architecture Specification**: [docs/architecture.md](docs/architecture.md)
- **Implementation Status & TODOs**: [docs/implementation_status.md](docs/implementation_status.md)

---

## 📁 Repository Structure

```
SEMICON_KLA/
├── configs/
│   └── default_config.yaml       # Path-agnostic system parameters
├── docs/
│   ├── architecture.md           # Full system architecture documentation
│   └── implementation_status.md  # Implementation audit and phased roadmap
├── src/
│   ├── models/                   # Backbone, multi-head network, and baselines
│   ├── degradation/              # Poisson-Gaussian noise & synthetic pipeline
│   ├── losses/                   # Multi-component loss objectives
│   ├── evaluation/               # Metrics calculation & offline report agent
│   └── utils/                    # Config parsing and image I/O
├── scripts/
│   ├── train.py                  # Training pipeline entry point
│   ├── demo.py                   # Visual quad-display demo script
│   └── evaluate_offline.py       # Offline evaluation and report generation script
├── inference.py                  # Mandatory timed inference script (--input_dir, --output_dir)
├── requirements.txt              # System dependencies
└── README.md
```
