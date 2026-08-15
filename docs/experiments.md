# Baseline Experiment Records & Metrics

## 1. Dataset & Split Specifications

- **Dataset Source**: Paired semiconductor image dataset (`C:\Users\abise\Downloads\train\train`)
- **Total Samples**: 3,200 paired files (`000000.npy` to `003199.npy`)
- **Ground Truth (GT)**: $256 \times 256$ `float32` 1-channel grayscale ($[0.0, 1.0]$)
- **Noisy Low-Resolution (NoisyLR)**: $128 \times 128$ `float32` 1-channel grayscale ($[-0.2786, 2.1580]$)
- **Resolution Scale Factor**: $2\times$ downsampling
- **Split Configuration**:
  - **Random Seed**: `42` (Fixed & deterministic)
  - **Training Count**: 2,560 paired samples (80%)
  - **Validation Count**: 640 paired samples (20%)
  - **Train/Val Overlap**: 0 samples
  - **Test Set**: 400 images (`Test_NoisyLR/NoisyLR`) - **Untouched & Reserved for Final Evaluation**

---

## 2. Measured Baseline Quantitative Comparison

All quantitative metrics were evaluated on the 640 validation samples using seed `42`.

| Model / Baseline | Architecture | Parameter Count | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Model Load Time | Total Inference Time | Avg ms / img | Images / sec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Bicubic** | 2x Bicubic Interpolation | 0 | **22.65** | **0.8793** | **N/A\*** | 0.0 ms | 4.22 s | 6.60 ms | 151.5 img/s |
| **Baseline 2: DnCNN** | 7-Layer Conv+BN+ReLU Residual CNN | 186,177 | **22.80** | **0.8812** | **N/A\*** | 69.54 ms | 124.94 s | 195.21 ms | 5.1 img/s |

> \* **LPIPS Status**: `LPIPS unavailable: lpips package not installed`. (In `baseline/evaluate_baseline.py`, 1-channel grayscale tensors are converted to 3-channel RGB via `pred.repeat(1, 3, 1, 1)` and scaled to $[-1, 1]$ when `lpips` is installed. When uninstalled, LPIPS returns `null` in JSON / `N/A` in tables without returning fake `0.0000` values).

---

## 3. Baseline Training Details (DnCNN)

- **Input Dimension**: $128 \times 128 \times 1$
- **Output Dimension**: $256 \times 256 \times 1$
- **Loss Objective**: $L_1$ Pixel Loss ($\| \hat{I} - I_{clean} \|_1$)
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$)
- **Learning Rate Scheduler**: Cosine Annealing
- **Checkpoint Location**: `checkpoints/dncnn_baseline.pth`
- **Output Image Artifacts**:
  - `results/baseline/bicubic/*.png` (10 representative validation outputs)
  - `results/baseline/dncnn/*.png` (10 representative validation outputs)
  - `results/baseline_comparison.json` (Structured JSON benchmark record)

---

## 4. Hardware & Environment

- **Evaluation Hardware**: CPU (Intel / AMD Multi-core CPU)
- **PyTorch Version**: 2.12.1+cpu
