# Architecture Specification - SEMICON KLA AI Image Restoration System

## System Overview
This project implements an AI-based image restoration system tailored for degraded semiconductor wafer inspection images (such as Scanning Electron Microscopy [SEM] and optical inspection images). The architecture is designed to handle signal-dependent Poisson-Gaussian noise, optical/defocus blur, downsampling/resolution loss, and charging artifacts while estimating restoration uncertainty and degradation characteristics.

---

## 1. Locked System Architecture

```
                                    +-----------------------------------------+
                                    |        Input Degraded Image (x)         |
                                    +-----------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |     Shared Encoder-Decoder Backbone     |
                                    |      (NAFNet Gated CNN Architecture)    |
                                    +-----------------------------------------+
                                                         |
                   +-------------------------------------+-------------------------------------+
                   |                                     |                                     |
                   v                                     v                                     v
     +---------------------------+         +---------------------------+         +---------------------------+
     |          HEAD 1           |         |          HEAD 2           |         |          HEAD 3           |
     |   Restored Image Output   |         |   Degradation Parameter   |         |   Per-Pixel Confidence /  |
     |     (\hat{I}_{restored})   |         |      Estimation (\hat{\theta})|        |   Uncertainty Map (\hat{U})|
     +---------------------------+         +---------------------------+         +---------------------------+
```

### 1.1 Shared Backbone (NAFNet Gated CNN)
- **Design Philosophy**: Nonlinear Activation Free (NAFNet) block structure utilizing SimpleGate (element-wise multiplication replacing traditional non-linear activations like ReLU/GELU) and Simplified Channel Attention (SCA).
- **Encoder-Decoder Pipeline**: Multi-scale feature extraction with skip connections to preserve high-frequency semiconductor pattern edges.

### 1.2 Multi-Head Outputs
1. **Head 1 (Restoration Head)**: Predicts the clean high-resolution semiconductor image $\hat{I}_{restored} \in \mathbb{R}^{C \times H \times W}$.
2. **Head 2 (Degradation Head)**: Estimates degradation vector $\hat{\theta}_{deg} = [\sigma_p, \sigma_g, k_{blur}, s_{scale}]$, representing Poisson noise scale, Gaussian noise std, blur kernel parameter, and downsampling factor.
3. **Head 3 (Uncertainty Head)**: Outputs per-pixel uncertainty map $\hat{U} \in \mathbb{R}^{1 \times H \times W}$ (where values correspond to predicted variance/error bound), yielding a per-pixel confidence map $\hat{C} = 1 / (1 + \hat{U})$.

---

## 2. Domain-Informed Synthetic Degradation Pipeline

The semiconductor synthetic degradation model accurately simulates physics of SEM imaging and optical inspection:
- **Poisson-Gaussian Noise**: 
  $$y = x + \sqrt{\sigma_g^2 + \sigma_p^2 x} \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
  where $\sigma_p$ represents photon shot noise (Poisson) and $\sigma_g$ represents electronic read noise (Gaussian).
- **Optical / SEM Blur**: Anisotropic Gaussian kernels and defocus modulation.
- **Resolution Loss**: Signal-dependent downsampling (bicubic/bilinear).
- **Parameter Logging**: Every synthetically generated sample logs exact degradation parameters $\theta_{deg}$, providing ground-truth pseudo-labels for Head 2 training.

---

## 3. Training Objective & Loss Functions

The multi-task combined loss objective is defined as:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_1 + \lambda_{ssim} \mathcal{L}_{SSIM} + \lambda_{lpips} \mathcal{L}_{LPIPS} + \lambda_{grad} \mathcal{L}_{grad} + \lambda_{fft} \mathcal{L}_{fft} + \lambda_{deg} \mathcal{L}_{deg} + \lambda_{unc} \mathcal{L}_{unc}$$

- **$\mathcal{L}_1$ (Pixel Loss)**: $\| \hat{I}_{restored} - I_{clean} \|_1$
- **$\mathcal{L}_{SSIM}$ (Structural Loss)**: $1 - \text{SSIM}(\hat{I}_{restored}, I_{clean})$
- **$\mathcal{L}_{LPIPS}$ (Perceptual Loss)**: VGG/AlexNet feature space distance.
- **$\mathcal{L}_{grad}$ (Edge Loss)**: Gradient map magnitude error ($\|\nabla \hat{I} - \nabla I\|_1$) for sharp wafer edges.
- **$\mathcal{L}_{fft}$ (Frequency Loss)**: High-frequency spectrum error ($\|\mathcal{F}(\hat{I}) - \mathcal{F}(I)\|_1$).
- **$\mathcal{L}_{deg}$ (Degradation Loss)**: MSE loss between predicted degradation parameters $\hat{\theta}_{deg}$ and logged pseudo-labels $\theta_{deg}$.
- **$\mathcal{L}_{unc}$ (Uncertainty Loss)**: Heteroscedastic negative log-likelihood loss:
  $$\mathcal{L}_{unc} = \frac{\|\hat{I}_{restored} - I_{clean}\|^2}{2\hat{U}} + \frac{1}{2}\log \hat{U}$$

---

## 4. Pipeline Specifications

### A. Training Pipeline
- **Input**: Clean high-resolution semiconductor dataset.
- **On-the-fly Data Augmentation**: Domain-informed synthetic degradation generator applies Poisson-Gaussian noise, blur, downsampling, and returns $(x_{degraded}, I_{clean}, \theta_{deg})$.
- **Forward Pass**: Multi-head model processes $x_{degraded}$ to compute $(\hat{I}_{restored}, \hat{\theta}_{deg}, \hat{U})$.
- **Loss Computation**: Total combined multi-task loss is computed and backpropagated.
- **Optimization**: AdamW optimizer with cosine annealing learning rate scheduler.
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp` FP16 / BF16) enabled for high-throughput training on NVIDIA H100 / A100 GPUs.
- **Checkpointing**: Validates PSNR/SSIM on validation set and saves best model weights to `checkpoints/best_model.pth`.

### B. Timed Inference Pipeline (`inference.py`)
- **Strict Constraint**: Must be fast, self-contained, and perform ZERO heavy metric calculations or report generations.
- **Command Line Signature**:
  ```bash
  python inference.py --input_dir <path_to_inputs> --output_dir <path_to_outputs> [--config <path>] [--weights <path>] [--device <auto|cuda|cpu>]
  ```
- **Execution Flow**:
  1. Parse input/output directories and load configuration.
  2. Initialize model architecture and load trained checkpoint weights.
  3. Scan `--input_dir` for supported image formats (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).
  4. Process images in optimized mini-batches (or image-by-image for large tiles).
  5. Run forward pass to generate $\hat{I}_{restored}$.
  6. Save restored images directly into `--output_dir` maintaining original filenames.
- **Portability**: Path-agnostic, works on Windows/Linux/macOS, auto-detects GPU (H100/CUDA) or falls back to CPU cleanly.

### C. Offline Evaluation & Reporting Pipeline (`scripts/evaluate_offline.py`)
- **Strict Isolation**: Runs strictly offline AFTER inference completes; NEVER runs during timed inference benchmarks.
- **Metrics Computation**: Full quantitative evaluation calculating PSNR, SSIM, LPIPS, and per-pixel uncertainty stats across evaluation sets.
- **Baseline Comparison**: Compares multi-head network against mandatory baselines (Bicubic upsampling, DnCNN/BM3D).
- **Report Generation**:
  - Automatically mines representative success cases (low uncertainty, high PSNR gain) and failure cases (high uncertainty, residual artifacts).
  - Produces structured Markdown / HTML failure-analysis reports (`docs/evaluation_report.md`).

### D. Demo Pipeline (`scripts/demo.py`)
- **Interactive Workbench**: Command-line or UI script for side-by-side visualization.
- **Visual Quad-Display**:
  1. Input Degraded Image ($x$)
  2. Baseline Result (Bicubic / Denoised)
  3. Multi-Head Restored Image ($\hat{I}_{restored}$)
  4. Per-Pixel Confidence/Uncertainty Heatmap ($\hat{C} / \hat{U}$) alongside estimated degradation vector $\hat{\theta}_{deg}$.

---

## 5. Hardware & Portability Standards

- **Target Compute**: NVIDIA H100 GPU (supports BF16 tensor cores and high batch throughput).
- **Fallback Compute**: Standard NVIDIA GPUs (CUDA), Apple Silicon (MPS), and CPU fallback.
- **Configurability**: Zero hardcoded local absolute paths (such as `C:\Users\...`). All paths are specified via relative configs or CLI arguments.
