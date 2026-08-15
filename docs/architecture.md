# System Architecture & Technical Specifications

## 🏛️ Locked Architecture Overview

The KLA SEMICON AI Image Restoration System implements a unified multi-head neural network powered by a shared lightweight NAFNet-style gated CNN encoder-decoder backbone.

```
Input: 128x128 Grayscale NoisyLR
          ↓
┌────────────────────────────────────────────────────────┐
│ Shared NAFNet-Style Gated CNN Encoder-Decoder Backbone │
│ Level 1 (128x128) -> Level 2 (64x64) -> Bottleneck     │
│ -> Level 2 Decoder -> Level 1 Shared Features (128x128)│
└────────────────────────────────────────────────────────┘
          ↓ Shared Features [B, 32, 128, 128]
          ├───────────────────────────────┬───────────────────────────────┐
          ↓                               ↓                               ↓
┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
│  Head 1: Restoration      │   │  Head 2: Degradation Est. │   │  Head 3: Uncertainty      │
│  PixelShuffle 2x          │   │  Global Avg Pool + MLP    │   │  PixelShuffle 2x          │
│  Output: [B, 1, 256, 256] │   │  Output: [B, 4]           │   │  Output: [B, 1, 256, 256] │
│  Range: [0.0, 1.0]        │   │  Parameters:              │   │  Clamped Log-Variance     │
│                           │   │  poisson, gaussian,       │   │  Range: [-10.0, 10.0]     │
│                           │   │  blur, downsample         │   │                           │
└───────────────────────────┘   └───────────────────────────┘   └───────────────────────────┘
```

---

## 🧱 1. Shared Backbone (NAFNet-Style Gated CNN)

The backbone is entirely **convolution-based**, avoiding transformers, self-attention, diffusion models, or GANs to ensure lightweight, real-time H100 GPU and CPU execution.

### Components:
1. **SimpleGate**: Non-linear activation-free gating mechanism. Splits channels $2C \to (C, C)$ and computes elementwise multiplication ($y = x_1 \odot x_2$).
2. **SCA (Simplified Channel Attention)**: Computes global average pooling per channel and applies channel weighting ($y = x \odot \text{Conv}_{1\times 1}(\text{GAP}(x))$).
3. **NAFBlock**:
   - `LayerNorm2d(C)`
   - $1\times 1$ Pointwise Conv ($C \to 2C$)
   - $3\times 3$ Depthwise Conv ($2C \to 2C$, `groups=2C`)
   - `SimpleGate` ($2C \to C$)
   - `SCA` ($C \to C$)
   - $1\times 1$ Pointwise Conv ($C \to C$)
   - Residual Skip Connection ($y = x + \text{Block}(x)$)
4. **Encoder-Decoder Multi-Stage Architecture**:
   - **Input Stem**: $3\times 3$ Conv ($1 \to 32$)
   - **Encoder Stage 1**: 2 NAFBlocks ($32 \times 128 \times 128$) $\to$ $2\times 2$ Conv Stride 2 ($32 \to 64$)
   - **Encoder Stage 2**: 2 NAFBlocks ($64 \times 64 \times 64$) $\to$ $2\times 2$ Conv Stride 2 ($64 \to 128$)
   - **Bottleneck Stage 3**: 4 NAFBlocks ($128 \times 32 \times 32$)
   - **Decoder Stage 2**: Transposed Conv $2\times$ ($128 \to 64$) + Concat Skip + 2 NAFBlocks ($64 \times 64 \times 64$)
   - **Decoder Stage 1**: Transposed Conv $2\times$ ($64 \to 32$) + Concat Skip + 2 NAFBlocks ($32 \times 128 \times 128$)

---

## 🎯 2. Specialized Task Prediction Heads

### Head 1: Image Restoration Head (`RestorationHead`)
- **Input**: Shared decoder features $[B, 32, 128, 128]$
- **Upsampling**: Learnable $2\times$ PixelShuffle ($32 \to 32 \cdot 2^2 = 128 \to 32$ at $256\times 256$)
- **Refinement**: $3\times 3$ Conv $\to$ GELU $\to 1\times 1$ Conv $\to$ Sigmoid
- **Output**: $[B, 1, 256, 256]$ restored grayscale image in $[0.0, 1.0]$.

### Head 2: Degradation Estimation Head (`DegradationHead`)
- **Input**: Shared decoder features $[B, 32, 128, 128]$
- **Pooling & MLP**: Global Average Pooling $[B, 32]$ $\to$ Linear($32 \to 64$) $\to$ GELU $\to$ Linear($64 \to 4$)
- **Output**: $[B, 4]$ predicting logged degradation parameters (`poisson_scale`, `gaussian_std`, `blur_ksize`, `downsample_scale`).
- **Supervision Note**: Real paired NoisyLR images do NOT contain known ground-truth degradation labels. Degradation loss is configurable and optional, using synthetic degradation samples for supervision in Phase 4.

### Head 3: Per-Pixel Uncertainty Head (`UncertaintyHead`)
- **Input**: Shared decoder features $[B, 32, 128, 128]$
- **Upsampling**: Learnable $2\times$ PixelShuffle ($32 \to 32 \cdot 2^2 = 128 \to 32$ at $256\times 256$)
- **Output Representation**: Spatially aligned log-variance $\log(\sigma^2)$ map $[B, 1, 256, 256]$.
- **Clamping**: Clamped to $[-10.0, 10.0]$ for numerical stability in heteroscedastic uncertainty loss.
- **Interpretation**: Higher log-variance = Higher uncertainty; Lower log-variance = Higher confidence.

---

## 📊 3. Model Size & Memory Footprint

- **Total Trainable Parameters**: **555,142**
- **FP32 Parameter Memory**: **2.12 MB**
- **FP16 Parameter Memory**: **1.06 MB**
- **Single Forward Pass Contract**: Accepts $[B, 1, 128, 128]$ and returns:
  ```python
  {
      "restored": [B, 1, 256, 256],
      "degradation": [B, 4],
      "confidence": [B, 1, 256, 256]
  }
  ```
- **Single Forward Pass Execution**: All three heads share the exact same backbone features in one pass.
