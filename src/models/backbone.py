"""Lightweight NAFNet-Style Gated CNN Encoder-Decoder Backbone.

Convolution-based architecture (No Transformers, Self-Attention, Diffusion, or GANs).
Components:
- SimpleGate (Channel splitting and elementwise product)
- SCA (Simplified Channel Attention)
- NAFBlock (LayerNorm, Depthwise Conv, SimpleGate, SCA, Pointwise Conv, Residual connection)
- NAFNetBackbone (Multi-stage Encoder-Decoder returning shared decoder features [B, C, 128, 128])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class LayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for 2D Spatial Tensors (B, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class SimpleGate(nn.Module):
    """SimpleGate Non-linear Activation Free Gating Mechanism.
    
    Splits input channels 2C into two equal halves (C, C) and computes elementwise product:
    y = x1 * x2
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """Simplified Channel Attention (SCA).
    
    Computes global average pooling per channel and applies channel weighting:
    y = x * Conv1x1(GlobalAvgPool(x))
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pool(x)
        attn = self.conv(attn)
        return x * attn


class NAFBlock(nn.Module):
    """Non-linear Activation Free Residual Block.
    
    Pipeline:
    x -> LayerNorm -> Conv1x1(C -> 2C) -> Depthwise3x3 -> SimpleGate -> SCA -> Conv1x1(C -> C) -> + x
    """

    def __init__(self, channels: int, dw_expand: int = 2, drop_rate: float = 0.0):
        super().__init__()
        dw_channels = channels * dw_expand

        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, stride=1, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = SCA(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.sg(out)
        out = self.sca(out)
        out = self.conv3(out)
        out = self.drop(out)
        return residual + out


class NAFNetBackbone(nn.Module):
    """Lightweight NAFNet-Style Encoder-Decoder Backbone.
    
    Args:
        in_channels: Input channels (default 1 for grayscale)
        encoder_channels: Channel width at each stage (e.g. [32, 64, 128])
        num_blocks: Number of NAFBlocks at each stage (e.g. [2, 2, 4])
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: List[int] = [32, 64, 128],
        num_blocks: List[int] = [2, 2, 4]
    ):
        super().__init__()
        c0, c1, c2 = encoder_channels
        b0, b1, b2 = num_blocks

        # 1. Input Stem
        self.intro = nn.Conv2d(in_channels, c0, kernel_size=3, stride=1, padding=1)

        # 2. Encoder Level 1 (128x128)
        self.enc1 = nn.Sequential(*[NAFBlock(c0) for _ in range(b0)])
        self.down1 = nn.Conv2d(c0, c1, kernel_size=2, stride=2, padding=0) # 128x128 -> 64x64

        # 3. Encoder Level 2 (64x64)
        self.enc2 = nn.Sequential(*[NAFBlock(c1) for _ in range(b1)])
        self.down2 = nn.Conv2d(c1, c2, kernel_size=2, stride=2, padding=0) # 64x64 -> 32x32

        # 4. Bottleneck Level 3 (32x32)
        self.bottleneck = nn.Sequential(*[NAFBlock(c2) for _ in range(b2)])

        # 5. Decoder Level 2 (32x32 -> 64x64)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, padding=0)
        self.fuse2 = nn.Conv2d(c1 * 2, c1, kernel_size=1, stride=1, padding=0)
        self.dec2 = nn.Sequential(*[NAFBlock(c1) for _ in range(b1)])

        # 6. Decoder Level 1 (64x64 -> 128x128)
        self.up1 = nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2, padding=0)
        self.fuse1 = nn.Conv2d(c0 * 2, c0, kernel_size=1, stride=1, padding=0)
        self.dec1 = nn.Sequential(*[NAFBlock(c0) for _ in range(b0)])

        self.out_channels = c0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_channels, 128, 128]
            
        Returns:
            Shared decoder feature tensor [B, out_channels, 128, 128]
        """
        # Intro
        feat_intro = self.intro(x) # [B, c0, 128, 128]

        # Encoder 1
        feat_enc1 = self.enc1(feat_intro) # [B, c0, 128, 128]
        feat_down1 = self.down1(feat_enc1) # [B, c1, 64, 64]

        # Encoder 2
        feat_enc2 = self.enc2(feat_down1) # [B, c1, 64, 64]
        feat_down2 = self.down2(feat_enc2) # [B, c2, 32, 32]

        # Bottleneck
        feat_btn = self.bottleneck(feat_down2) # [B, c2, 32, 32]

        # Decoder 2
        feat_up2 = self.up2(feat_btn) # [B, c1, 64, 64]
        feat_cat2 = torch.cat([feat_up2, feat_enc2], dim=1) # [B, c1*2, 64, 64]
        feat_dec2 = self.dec2(self.fuse2(feat_cat2)) # [B, c1, 64, 64]

        # Decoder 1
        feat_up1 = self.up1(feat_dec2) # [B, c0, 128, 128]
        feat_cat1 = torch.cat([feat_up1, feat_enc1], dim=1) # [B, c0*2, 128, 128]
        feat_dec1 = self.dec1(self.fuse1(feat_cat1)) # [B, c0, 128, 128]

        return feat_dec1
