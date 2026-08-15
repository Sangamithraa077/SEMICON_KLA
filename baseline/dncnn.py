"""Simple DnCNN Denoising Baseline Architecture (Baseline 2).

Flow:
128x128 NoisyLR
     ↓
2x Bicubic Upsampling (256x256)
     ↓
7-Layer DnCNN Residual Network
     ↓
256x256 Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DnCNNBaseline(nn.Module):
    """Simple 7-Layer DnCNN Baseline for semiconductor image restoration.

    Predicts residual noise map R(x) such that clean restored image = x_upsampled - R(x).
    """
    def __init__(self, in_channels: int = 1, num_features: int = 64, num_layers: int = 7):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        # Layer 1: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # Layers 2 to num_layers - 1: Conv + BatchNorm + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Layer num_layers: Conv
        layers.append(nn.Conv2d(num_features, in_channels, kernel_size=3, padding=1, bias=True))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W) - either 128x128 LR or pre-upsampled 256x256.

        Returns:
            Restored tensor (B, C, 256, 256).
        """
        # If input is 128x128, upsample by 2x via bicubic interpolation
        if x.shape[-2] == 128 and x.shape[-1] == 128:
            x_upsampled = F.interpolate(x, scale_factor=2.0, mode="bicubic", align_corners=False)
        else:
            x_upsampled = x

        # Predict residual noise map
        residual = self.dncnn(x_upsampled)

        # Restored image = upsampled input - predicted noise
        restored = x_upsampled - residual
        return torch.clamp(restored, 0.0, 1.0)

def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
