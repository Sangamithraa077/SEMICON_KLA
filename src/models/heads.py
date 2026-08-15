"""Task-Specific Prediction Heads for Multi-Head Restoration Network.

Includes:
1. RestorationHead: 2x learnable PixelShuffle upsampling restoring [B, 1, 256, 256] image in [0.0, 1.0].
2. DegradationHead: Global Average Pooling + MLP predicting [B, 4] degradation parameters.
3. UncertaintyHead: 2x learnable PixelShuffle upsampling predicting [B, 1, 256, 256] log-variance map clamped in [-10.0, 10.0].
"""

import torch
import torch.nn as nn
from typing import List, Optional


class RestorationHead(nn.Module):
    """Head 1: Image Restoration Head (128x128 -> 256x256x1).
    
    Uses learnable 2x PixelShuffle upsampling to produce high-resolution restored image.
    """

    def __init__(self, in_channels: int, mid_channels: int = 32, scale_factor: int = 2):
        super().__init__()
        # PixelShuffle upsampling: in_channels -> mid_channels * (scale_factor^2)
        self.conv_up = nn.Conv2d(in_channels, mid_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()  # Guarantees output range [0.0, 1.0] matching GT contract
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Shared decoder features [B, in_channels, 128, 128]
            
        Returns:
            Restored image tensor [B, 1, 256, 256]
        """
        out = self.conv_up(x)          # [B, mid_channels * 4, 128, 128]
        out = self.pixel_shuffle(out)   # [B, mid_channels, 256, 256]
        out = self.refine(out)          # [B, 1, 256, 256]
        return out


class DegradationHead(nn.Module):
    """Head 2: Degradation Estimation Head.
    
    Predicts N degradation parameters logged in Phase 1:
    [poisson_scale, gaussian_std, blur_ksize, downsample_scale]
    """

    def __init__(self, in_channels: int, num_params: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_params)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Shared decoder features [B, in_channels, 128, 128]
            
        Returns:
            Predicted degradation parameters [B, num_params]
        """
        feat = self.pool(x).squeeze(-1).squeeze(-1) # [B, in_channels]
        out = self.mlp(feat)                        # [B, num_params]
        return out


class UncertaintyHead(nn.Module):
    """Head 3: Per-Pixel Log-Variance / Uncertainty Head (128x128 -> 256x256x1).
    
    Predicts log(sigma^2) reconstruction log-variance map spatially aligned with restored image.
    Higher values = higher uncertainty; Lower values = higher confidence.
    Log-variance is clamped to [min_log_var, max_log_var] for numerical stability in heteroscedastic loss.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 32,
        scale_factor: int = 2,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0
    ):
        super().__init__()
        self.min_log_variance = min_log_variance
        self.max_log_variance = max_log_variance

        self.conv_up = nn.Conv2d(in_channels, mid_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Shared decoder features [B, in_channels, 128, 128]
            
        Returns:
            Spatially aligned log-variance uncertainty map [B, 1, 256, 256]
        """
        out = self.conv_up(x)          # [B, mid_channels * 4, 128, 128]
        out = self.pixel_shuffle(out)   # [B, mid_channels, 256, 256]
        out = self.refine(out)          # [B, 1, 256, 256]
        
        # Clamp log-variance for numerical stability
        out = torch.clamp(out, min=self.min_log_variance, max=self.max_log_variance)
        return out
