"""Multi-Head Semiconductor Image Restoration Network.

Architecture:
- Shared Backbone: NAFNet Gated CNN Encoder-Decoder
- Head 1: Restored clean image (C, H, W)
- Head 2: Degradation parameters & strength estimation (4-dim vector: poisson_scale, gaussian_std, blur, scale)
- Head 3: Per-pixel uncertainty/confidence map (1, H, W)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .backbone import NAFNetBackbone

class MultiHeadRestorationNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, width: int = 64):
        super().__init__()
        # Shared Encoder-Decoder Backbone
        self.backbone = NAFNetBackbone(in_channels=in_channels, width=width)
        
        # Head 1: Image Restoration Head
        self.head1_restoration = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Head 2: Degradation Parameter Estimation Head (Poisson scale, Gaussian std, Blur sigma, Scale factor)
        self.head2_degradation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, width // 2),
            nn.ReLU(inplace=True),
            nn.Linear(width // 2, 4)
        )
        
        # Head 3: Per-Pixel Uncertainty Map Head
        self.head3_uncertainty = nn.Sequential(
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, 1, 3, padding=1),
            nn.Softplus() # Ensures strictly positive uncertainty bounds
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head architecture.
        
        Args:
            x: Degraded image tensor (B, C, H, W)
            
        Returns:
            Tuple of:
            - restored_image: (B, C, H, W)
            - degradation_params: (B, 4)
            - uncertainty_map: (B, 1, H, W)
        """
        features = self.backbone(x)
        
        restored = self.head1_restoration(features)
        deg_params = self.head2_degradation(features)
        uncertainty = self.head3_uncertainty(features)
        
        return restored, deg_params, uncertainty
