"""Multi-Head Restoration Network Entry Point.

Unified model combining shared NAFNetBackbone with:
- RestorationHead (Head 1: [B, 1, 256, 256])
- DegradationHead (Head 2: [B, 4])
- UncertaintyHead (Head 3: [B, 1, 256, 256] log-variance uncertainty map)

Performs a single backbone forward pass for fast, joint multi-task inference.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import os
import yaml

from .backbone import NAFNetBackbone
from .heads import RestorationHead, DegradationHead, UncertaintyHead


class MultiHeadRestorationNet(nn.Module):
    """Locked KLA Multi-Head AI Restoration Network.
    
    Args:
        in_channels: Input image channels (default 1 for grayscale)
        encoder_channels: Encoder/decoder channel widths per stage (e.g. [32, 64, 128])
        num_blocks: Number of NAFBlocks per stage (e.g. [2, 2, 4])
        scale_factor: Spatial super-resolution scale factor (default 2)
        num_degradation_params: Number of logged degradation parameters (default 4)
        min_log_variance: Minimum log-variance clamp boundary (default -10.0)
        max_log_variance: Maximum log-variance clamp boundary (default 10.0)
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: List[int] = [32, 64, 128],
        num_blocks: List[int] = [2, 2, 4],
        scale_factor: int = 2,
        num_degradation_params: int = 4,
        min_log_variance: float = -10.0,
        max_log_variance: float = 10.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.num_blocks = num_blocks
        self.scale_factor = scale_factor
        self.num_degradation_params = num_degradation_params

        # 1. Shared NAFNet Gated CNN Encoder-Decoder Backbone
        self.backbone = NAFNetBackbone(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            num_blocks=num_blocks
        )
        shared_dim = self.backbone.out_channels

        # 2. Specialized Task Prediction Heads
        self.restoration_head = RestorationHead(
            in_channels=shared_dim,
            mid_channels=shared_dim,
            scale_factor=scale_factor
        )

        self.degradation_head = DegradationHead(
            in_channels=shared_dim,
            num_params=num_degradation_params,
            hidden_dim=shared_dim * 2
        )

        self.uncertainty_head = UncertaintyHead(
            in_channels=shared_dim,
            mid_channels=shared_dim,
            scale_factor=scale_factor,
            min_log_variance=min_log_variance,
            max_log_variance=max_log_variance
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single forward pass.
        
        Args:
            x: Degraded NoisyLR input tensor [B, 1, 128, 128]
            
        Returns:
            Dictionary containing:
            - "restored": [B, 1, 256, 256] Restored image in [0.0, 1.0]
            - "degradation": [B, 4] Predicted degradation parameters
            - "confidence": [B, 1, 256, 256] Spatially aligned log-variance uncertainty map
        """
        # 1. Single Shared Backbone Pass
        shared_features = self.backbone(x) # [B, C, 128, 128]

        # 2. Multi-Head Predictions from Shared Features
        restored = self.restoration_head(shared_features)   # [B, 1, 256, 256]
        degradation = self.degradation_head(shared_features) # [B, 4]
        uncertainty = self.uncertainty_head(shared_features) # [B, 1, 256, 256]

        return {
            "restored": restored,
            "degradation": degradation,
            "confidence": uncertainty  # Contains log-variance log(sigma^2)
        }

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @classmethod
    def from_config(cls, config_path: str) -> "MultiHeadRestorationNet":
        """Factory method constructing MultiHeadRestorationNet from YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        m_cfg = cfg.get("model", cfg)
        deg_cfg = m_cfg.get("degradation_params", {})
        unc_cfg = m_cfg.get("uncertainty", {})

        return cls(
            in_channels=m_cfg.get("in_channels", 1),
            encoder_channels=m_cfg.get("encoder_channels", [32, 64, 128]),
            num_blocks=m_cfg.get("num_blocks", [2, 2, 4]),
            scale_factor=m_cfg.get("scale_factor", 2),
            num_degradation_params=deg_cfg.get("num_params", 4),
            min_log_variance=unc_cfg.get("min_log_variance", -10.0),
            max_log_variance=unc_cfg.get("max_log_variance", 10.0)
        )
