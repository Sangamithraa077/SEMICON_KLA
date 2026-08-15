"""Shared NAFNet-Style / Gated CNN Encoder-Decoder Backbone.

NAFNet (Nonlinear Activation Free Network) uses SimpleGate blocks:
SimpleGate splits feature maps into two halves along the channel dimension
and computes their element-wise product.
"""

import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    """Element-wise multiplication gating mechanism replacing traditional non-linear activations."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """NAFNet block skeleton with gated CNN structure and simplified channel attention."""
    def __init__(self, c: int):
        super().__init__()
        # TODO Phase 2: Implement full NAFBlock with Depthwise Conv, SimpleGate, and SCA
        self.conv1 = nn.Conv2d(c, c * 2, 3, padding=1)
        self.gate = SimpleGate()
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.gate(self.conv1(x))
        out = self.conv2(out)
        return res + out

class NAFNetBackbone(nn.Module):
    """NAFNet-style Gated CNN Encoder-Decoder Backbone skeleton."""
    def __init__(self, in_channels: int = 3, width: int = 64):
        super().__init__()
        # TODO Phase 2: Complete encoder-decoder levels with skip connections
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.block = NAFBlock(width)
        self.outro = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.intro(x)
        feats = self.block(feats)
        return self.outro(feats)
