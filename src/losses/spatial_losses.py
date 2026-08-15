"""Spatial Loss Objectives (L1, SSIM, Gradient/Edge Loss)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientEdgeLoss(nn.Module):
    """Gradient / Edge loss to enforce sharp semiconductor pattern edges."""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pred.shape
        kx = self.kernel_x.repeat(c, 1, 1, 1)
        ky = self.kernel_y.repeat(c, 1, 1, 1)
        
        pred_grad_x = F.conv2d(pred, kx, padding=1, groups=c)
        pred_grad_y = F.conv2d(pred, ky, padding=1, groups=c)
        
        target_grad_x = F.conv2d(target, kx, padding=1, groups=c)
        target_grad_y = F.conv2d(target, ky, padding=1, groups=c)
        
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return loss_x + loss_y
