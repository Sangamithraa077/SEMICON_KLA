"""Perceptual, Degradation Head, and Heteroscedastic Uncertainty Loss Functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteroscedasticUncertaintyLoss(nn.Module):
    """Head 3 per-pixel uncertainty loss:
    L_unc = (||I_restored - I_clean||^2) / (2 * U) + 0.5 * log(U)
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        # Clamp uncertainty to avoid division by zero
        unc = torch.clamp(uncertainty, min=1e-4, max=10.0)
        diff_sq = (pred - target) ** 2
        loss = (diff_sq / (2.0 * unc)) + 0.5 * torch.log(unc)
        return loss.mean()

class DegradationHeadLoss(nn.Module):
    """Head 2 loss matching predicted degradation parameters with logged pseudo-labels."""
    def forward(self, pred_params: torch.Tensor, target_params: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_params, target_params)
