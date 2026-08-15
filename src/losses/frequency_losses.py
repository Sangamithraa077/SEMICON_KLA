"""FFT / Frequency-Domain Loss Objective."""

import torch
import torch.nn as nn

class FFTLoss(nn.Module):
    """Frequency-domain L1 loss using 2D Fast Fourier Transform."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        
        # Real and imaginary components distance
        loss_real = torch.abs(pred_fft.real - target_fft.real).mean()
        loss_imag = torch.abs(pred_fft.imag - target_fft.imag).mean()
        return loss_real + loss_imag
