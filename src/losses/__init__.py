from .spatial_losses import GradientEdgeLoss
from .frequency_losses import FFTLoss
from .perceptual_losses import HeteroscedasticUncertaintyLoss, DegradationHeadLoss

__all__ = [
    "GradientEdgeLoss",
    "FFTLoss",
    "HeteroscedasticUncertaintyLoss",
    "DegradationHeadLoss"
]
