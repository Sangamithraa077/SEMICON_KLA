"""Quantitative Image Quality Metrics (PSNR, SSIM) with Zero External Hard Dependencies."""

import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: Restored image array [0, 1]
        img2: Ground truth image array [0, 1]
        data_range: Maximum pixel value range (1.0)
        
    Returns:
        PSNR value in decibels (dB)
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range / np.sqrt(mse)))

def _numpy_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Fallback NumPy implementation of Structural Similarity Index (SSIM)."""
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    var1 = np.var(img1)
    var2 = np.var(img2)
    cov12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim_num = (2 * mu1 * mu2 + C1) * (2 * cov12 + C2)
    ssim_den = (mu1**2 + mu2**2 + C1) * (var1 + var2 + C2)
    
    return float(ssim_num / ssim_den)

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Calculates Structural Similarity Index (SSIM).
    
    Args:
        img1: Restored image array [0, 1]
        img2: Ground truth image array [0, 1]
        data_range: Maximum pixel value range
        
    Returns:
        SSIM index score in [0, 1]
    """
    if HAS_SKIMAGE:
        if img1.ndim == 3 and img1.shape[2] == 3:
            return float(skimage_ssim(img1, img2, channel_axis=2, data_range=data_range))
        return float(skimage_ssim(img1, img2, data_range=data_range))
    else:
        return _numpy_ssim(img1, img2, data_range=data_range)
