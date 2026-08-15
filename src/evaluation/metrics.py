"""Quantitative Image Quality Metrics (PSNR, SSIM)."""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as skimage_ssim

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: Restored image array [0, 1]
        img2: Ground truth image array [0, 1]
        data_range: Maximum pixel value range (1.0 or 255.0)
        
    Returns:
        PSNR value in decibels (dB)
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range / np.sqrt(mse)))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0) -> float:
    """Calculates Structural Similarity Index (SSIM).
    
    Args:
        img1: Restored image array [0, 1]
        img2: Ground truth image array [0, 1]
        data_range: Maximum pixel value range
        
    Returns:
        SSIM index score in [0, 1]
    """
    if img1.ndim == 3:
        return float(skimage_ssim(img1, img2, channel_axis=2, data_range=data_range))
    return float(skimage_ssim(img1, img2, data_range=data_range))
