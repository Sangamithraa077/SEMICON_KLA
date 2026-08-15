"""Domain-Informed Synthetic Degradation Pipeline.

Simulates semiconductor defect types:
- Poisson-Gaussian shot & read noise
- Optical/Defocus blur
- Downsampling / resolution loss
- Logging exact degradation parameters theta_deg = [sigma_p, sigma_g, blur_k, scale] for Head 2 training
"""

import numpy as np
from typing import Tuple, Dict, Any
from .poisson_gaussian import apply_poisson_gaussian_noise

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    from PIL import Image, ImageFilter

def apply_synthetic_degradation(
    clean_image: np.ndarray,
    poisson_scale: float = None,
    gaussian_std: float = None,
    blur_ksize: int = None,
    downsample_scale: float = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Applies synthetic degradation pipeline and logs parameter pseudo-labels.
    
    Args:
        clean_image: Input clean image array (H, W, C) in [0, 1]
        poisson_scale: Optional fixed Poisson noise scale
        gaussian_std: Optional fixed Gaussian noise std
        blur_ksize: Optional fixed blur kernel size (odd int)
        downsample_scale: Optional fixed scale factor for downsampling
        
    Returns:
        Tuple of (degraded_image, degradation_params_dict)
    """
    if poisson_scale is None:
        poisson_scale = float(np.random.uniform(0.001, 0.04))
    if gaussian_std is None:
        gaussian_std = float(np.random.uniform(0.005, 0.05))
    if blur_ksize is None:
        blur_ksize = int(np.random.choice([3, 5, 7]))
    if downsample_scale is None:
        downsample_scale = float(np.random.uniform(1.0, 3.0))
        
    degraded = clean_image.copy()
    
    # 1. Apply Defocus/Optical Blur
    if blur_ksize > 1:
        if HAS_OPENCV:
            degraded = cv2.GaussianBlur(degraded, (blur_ksize, blur_ksize), 0)
        else:
            radius = (blur_ksize - 1) / 2.0
            pil_img = Image.fromarray((degraded * 255).astype(np.uint8))
            pil_blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            degraded = (np.array(pil_blurred) / 255.0).astype(np.float32)
        
    # 2. Apply Resolution Downsampling & Upsampling
    if downsample_scale > 1.0:
        h, w = degraded.shape[:2]
        low_h, low_w = max(4, int(h / downsample_scale)), max(4, int(w / downsample_scale))
        if HAS_OPENCV:
            degraded_low = cv2.resize(degraded, (low_w, low_h), interpolation=cv2.INTER_AREA)
            degraded = cv2.resize(degraded_low, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            pil_img = Image.fromarray((degraded * 255).astype(np.uint8))
            pil_low = pil_img.resize((low_w, low_h), Image.Resampling.BILINEAR)
            pil_up = pil_low.resize((w, h), Image.Resampling.BICUBIC)
            degraded = (np.array(pil_up) / 255.0).astype(np.float32)
        
    # 3. Apply Signal-Dependent Poisson-Gaussian Noise
    degraded = apply_poisson_gaussian_noise(degraded, poisson_scale, gaussian_std)
    
    degradation_params = {
        "poisson_scale": poisson_scale,
        "gaussian_std": gaussian_std,
        "blur_ksize": float(blur_ksize),
        "downsample_scale": downsample_scale
    }
    
    return degraded, degradation_params

