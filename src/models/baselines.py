import numpy as np
import torch

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    from PIL import Image, ImageFilter

def bicubic_baseline(img_array: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """Bicubic interpolation baseline.
    
    Args:
        img_array: (H, W, C) float32 in [0, 1]
        scale_factor: Scale factor for resizing
        
    Returns:
        Interpolated numpy array (H, W, C)
    """
    if scale_factor == 1.0:
        return img_array.copy()
    
    h, w = img_array.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    if HAS_OPENCV:
        img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        resized = cv2.resize(img_uint8, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return (resized / 255.0).astype(np.float32)
    else:
        pil_img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))
        pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return (np.array(pil_resized) / 255.0).astype(np.float32)

def simple_denoise_baseline(img_array: np.ndarray, h_param: float = 10.0) -> np.ndarray:
    """Simple Non-Local Means / Gaussian Denoising Baseline (DnCNN substitute for CPU/light tests).
    
    Args:
        img_array: (H, W, C) float32 in [0, 1]
        h_param: Filter strength
        
    Returns:
        Denoised image array (H, W, C)
    """
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    if HAS_OPENCV:
        denoised_bgr = cv2.fastNlMeansDenoisingColored(
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
            None,
            h_param,
            h_param,
            7,
            21
        )
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        return (denoised_rgb / 255.0).astype(np.float32)
    else:
        pil_img = Image.fromarray(img_uint8)
        pil_denoised = pil_img.filter(ImageFilter.BoxBlur(radius=1))
        return (np.array(pil_denoised) / 255.0).astype(np.float32)

