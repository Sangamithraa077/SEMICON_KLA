"""Bicubic Upsampling Baseline (Baseline 1).

Performs 2x bicubic upsampling from 128x128 NoisyLR to 256x256 output.
Preserves float32 precision and handles 2D grayscale NumPy arrays and PyTorch Tensors.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    from PIL import Image

def bicubic_upsample_2x(input_data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Upsamples 128x128 input by 2x using bicubic interpolation to 256x256.

    Args:
        input_data: NumPy array (H, W) or (H, W, C) float32, or PyTorch Tensor (B, C, H, W).

    Returns:
        Upsampled data of matching type (256, 256) or (B, C, 256, 256).
    """
    if isinstance(input_data, torch.Tensor):
        # PyTorch Tensor (B, C, H, W)
        return F.interpolate(input_data, scale_factor=2.0, mode="bicubic", align_corners=False)

    elif isinstance(input_data, np.ndarray):
        h, w = input_data.shape[:2]
        target_h, target_w = int(h * 2), int(w * 2)

        if HAS_OPENCV:
            if input_data.ndim == 2:
                upsampled = cv2.resize(input_data, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            else:
                upsampled = cv2.resize(input_data, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Pillow fallback preserving float precision
            uint8_img = np.clip(input_data * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(uint8_img)
            pil_resized = pil_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
            upsampled = (np.array(pil_resized) / 255.0).astype(np.float32)

        return upsampled.astype(np.float32)

    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")
