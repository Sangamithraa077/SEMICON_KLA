import os
import numpy as np
from typing import List, Tuple

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    from PIL import Image

SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

def list_image_files(directory: str) -> List[str]:
    """Finds all supported image files in a directory.
    
    Args:
        directory: Target directory path.
        
    Returns:
        List of absolute file paths to valid images.
    """
    if not os.path.exists(directory):
        return []
        
    files = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            files.append(os.path.join(directory, fname))
    return files

def load_image(image_path: str) -> np.ndarray:
    """Loads image from path in RGB format scaled to [0, 1].
    
    Args:
        image_path: Path to image file.
        
    Returns:
        Numpy array (H, W, C) float32 in [0, 1].
    """
    if HAS_OPENCV:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_pil = Image.open(image_path).convert('RGB')
        img = np.array(img_pil)
    return (img / 255.0).astype(np.float32)

def save_image(img_array: np.ndarray, save_path: str) -> None:
    """Saves numpy float array [0, 1] to disk as RGB/BGR image.
    
    Args:
        img_array: Numpy array (H, W, C) float32 in [0, 1] or uint8 [0, 255].
        save_path: Output file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)
        
    if HAS_OPENCV:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
    else:
        img_pil = Image.fromarray(img_array)
        img_pil.save(save_path)

