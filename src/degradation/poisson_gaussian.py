"""Poisson-Gaussian Signal-Dependent Noise Model.

Physics of Semiconductor Imaging (SEM / Optical):
Shot noise (Poisson) scales with photon count (signal intensity), while
readout/electronic noise (Gaussian) remains additive and constant:

    y = x + sqrt(sigma_g^2 + sigma_p^2 * x) * epsilon
    where epsilon ~ N(0, I)
"""

import numpy as np

def apply_poisson_gaussian_noise(
    image: np.ndarray,
    poisson_scale: float = 0.01,
    gaussian_std: float = 0.02
) -> np.ndarray:
    """Applies signal-dependent Poisson-Gaussian noise to image [0, 1].
    
    Args:
        image: Numpy float32 array in [0, 1]
        poisson_scale (sigma_p): Scale parameter for signal-dependent noise
        gaussian_std (sigma_g): Standard deviation for electronic read noise
        
    Returns:
        Noisy image array clipped to [0, 1]
    """
    image_clamped = np.clip(image, 0, 1)
    
    # Calculate local noise standard deviation sqrt(sigma_g^2 + sigma_p^2 * x)
    variance = (gaussian_std ** 2) + (poisson_scale ** 2) * image_clamped
    sigma_total = np.sqrt(variance)
    
    noise = np.random.normal(0, 1, size=image.shape).astype(np.float32)
    noisy_image = image_clamped + sigma_total * noise
    
    return np.clip(noisy_image, 0, 1).astype(np.float32)
