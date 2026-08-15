"""Interactive Demo Pipeline Entry Point.

Generates a visual quad-display side-by-side:
1. Input Degraded Image
2. Baseline Result (Bicubic / Denoised)
3. Multi-Head Restored Image
4. Per-Pixel Confidence/Uncertainty Heatmap alongside estimated degradation vector.
"""

import os
import argparse
import numpy as np
import torch
import cv2

from src.utils.config import load_config, get_device
from src.utils.image_io import load_image, save_image
from src.models.multi_head_net import MultiHeadRestorationNet
from src.models.baselines import simple_denoise_baseline
from src.degradation.synthetic_pipeline import apply_synthetic_degradation

def main():
    parser = argparse.ArgumentParser(description="Run Demo Visual Quad-Display Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to clean test image")
    parser.add_argument("--output", type=str, default="results/demo_quad.png", help="Path to save output quad visual")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image '{args.image}' not found.")
        return

    clean_img = load_image(args.image)
    degraded_img, deg_params = apply_synthetic_degradation(clean_img)
    baseline_img = simple_denoise_baseline(degraded_img)

    device = torch.device(get_device("auto"))
    model = MultiHeadRestorationNet().to(device)
    model.eval()

    with torch.no_grad():
        inp_tensor = torch.from_numpy(degraded_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
        restored_tensor, est_deg, unc_map = model(inp_tensor)
        
        restored_img = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        unc_np = unc_map.squeeze().cpu().numpy()

    # Normalize uncertainty map to color heatmap
    unc_norm = cv2.normalize(unc_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(unc_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    # Build 2x2 Quad Visual Grid
    top_row = np.hstack([degraded_img, baseline_img])
    bot_row = np.hstack([np.clip(restored_img, 0, 1), heatmap_rgb])
    quad_visual = np.vstack([top_row, bot_row])

    save_image(quad_visual, args.output)
    print(f"[Demo] Saved quad visualization to '{args.output}'.")
    print(f"[Demo] Estimated Degradation Parameters: {est_deg.squeeze().cpu().tolist()}")

if __name__ == "__main__":
    main()
