"""Visualization Utility for KLA Semiconductor Restoration Pipeline.

Loads a sample clean image, applies synthetic Poisson-Gaussian degradation,
saves side-by-side visualization PNG, and logs exact degradation parameters.
"""

import os
import sys
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.image_io import save_image
from src.data.dataset import load_file_array
from src.data.degradation import generate_poisson_gaussian_degradation


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Clean vs Degraded Semiconductor Image")
    parser.add_argument("--gt_path", type=str, default=None, help="Path to clean GT file (.npy or image)")
    parser.add_argument("--output", type=str, default="results/degradation_sample.png", help="Output PNG path")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible degradation")
    return parser.parse_args()

def main():
    args = parse_args()
    gt_path = args.gt_path

    if not gt_path and os.path.exists(args.config):
        cfg = load_config(args.config).get("data", {})
        gt_dir = cfg.get("train_gt_dir")
        if gt_dir and os.path.exists(gt_dir):
            files = [f for f in os.listdir(gt_dir) if f.endswith('.npy')]
            if files:
                gt_path = os.path.join(gt_dir, files[0])

    if not gt_path or not os.path.exists(gt_path):
        print(f"Generating synthetic pattern (GT file not specified or found at '{gt_path}')...")
        clean_img = np.zeros((256, 256), dtype=np.float32)
        clean_img[64:192, 64:192] = 0.8
        clean_img[96:160, 96:160] = 0.2
    else:
        print(f"Loading GT sample from: {gt_path}")
        clean_img = load_file_array(gt_path)

    degraded_img, clean_img, deg_params = generate_poisson_gaussian_degradation(
        clean_img,
        seed=args.seed
    )

    print("\n--- Applied Degradation Parameters ---")
    for k, v in deg_params.items():
        print(f"  {k}: {v}")

    # Convert 2D arrays to 3D for visualization side-by-side
    if clean_img.ndim == 2:
        clean_rgb = np.stack([clean_img]*3, axis=-1)
        degraded_rgb = np.stack([degraded_img]*3, axis=-1)
    else:
        clean_rgb = clean_img
        degraded_rgb = degraded_img

    side_by_side = np.hstack([clean_rgb, degraded_rgb])
    save_image(side_by_side, args.output)

    print(f"\n[Visualization] Saved side-by-side comparison to: {args.output}")

if __name__ == "__main__":
    main()
