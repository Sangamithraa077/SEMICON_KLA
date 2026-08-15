"""Timed Inference Script for SEMICON KLA Hackathon 2026.

CRITICAL SUBMISSION FILE:
Accepts --input_dir and --output_dir arguments.
Restores every supported image in input_dir and saves to output_dir.
Strictly decoupled from heavy metric evaluation or report generation.
"""

import os
import argparse
import time
import torch
import numpy as np

from src.utils.config import load_config, get_device
from src.utils.image_io import list_image_files, load_image, save_image
from src.models.multi_head_net import MultiHeadRestorationNet
from src.models.baselines import simple_denoise_baseline

def parse_args():
    parser = argparse.ArgumentParser(description="SEMICON KLA Timed Image Restoration Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input degraded images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored images")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cuda, cpu, mps")
    return parser.parse_args()

def run_inference():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = {}
        
    device_name = get_device(args.device)
    print(f"[Inference] Target Device: {device_name}")
    print(f"[Inference] Input Directory: {args.input_dir}")
    print(f"[Inference] Output Directory: {args.output_dir}")
    
    image_paths = list_image_files(args.input_dir)
    if not image_paths:
        print(f"[Inference] Warning: No supported images found in '{args.input_dir}'.")
        return
        
    print(f"[Inference] Found {len(image_paths)} images to restore.")
    
    # Initialize Multi-Head Restoration Network
    model_loaded = False
    if torch.__version__:
        device = torch.device(device_name)
        model = MultiHeadRestorationNet(in_channels=3, out_channels=3, width=64).to(device)
        model.eval()
        
        weights_path = args.weights or config.get("inference", {}).get("weights_path")
        if weights_path and os.path.exists(weights_path):
            print(f"[Inference] Loading weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            model_loaded = True
        else:
            print(f"[Inference] Note: No checkpoint found at '{weights_path}'. Running initial forward pass mode.")
            model_loaded = True
            
    start_time = time.time()
    
    with torch.no_grad():
        for img_path in image_paths:
            fname = os.path.basename(img_path)
            out_path = os.path.join(args.output_dir, fname)
            
            img_np = load_image(img_path) # (H, W, 3) float32 in [0, 1]
            
            if model_loaded:
                # Convert to tensor (1, C, H, W)
                img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                # Forward pass through Multi-Head Network
                restored_tensor, deg_params, unc_map = model(img_tensor)
                
                # Convert back to numpy (H, W, C)
                restored_np = restored_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                restored_np = np.clip(restored_np, 0, 1)
            else:
                # Baseline fallback
                restored_np = simple_denoise_baseline(img_np)
                
            save_image(restored_np, out_path)
            
    total_time = time.time() - start_time
    print(f"[Inference] Completed restoration of {len(image_paths)} images in {total_time:.2f} seconds ({total_time / len(image_paths):.4f} s/img).")

if __name__ == "__main__":
    run_inference()
