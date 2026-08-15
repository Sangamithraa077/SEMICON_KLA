"""Evaluation and Comparison Script for Bicubic and DnCNN Baselines.

Evaluates Bicubic and DnCNN baselines on the deterministic 640-sample validation set.
Calculates:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity, converting 1-channel grayscale to 3-channel RGB)
- Detailed Runtime Benchmarking (Model Load Time, Total Val Inference Time, Avg ms/img, FPS)

Outputs:
- results/baseline_comparison.json
- results/baseline/bicubic/*.png
- results/baseline/dncnn/*.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config, get_device
from src.utils.image_io import save_image
from src.evaluation.metrics import calculate_psnr, calculate_ssim
from src.data.dataloader import build_dataloaders
from baseline.bicubic import bicubic_upsample_2x
from baseline.dncnn import DnCNNBaseline, count_parameters

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Baselines on Validation Dataset")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--dncnn_weights", type=str, default="checkpoints/dncnn_baseline.pth", help="DnCNN checkpoint path")
    parser.add_argument("--output_json", type=str, default="results/baseline_comparison.json", help="Output JSON metrics path")
    parser.add_argument("--output_img_dir", type=str, default="results/baseline", help="Directory to save sample outputs")
    parser.add_argument("--num_save_samples", type=int, default=10, help="Number of representative validation images to save")
    return parser.parse_args()

def compute_lpips_batch(lpips_model, pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, device: torch.device):
    """Computes LPIPS score converting 1-channel grayscale tensors to 3-channel RGB.
    
    LPIPS requires 3-channel RGB input in [-1, 1].
    Conversion logic: repeat 1-channel along dim 1 and scale [0, 1] to [-1, 1].
    Returns None if lpips_model is None.
    """
    if lpips_model is None:
        return None

    if pred_tensor.shape[1] == 1:
        pred_rgb = pred_tensor.repeat(1, 3, 1, 1)
        gt_rgb = gt_tensor.repeat(1, 3, 1, 1)
    else:
        pred_rgb = pred_tensor
        gt_rgb = gt_tensor

    # Scale [0, 1] -> [-1, 1]
    pred_scaled = pred_rgb * 2.0 - 1.0
    gt_scaled = gt_rgb * 2.0 - 1.0

    with torch.no_grad():
        dist = lpips_model(pred_scaled.to(device), gt_scaled.to(device))
    return float(dist.mean().item())


def evaluate_baselines():
    args = parse_args()

    config = load_config(args.config).get("data", {})
    gt_dir = config.get("train_gt_dir")
    noisy_dir = config.get("train_noisy_dir")
    seed = config.get("seed", 42)

    device_str = get_device("auto")
    device = torch.device(device_str)
    print(f"[Evaluate] Compute Device: {device}")

    # Build validation dataloader (640 samples, seed 42)
    loaders = build_dataloaders(
        mode="paired",
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        batch_size=16,
        num_workers=0,
        val_ratio=config.get("val_ratio", 0.2),
        scale_factor=2.0,
        seed=seed
    )
    val_loader = loaders["val"]
    val_count = len(val_loader.dataset)
    print(f"[Evaluate] Validation Sample Count: {val_count}")


    # Initialize LPIPS model if available
    lpips_model = None
    if HAS_LPIPS:
        try:
            print("[Evaluate] Loading LPIPS feature extractor (VGG)...")
            lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device)
            lpips_model.eval()
        except Exception as e:
            print(f"[Evaluate] LPIPS initialization note: {e}")
            lpips_model = None
    else:
        print("[Evaluate] LPIPS package not installed. (Grayscale to 3-channel RGB conversion prepared).")

    # =========================================================================
    # 1. EVALUATE BASELINE 1 — BICUBIC UPSAMPLING
    # =========================================================================
    print("\n--- Evaluating Baseline 1: Bicubic 2x Upsampling ---")
    bicubic_psnrs, bicubic_ssims, bicubic_lpipss = [], [], []

    bicubic_save_dir = os.path.join(args.output_img_dir, "bicubic")
    os.makedirs(bicubic_save_dir, exist_ok=True)

    start_bicubic_time = time.time()

    saved_bicubic_count = 0
    for batch in val_loader:
        noisy_tensor = batch["degraded"] # (B, 1, 128, 128)
        gt_tensor = batch["clean"]       # (B, 1, 256, 256)
        fnames = batch["filename"]

        restored_tensor = bicubic_upsample_2x(noisy_tensor)
        lpips_v = compute_lpips_batch(lpips_model, restored_tensor, gt_tensor, device)

        b_size = noisy_tensor.size(0)
        for i in range(b_size):
            rest_np = restored_tensor[i].squeeze().numpy()
            gt_np = gt_tensor[i].squeeze().numpy()

            psnr_v = calculate_psnr(rest_np, gt_np)
            ssim_v = calculate_ssim(rest_np, gt_np)

            bicubic_psnrs.append(psnr_v)
            bicubic_ssims.append(ssim_v)

            if lpips_v is not None:
                bicubic_lpipss.append(lpips_v)


            if saved_bicubic_count < args.num_save_samples:
                save_name = os.path.splitext(fnames[i])[0] + ".png"
                save_image(rest_np, os.path.join(bicubic_save_dir, save_name))
                saved_bicubic_count += 1

    bicubic_total_time = time.time() - start_bicubic_time
    bicubic_avg_ms = (bicubic_total_time / val_count) * 1000.0
    bicubic_fps = val_count / bicubic_total_time

    bicubic_mean_psnr = float(np.mean(bicubic_psnrs))
    bicubic_mean_ssim = float(np.mean(bicubic_ssims))
    bicubic_mean_lpips = float(np.mean(bicubic_lpipss)) if bicubic_lpipss else None

    lpips_str_bicubic = f"{bicubic_mean_lpips:.4f}" if bicubic_mean_lpips is not None else "N/A (lpips package not installed)"
    print(f"Bicubic Results: PSNR = {bicubic_mean_psnr:.2f} dB | SSIM = {bicubic_mean_ssim:.4f} | LPIPS = {lpips_str_bicubic}", flush=True)
    print(f"Bicubic Runtime: Total = {bicubic_total_time:.3f}s | Avg = {bicubic_avg_ms:.2f} ms/img | Speed = {bicubic_fps:.1f} img/s", flush=True)

    # =========================================================================
    # 2. EVALUATE BASELINE 2 — SIMPLE DnCNN
    # =========================================================================
    print("\n--- Evaluating Baseline 2: Simple DnCNN Model ---", flush=True)
    dncnn_save_dir = os.path.join(args.output_img_dir, "dncnn")
    os.makedirs(dncnn_save_dir, exist_ok=True)

    start_dncnn_load = time.time()
    dncnn_model = DnCNNBaseline(in_channels=1, num_features=64, num_layers=7).to(device)
    dncnn_param_count = count_parameters(dncnn_model)

    if os.path.exists(args.dncnn_weights):
        print(f"[Evaluate] Loading trained DnCNN weights from: {args.dncnn_weights}", flush=True)
        ckpt = torch.load(args.dncnn_weights, map_location=device)
        dncnn_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    else:
        print(f"[Evaluate] Note: Checkpoint '{args.dncnn_weights}' not found. Running forward pass with initial weights.", flush=True)

    dncnn_model.eval()
    dncnn_load_time = time.time() - start_dncnn_load
    print(f"DnCNN Model Load Time: {dncnn_load_time * 1000.0:.2f} ms | Parameters: {dncnn_param_count:,}", flush=True)

    dncnn_psnrs, dncnn_ssims, dncnn_lpipss = [], [], []
    saved_dncnn_count = 0
    start_dncnn_inference = time.time()

    with torch.no_grad():
        for batch in val_loader:
            noisy_tensor = batch["degraded"].to(device) # (B, 1, 128, 128)
            gt_tensor = batch["clean"].to(device)       # (B, 1, 256, 256)
            fnames = batch["filename"]

            restored_tensor = dncnn_model(noisy_tensor) # (B, 1, 256, 256)
            lpips_v = compute_lpips_batch(lpips_model, restored_tensor.cpu(), gt_tensor.cpu(), device)

            b_size = noisy_tensor.size(0)
            for i in range(b_size):
                rest_np = restored_tensor[i].squeeze().cpu().numpy()
                gt_np = gt_tensor[i].squeeze().cpu().numpy()

                psnr_v = calculate_psnr(rest_np, gt_np)
                ssim_v = calculate_ssim(rest_np, gt_np)

                dncnn_psnrs.append(psnr_v)
                dncnn_ssims.append(ssim_v)
                if lpips_v is not None:
                    dncnn_lpipss.append(lpips_v)

                if saved_dncnn_count < args.num_save_samples:
                    save_name = os.path.splitext(fnames[i])[0] + ".png"
                    save_image(rest_np, os.path.join(dncnn_save_dir, save_name))
                    saved_dncnn_count += 1

    dncnn_inference_time = time.time() - start_dncnn_inference
    dncnn_avg_ms = (dncnn_inference_time / val_count) * 1000.0
    dncnn_fps = val_count / dncnn_inference_time

    dncnn_mean_psnr = float(np.mean(dncnn_psnrs))
    dncnn_mean_ssim = float(np.mean(dncnn_ssims))
    dncnn_mean_lpips = float(np.mean(dncnn_lpipss)) if dncnn_lpipss else None

    lpips_str_dncnn = f"{dncnn_mean_lpips:.4f}" if dncnn_mean_lpips is not None else "N/A (lpips package not installed)"
    print(f"DnCNN Results:   PSNR = {dncnn_mean_psnr:.2f} dB | SSIM = {dncnn_mean_ssim:.4f} | LPIPS = {lpips_str_dncnn}", flush=True)
    print(f"DnCNN Runtime:   Inference = {dncnn_inference_time:.3f}s | Avg = {dncnn_avg_ms:.2f} ms/img | Speed = {dncnn_fps:.1f} img/s", flush=True)

    # =========================================================================
    # 3. SAVE RESULTS TO JSON
    # =========================================================================
    lpips_status_msg = "Measured using lpips package (VGG feature space)" if HAS_LPIPS else "LPIPS unavailable: lpips package not installed"

    results = {
        "dataset_info": {
            "total_paired_samples": 3200,
            "train_samples": len(val_loader.dataset) * 4, # 2560
            "validation_samples": val_count,              # 640
            "gt_shape": [256, 256],
            "noisylr_shape": [128, 128],
            "scale_factor": 2.0,
            "seed": seed
        },
        "lpips_status": lpips_status_msg,
        "baselines": {
            "bicubic": {
                "psnr": round(bicubic_mean_psnr, 2),
                "ssim": round(bicubic_mean_ssim, 4),
                "lpips": round(bicubic_mean_lpips, 4) if bicubic_mean_lpips is not None else None,
                "runtime": {
                    "total_inference_seconds": round(bicubic_total_time, 3),
                    "avg_ms_per_image": round(bicubic_avg_ms, 2),
                    "images_per_second": round(bicubic_fps, 1)
                }
            },
            "dncnn": {
                "architecture": "7-Layer Conv+BatchNorm+ReLU Residual Denoising CNN",
                "parameter_count": dncnn_param_count,
                "weights_checkpoint": args.dncnn_weights,
                "psnr": round(dncnn_mean_psnr, 2),
                "ssim": round(dncnn_mean_ssim, 4),
                "lpips": round(dncnn_mean_lpips, 4) if dncnn_mean_lpips is not None else None,
                "runtime": {
                    "model_load_seconds": round(dncnn_load_time, 4),
                    "total_inference_seconds": round(dncnn_inference_time, 3),
                    "avg_ms_per_image": round(dncnn_avg_ms, 2),
                    "images_per_second": round(dncnn_fps, 1)
                }
            }
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Evaluate] Saved baseline comparison metrics to: {args.output_json}", flush=True)
    return results


if __name__ == "__main__":
    evaluate_baselines()
