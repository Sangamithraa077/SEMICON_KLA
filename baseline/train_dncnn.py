"""Train DnCNN Denoising Baseline (Baseline 2).

Trains DnCNN model from scratch using the REAL paired dataset:
NoisyLR 128x128 -> GT 256x256 (2,560 training samples, 640 validation samples).
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config, get_device
from src.data.dataloader import build_dataloaders
from baseline.dncnn import DnCNNBaseline, count_parameters

def parse_args():
    parser = argparse.ArgumentParser(description="Train DnCNN Baseline on Real Semiconductor Paired Dataset")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--crop_size", type=int, default=128, help="Crop size for fast training")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of training samples for fast CPU runs")
    parser.add_argument("--save_path", type=str, default="checkpoints/dncnn_baseline.pth", help="Checkpoint save path")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for reproducibility")
    return parser.parse_args()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        try:
            torch.set_num_threads(os.cpu_count() or 4)
        except Exception:
            pass


def calculate_psnr_batch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return float(psnr.mean().item())

def train_dncnn():
    args = parse_args()
    set_seed(args.seed)

    config = load_config(args.config).get("data", {})
    gt_dir = config.get("train_gt_dir")
    noisy_dir = config.get("train_noisy_dir")

    device_str = get_device("auto")
    device = torch.device(device_str)
    print(f"[Train DnCNN] Device: {device}")
    print(f"[Train DnCNN] GT Directory: {gt_dir}")
    print(f"[Train DnCNN] NoisyLR Directory: {noisy_dir}")

    # Build DataLoaders on real paired dataset
    loaders = build_dataloaders(
        mode="paired",
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        batch_size=args.batch_size,
        num_workers=0, # Multi-processing safe for Windows
        val_ratio=config.get("val_ratio", 0.2),
        crop_size=args.crop_size,
        scale_factor=2.0,
        seed=args.seed,
        max_samples=args.max_samples
    )


    train_loader = loaders["train"]
    val_loader = loaders["val"]

    print(f"[Train DnCNN] Training Samples: {len(train_loader.dataset)}")
    print(f"[Train DnCNN] Validation Samples: {len(val_loader.dataset)}")

    # Instantiate DnCNN Baseline
    model = DnCNNBaseline(in_channels=1, num_features=64, num_layers=7).to(device)
    param_count = count_parameters(model)
    print(f"[Train DnCNN] Total Trainable Parameters: {param_count:,}")

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_psnr = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    start_train_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            noisy = batch["degraded"].to(device) # (B, 1, 128, 128)
            gt = batch["clean"].to(device)       # (B, 1, 256, 256)

            optimizer.zero_grad()
            output = model(noisy) # (B, 1, 256, 256)
            loss = criterion(output, gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy.size(0)

            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.5f}", flush=True)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation pass
        model.eval()
        val_psnr_list = []
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["degraded"].to(device)
                gt = batch["clean"].to(device)
                output = model(noisy)
                val_psnr_list.append(calculate_psnr_batch(output, gt))

        avg_val_psnr = float(np.mean(val_psnr_list))
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {epoch_loss:.5f} | Val PSNR: {avg_val_psnr:.2f} dB", flush=True)

        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_psnr": best_val_psnr,
                "param_count": param_count,
                "seed": args.seed
            }, args.save_path)
            print(f" -> Saved new best checkpoint to '{args.save_path}' (Val PSNR: {best_val_psnr:.2f} dB)", flush=True)

    total_time = time.time() - start_train_time
    print(f"\n[Train DnCNN] Training completed in {total_time:.2f}s. Best Val PSNR: {best_val_psnr:.2f} dB", flush=True)


if __name__ == "__main__":
    train_dncnn()
