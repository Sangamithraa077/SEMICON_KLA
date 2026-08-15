"""Training-Flow Smoke Test Script.

Sanity check verifying model construction, forward pass execution, loss calculation,
and optimizer gradient updates over 10 optimization steps on 1-2 real training samples.

Note: This is an architecture/training-flow sanity check, NOT a full training or overfit experiment.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import MultiHeadRestorationNet
from src.data.dataset import KLASemiconductorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Head Model Training-Flow Smoke Test")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config file")
    parser.add_argument("--gt_dir", type=str, default="C:/Users/abise/Downloads/train/train/GT", help="GT directory")
    parser.add_argument("--noisy_dir", type=str, default="C:/Users/abise/Downloads/train/train/NoisyLR", help="NoisyLR directory")
    parser.add_argument("--steps", type=int, default=10, help="Number of optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def run_smoke_test():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Multi-Head Model Training-Flow Smoke Test ===", flush=True)
    print(f"[Smoke Test] Device: {device}", flush=True)
    print(f"[Smoke Test] Model Config: {args.config}", flush=True)

    # 1. Load 2 Real Training Pairs
    dataset = KLASemiconductorDataset(
        mode="paired",
        gt_dir=args.gt_dir,
        noisy_dir=args.noisy_dir,
        scale_factor=2.0
    )
    print(f"[Smoke Test] Dataset Total Paired Samples: {len(dataset)}", flush=True)

    sample0 = dataset[0]
    sample1 = dataset[1]

    noisy_batch = torch.stack([sample0["degraded"], sample1["degraded"]]).to(device) # [2, 1, 128, 128]
    gt_batch = torch.stack([sample0["clean"], sample1["clean"]]).to(device)          # [2, 1, 256, 256]

    print(f"[Smoke Test] Input NoisyLR Shape:  {list(noisy_batch.shape)}", flush=True)
    print(f"[Smoke Test] Target GT Shape:      {list(gt_batch.shape)}", flush=True)

    # 2. Build Model
    model = MultiHeadRestorationNet.from_config(args.config).to(device)
    param_count = model.count_parameters()
    param_bytes_fp32 = param_count * 4
    param_bytes_fp16 = param_count * 2

    print(f"[Smoke Test] Total Trainable Parameters: {param_count:,}", flush=True)
    print(f"[Smoke Test] FP32 Parameter Footprint:   {param_bytes_fp32 / (1024**2):.2f} MB", flush=True)
    print(f"[Smoke Test] FP16 Parameter Footprint:   {param_bytes_fp16 / (1024**2):.2f} MB", flush=True)

    # 3. Setup Simple L1 Reconstruction Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Perform 10 Optimizer Steps
    model.train()
    loss_history = []
    start_time = time.time()

    print(f"\n--- Running {args.steps} Optimizer Steps ---", flush=True)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        outputs = model(noisy_batch)
        
        restored = outputs["restored"] # [2, 1, 256, 256]
        loss = criterion(restored, gt_batch)
        
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)
        print(f"Step [{step:02d}/{args.steps:02d}] - L1 Reconstruction Loss: {loss_val:.6f}", flush=True)

    elapsed = time.time() - start_time
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = initial_loss - final_loss

    print(f"\n--- Smoke Test Summary ---", flush=True)
    print(f"Initial Loss (Step 1): {initial_loss:.6f}", flush=True)
    print(f"Final Loss (Step 10):  {final_loss:.6f}", flush=True)
    print(f"Absolute Loss Change:  {loss_reduction:+.6f}", flush=True)
    print(f"Elapsed Time:          {elapsed * 1000.0:.2f} ms ({elapsed / args.steps * 1000.0:.2f} ms/step)", flush=True)

    if final_loss < initial_loss:
        print("RESULT: SUCCESS - Loss showed a clear downward trend across 10 optimization steps.", flush=True)
    else:
        print("RESULT: WARNING - Loss did not decrease strictly within 10 steps.", flush=True)

    print("==================================================", flush=True)



if __name__ == "__main__":
    run_smoke_test()
