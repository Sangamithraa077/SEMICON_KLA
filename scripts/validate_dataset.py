"""Dataset Audit and Validation Script for KLA Semiconductor Restoration.

Audits dataset files and prints:
- sample counts
- image dimensions
- channel count
- min/max/mean/std statistics
- corrupted file list
- duplicate file warnings
- train/val/test split counts
"""

import os
import sys
import argparse
import hashlib
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.data.dataset import get_train_val_split, load_file_array, SUPPORTED_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and Audit KLA Semiconductor Dataset")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory containing GT clean images")
    parser.add_argument("--noisy_dir", type=str, default=None, help="Directory containing NoisyLR degraded images")
    parser.add_argument("--test_noisy_dir", type=str, default=None, help="Directory containing Test NoisyLR images")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation dataset split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return parser.parse_args()

def inspect_directory(dir_path: str, label: str):
    print(f"\n================ AUDITING {label} ({dir_path}) ================")
    if not dir_path or not os.path.exists(dir_path):
        print(f"[{label}] Warning: Directory path does not exist or was not specified.")
        return [], 0

    files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)])
    print(f"[{label}] Total files found: {len(files)}")

    if not files:
        return files, 0

    shapes = set()
    channels = set()
    corrupted = []
    hashes = {}
    global_min, global_max = float("inf"), float("-inf")
    means, stds = [], []

    for idx, fname in enumerate(files):
        fpath = os.path.join(dir_path, fname)
        try:
            arr = load_file_array(fpath)
            if np.isnan(arr).any() or np.isinf(arr).any():
                corrupted.append((fname, "Contains NaN or Inf"))
                continue

            shapes.add(arr.shape[:2])
            c = 1 if arr.ndim == 2 else arr.shape[2]
            channels.add(c)

            global_min = min(global_min, float(arr.min()))
            global_max = max(global_max, float(arr.max()))
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))

            h = hashlib.md5(arr.tobytes()).hexdigest()
            hashes.setdefault(h, []).append(fname)
        except Exception as e:
            corrupted.append((fname, str(e)))

    avg_mean = float(np.mean(means)) if means else 0.0
    avg_std = float(np.mean(stds)) if stds else 0.0
    dup_count = sum(len(v) - 1 for v in hashes.values() if len(v) > 1)

    print(f"[{label}] Image Dimensions (H, W): {shapes}")
    print(f"[{label}] Channel Count: {channels}")
    print(f"[{label}] Min Value: {global_min:.4f}")
    print(f"[{label}] Max Value: {global_max:.4f}")
    print(f"[{label}] Mean Value: {avg_mean:.4f}")
    print(f"[{label}] Std Dev: {avg_std:.4f}")
    print(f"[{label}] Corrupted Files Count: {len(corrupted)}")

    if corrupted:
        print(f"[{label}] Corrupted Files: {corrupted[:5]}")
    if dup_count > 0:
        print(f"[{label}] Duplicate Image Warning: {dup_count} exact content duplicate files detected.")
    else:
        print(f"[{label}] Duplicate Image Status: 0 duplicate files found.")

    return files, len(corrupted)

def main():
    args = parse_args()

    gt_dir = args.gt_dir
    noisy_dir = args.noisy_dir
    test_noisy_dir = args.test_noisy_dir

    if os.path.exists(args.config):
        cfg = load_config(args.config).get("data", {})
        gt_dir = gt_dir or cfg.get("train_gt_dir")
        noisy_dir = noisy_dir or cfg.get("train_noisy_dir")
        test_noisy_dir = test_noisy_dir or cfg.get("test_noisy_dir")

    gt_files, gt_corrupt = inspect_directory(gt_dir, "TRAIN GT")
    noisy_files, noisy_corrupt = inspect_directory(noisy_dir, "TRAIN NOISY_LR")
    test_files, test_corrupt = inspect_directory(test_noisy_dir, "TEST NOISY_LR")

    if gt_dir and os.path.exists(gt_dir):
        train_files, val_files = get_train_val_split(gt_dir, val_ratio=args.val_ratio, seed=args.seed)
        overlap = set(train_files).intersection(set(val_files))

        print("\n================ DATASET SPLIT SUMMARY ================")
        print(f"Total Train-Val Source Samples: {len(gt_files)}")
        print(f"Training Split Count ({1.0 - args.val_ratio:.0%}): {len(train_files)}")
        print(f"Validation Split Count ({args.val_ratio:.0%}): {len(val_files)}")
        print(f"Test Split Count: {len(test_files)}")
        print(f"Train/Val Overlap Count: {len(overlap)} (Must be 0)")

if __name__ == "__main__":
    main()
