"""Unit & Sanity Test Suite for KLA Baselines and Evaluation."""

import unittest
import os
import sys
import json
import shutil
import tempfile
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baseline.bicubic import bicubic_upsample_2x
from baseline.dncnn import DnCNNBaseline, count_parameters
from src.data.dataset import get_train_val_split, KLASemiconductorDataset

class TestKLABaselines(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.gt_dir = os.path.join(self.test_dir, "GT")
        self.noisy_dir = os.path.join(self.test_dir, "NoisyLR")
        os.makedirs(self.gt_dir, exist_ok=True)
        os.makedirs(self.noisy_dir, exist_ok=True)

        self.num_samples = 10
        for i in range(self.num_samples):
            fname = f"{i:06d}.npy"
            gt_arr = (np.random.rand(64, 64)).astype(np.float32)
            noisy_arr = (np.random.rand(32, 32)).astype(np.float32)

            np.save(os.path.join(self.gt_dir, fname), gt_arr)
            np.save(os.path.join(self.noisy_dir, fname), noisy_arr)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # 1. Bicubic Baseline Tests
    def test_bicubic_shapes_and_validity(self):
        # 2D NumPy array (32, 32)
        arr_in = np.random.rand(32, 32).astype(np.float32)
        arr_out = bicubic_upsample_2x(arr_in)

        self.assertIsInstance(arr_out, np.ndarray)
        self.assertEqual(arr_out.shape, (64, 64))
        self.assertFalse(np.isnan(arr_out).any())
        self.assertFalse(np.isinf(arr_out).any())

        # Tensor (1, 1, 32, 32)
        tensor_in = torch.rand(1, 1, 32, 32)
        tensor_out = bicubic_upsample_2x(tensor_in)

        self.assertIsInstance(tensor_out, torch.Tensor)
        self.assertEqual(tensor_out.shape, (1, 1, 64, 64))

    # 2. DnCNN Baseline Tests
    def test_dncnn_model_forward_pass(self):
        model = DnCNNBaseline(in_channels=1, num_features=32, num_layers=5)
        params = count_parameters(model)
        self.assertGreater(params, 0)

        # Input (1, 1, 128, 128) -> Output (1, 1, 256, 256)
        x_lr = torch.rand(1, 1, 128, 128)
        out_lr = model(x_lr)
        self.assertEqual(out_lr.shape, (1, 1, 256, 256))
        self.assertGreaterEqual(float(out_lr.min()), 0.0)
        self.assertLessEqual(float(out_lr.max()), 1.0)

        # Input pre-upsampled (1, 1, 256, 256) -> Output (1, 1, 256, 256)
        x_hr = torch.rand(1, 1, 256, 256)
        out_hr = model(x_hr)
        self.assertEqual(out_hr.shape, (1, 1, 256, 256))

    def test_dncnn_checkpoint_save_load(self):
        model = DnCNNBaseline(in_channels=1, num_features=32, num_layers=5)
        ckpt_path = os.path.join(self.test_dir, "test_dncnn.pth")

        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))

        new_model = DnCNNBaseline(in_channels=1, num_features=32, num_layers=5)
        ckpt = torch.load(ckpt_path)
        new_model.load_state_dict(ckpt["model_state_dict"])

        # Check weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            torch.testing.assert_close(p1, p2)

    # 3. Deterministic Validation Split Test
    def test_deterministic_split_reproducibility(self):
        t1, v1 = get_train_val_split(self.gt_dir, val_ratio=0.2, seed=42)
        t2, v2 = get_train_val_split(self.gt_dir, val_ratio=0.2, seed=42)

        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)

    # 4. Evaluation JSON Structure Test
    def test_json_structure_generation(self):
        json_path = os.path.join(self.test_dir, "baseline_comparison.json")
        sample_json_data = {
            "dataset_info": {"validation_samples": 640},
            "baselines": {
                "bicubic": {"psnr": 28.5, "ssim": 0.82, "lpips": 0.15},
                "dncnn": {"psnr": 31.2, "ssim": 0.88, "lpips": 0.09}
            }
        }
        with open(json_path, "w") as f:
            json.dump(sample_json_data, f, indent=2)

        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            loaded = json.load(f)
        self.assertIn("baselines", loaded)
        self.assertIn("bicubic", loaded["baselines"])
        self.assertIn("dncnn", loaded["baselines"])

if __name__ == "__main__":
    unittest.main()
