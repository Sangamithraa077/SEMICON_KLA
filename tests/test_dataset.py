import unittest
import os
import sys
import shutil
import tempfile
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.degradation import generate_poisson_gaussian_degradation
from src.data.dataset import KLASemiconductorDataset, get_train_val_split
from src.data.dataloader import build_dataloaders


class TestKLADataPipeline(unittest.TestCase):
    def setUp(self):
        # Create temporary dummy .npy dataset directory
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

    def test_poisson_gaussian_degradation_output_keys(self):
        clean = (np.random.rand(64, 64)).astype(np.float32)
        degraded, clean_out, params = generate_poisson_gaussian_degradation(clean, seed=42)

        self.assertEqual(degraded.shape, (64, 64))
        self.assertEqual(clean_out.shape, (64, 64))
        self.assertIn("poisson_scale", params)
        self.assertIn("gaussian_std", params)
        self.assertIn("blur_ksize", params)
        self.assertIn("downsample_scale", params)

    def test_synthetic_degradation_determinism(self):
        clean = (np.random.rand(64, 64)).astype(np.float32)
        deg1, _, params1 = generate_poisson_gaussian_degradation(clean, seed=12345)
        deg2, _, params2 = generate_poisson_gaussian_degradation(clean, seed=12345)

        np.testing.assert_array_almost_equal(deg1, deg2)
        self.assertEqual(params1, params2)

    def test_train_val_split_non_overlap(self):
        train_files, val_files = get_train_val_split(self.gt_dir, val_ratio=0.2, seed=42)

        self.assertEqual(len(train_files) + len(val_files), self.num_samples)
        overlap = set(train_files).intersection(set(val_files))
        self.assertEqual(len(overlap), 0)

    def test_paired_dataset_loading(self):
        ds = KLASemiconductorDataset(
            mode="paired",
            gt_dir=self.gt_dir,
            noisy_dir=self.noisy_dir,
            is_train=False,
            scale_factor=2.0
        )
        self.assertEqual(len(ds), self.num_samples)

        sample = ds[0]
        self.assertIn("degraded", sample)
        self.assertIn("clean", sample)
        self.assertIn("degradation_params", sample)
        self.assertIn("filename", sample)

        # Degraded shape: (1, 32, 32), Clean shape: (1, 64, 64)
        self.assertEqual(sample["degraded"].shape, (1, 32, 32))
        self.assertEqual(sample["clean"].shape, (1, 64, 64))
        self.assertEqual(sample["degradation_params"].shape, (4,))

    def test_dataloader_batch_dimensions(self):
        loaders = build_dataloaders(
            mode="paired",
            gt_dir=self.gt_dir,
            noisy_dir=self.noisy_dir,
            batch_size=2,
            num_workers=0,
            val_ratio=0.2,
            crop_size=16,
            scale_factor=2.0
        )

        train_loader = loaders["train"]
        self.assertIsNotNone(train_loader)

        for batch in train_loader:
            self.assertEqual(batch["degraded"].shape[0], 2)
            self.assertEqual(batch["clean"].shape[0], 2)
            self.assertEqual(batch["degradation_params"].shape[0], 2)
            break

if __name__ == "__main__":
    unittest.main()
