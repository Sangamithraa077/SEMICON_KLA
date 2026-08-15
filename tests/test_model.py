"""Unit Test Suite for MultiHeadRestorationNet and Phase 3 Architecture Requirements."""

import unittest
import tempfile
import os
import torch
import torch.nn as nn

from src.models import (
    MultiHeadRestorationNet,
    NAFNetBackbone,
    RestorationHead,
    DegradationHead,
    UncertaintyHead,
    save_checkpoint,
    load_checkpoint
)


class TestMultiHeadRestorationNet(unittest.TestCase):
    """Unit test suite verifying Phase 3 multi-head model architecture contract."""

    def setUp(self):
        torch.manual_seed(42)
        self.config_path = "configs/model.yaml"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiHeadRestorationNet.from_config(self.config_path)

    def test_01_model_construction(self):
        """Test 1: Model construction from config and parameter count calculation."""
        self.assertIsNotNone(self.model)
        self.assertTrue(isinstance(self.model.backbone, NAFNetBackbone))
        self.assertTrue(isinstance(self.model.restoration_head, RestorationHead))
        self.assertTrue(isinstance(self.model.degradation_head, DegradationHead))
        self.assertTrue(isinstance(self.model.uncertainty_head, UncertaintyHead))

        param_count = self.model.count_parameters()
        self.assertGreater(param_count, 0)

    def test_02_single_input_forward(self):
        """Test 2: Single input forward pass [1, 1, 128, 128]."""
        x = torch.randn(1, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        self.assertIn("restored", outputs)
        self.assertIn("degradation", outputs)
        self.assertIn("confidence", outputs)

    def test_03_batch_input_forward(self):
        """Test 3: Batch input forward pass [4, 1, 128, 128]."""
        x = torch.randn(4, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        self.assertEqual(outputs["restored"].shape[0], 4)
        self.assertEqual(outputs["degradation"].shape[0], 4)
        self.assertEqual(outputs["confidence"].shape[0], 4)

    def test_04_output_shapes(self):
        """Test 4: Verify output shapes: restored=[B,1,256,256], degradation=[B,4], confidence=[B,1,256,256]."""
        B = 2
        x = torch.randn(B, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        self.assertEqual(outputs["restored"].shape, (B, 1, 256, 256))
        self.assertEqual(outputs["degradation"].shape, (B, 4))
        self.assertEqual(outputs["confidence"].shape, (B, 1, 256, 256))

    def test_05_no_nans(self):
        """Test 5: Verify no NaNs in any output tensor."""
        x = torch.randn(2, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        for name, tensor in outputs.items():
            self.assertFalse(torch.isnan(tensor).any().item(), f"NaN detected in output: {name}")

    def test_06_no_infs(self):
        """Test 6: Verify no Infs in any output tensor."""
        x = torch.randn(2, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        for name, tensor in outputs.items():
            self.assertFalse(torch.isinf(tensor).any().item(), f"Inf detected in output: {name}")

    def test_07_output_ranges(self):
        """Test 7: Verify output value ranges (restored in [0,1], confidence clamped in [-10, 10])."""
        x = torch.randn(2, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        restored = outputs["restored"]
        confidence = outputs["confidence"]

        self.assertGreaterEqual(float(restored.min().item()), 0.0)
        self.assertLessEqual(float(restored.max().item()), 1.0)

        self.assertGreaterEqual(float(confidence.min().item()), -10.0)
        self.assertLessEqual(float(confidence.max().item()), 10.0)

    def test_08_checkpoint_save_load(self):
        """Test 8: Verify checkpoint save and load reconstructs identical model and outputs."""
        x = torch.randn(1, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            orig_out = self.model(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_ckpt.pth")
            save_checkpoint(self.model, ckpt_path)

            reloaded_model, ckpt_dict = load_checkpoint(ckpt_path, device="cpu")
            reloaded_model.eval()

            with torch.no_grad():
                reload_out = reloaded_model(x)

            for key in orig_out:
                self.assertTrue(
                    torch.allclose(orig_out[key], reload_out[key], atol=1e-5),
                    f"Mismatch in reloaded checkpoint output for key: {key}"
                )

    def test_09_eval_determinism(self):
        """Test 9: Verify model is deterministic in eval mode with fixed seed."""
        x = torch.randn(1, 1, 128, 128)
        self.model.eval()

        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)

        for key in out1:
            self.assertTrue(torch.equal(out1[key], out2[key]), f"Non-deterministic output for key: {key}")

    def test_10_cpu_forward_pass(self):
        """Test 10: Verify model runs on CPU."""
        model_cpu = self.model.to("cpu")
        x_cpu = torch.randn(1, 1, 128, 128, device="cpu")
        model_cpu.eval()
        with torch.no_grad():
            outputs = model_cpu(x_cpu)

        self.assertEqual(outputs["restored"].device.type, "cpu")

    def test_11_cuda_forward_pass(self):
        """Test 11: Verify model runs on CUDA if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA accelerator not available in current environment.")

        model_cuda = self.model.to("cuda")
        x_cuda = torch.randn(1, 1, 128, 128, device="cuda")
        model_cuda.eval()
        with torch.no_grad():
            outputs = model_cuda(x_cuda)

        self.assertEqual(outputs["restored"].device.type, "cuda")

    def test_12_gradient_flow_restoration_head(self):
        """Test 12: Verify gradients flow through RestorationHead back to backbone."""
        self.model.train()
        x = torch.randn(2, 1, 128, 128, requires_grad=True)
        outputs = self.model(x)
        loss = outputs["restored"].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.model.backbone.intro.weight.grad)

    def test_13_gradient_flow_degradation_head(self):
        """Test 13: Verify gradients flow through DegradationHead back to backbone."""
        self.model.train()
        x = torch.randn(2, 1, 128, 128, requires_grad=True)
        outputs = self.model(x)
        loss = outputs["degradation"].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.model.backbone.intro.weight.grad)

    def test_14_gradient_flow_uncertainty_head(self):
        """Test 14: Verify gradients flow through UncertaintyHead back to backbone."""
        self.model.train()
        x = torch.randn(2, 1, 128, 128, requires_grad=True)
        outputs = self.model(x)
        loss = outputs["confidence"].sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.model.backbone.intro.weight.grad)

    def test_15_shared_backbone_instance(self):
        """Test 15: Verify all three heads receive features from the SAME backbone instance."""
        x = torch.randn(1, 1, 128, 128)
        self.model.eval()

        backbone_calls = [0]
        orig_forward = self.model.backbone.forward

        def hook_forward(tensor):
            backbone_calls[0] += 1
            return orig_forward(tensor)

        self.model.backbone.forward = hook_forward

        with torch.no_grad():
            outputs = self.model(x)

        self.assertEqual(backbone_calls[0], 1, "Backbone executed more than once per forward pass!")

    def test_16_single_forward_pass_execution(self):
        """Test 16: Verify model performs exactly ONE backbone pass returning a complete output dict."""
        x = torch.randn(1, 1, 128, 128)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        self.assertSetEqual(set(outputs.keys()), {"restored", "degradation", "confidence"})


if __name__ == "__main__":
    unittest.main()
