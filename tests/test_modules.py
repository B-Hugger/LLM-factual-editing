#!/usr/bin/env python3
"""
Unit tests for the stabilized model editing modules.
These tests don't require loading the full Llama model.
"""

import sys
import os
from pathlib import Path
import unittest
import torch

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Add paths
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "memit"))

# Change to memit directory for globals.yml to be found
os.chdir(PROJECT_ROOT / "memit")


class TestWeightUpdate(unittest.TestCase):
    """Test WeightUpdate class functionality."""

    def test_weight_update_creation(self):
        """Test creating a WeightUpdate object."""
        from src.editing.weight_update import WeightUpdate

        deltas = {
            "layer1.weight": torch.randn(100, 100),
            "layer2.weight": torch.randn(100, 100),
        }

        update = WeightUpdate(
            deltas=deltas,
            method="rome",
            request_info={"test": True}
        )

        self.assertEqual(len(update.deltas), 2)
        self.assertEqual(update.method, "rome")
        self.assertGreater(update.total_norm, 0)

    def test_weight_update_scaling(self):
        """Test scaling a WeightUpdate."""
        from src.editing.weight_update import WeightUpdate

        deltas = {
            "layer.weight": torch.ones(10, 10),
        }

        update = WeightUpdate(deltas=deltas, method="test")
        scaled = update.scale(0.5)

        self.assertTrue(torch.allclose(scaled.deltas["layer.weight"], torch.ones(10, 10) * 0.5))

    def test_weight_update_to_device(self):
        """Test moving WeightUpdate to device."""
        from src.editing.weight_update import WeightUpdate

        deltas = {"layer.weight": torch.randn(10, 10)}
        update = WeightUpdate(deltas=deltas, method="test")

        moved = update.to("cpu")
        self.assertEqual(moved.deltas["layer.weight"].device.type, "cpu")


class TestDriftMetrics(unittest.TestCase):
    """Test drift measurement functionality."""

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        from src.drift.metrics import compute_kl_divergence

        # Same distributions should have zero KL
        logits = torch.randn(1, 1000)
        kl = compute_kl_divergence(logits, logits)
        self.assertAlmostEqual(kl.item(), 0.0, places=5)

        # Different distributions should have positive KL
        logits1 = torch.zeros(1, 1000)
        logits1[0, 0] = 10  # Peaked distribution
        logits2 = torch.zeros(1, 1000)  # Uniform-ish
        kl = compute_kl_divergence(logits1, logits2)
        self.assertGreater(kl.item(), 0)

    def test_drift_metrics_dataclass(self):
        """Test DriftMetrics dataclass."""
        from src.drift.metrics import DriftMetrics

        metrics = DriftMetrics(
            kl_divergences=[0.1, 0.2, 0.3],
            mean_kl=0.2,
            max_kl=0.3
        )

        self.assertEqual(len(metrics.kl_divergences), 3)
        self.assertEqual(metrics.mean_kl, 0.2)


class TestBaselineManager(unittest.TestCase):
    """Test baseline prompts manager."""

    def test_default_prompts(self):
        """Test that default prompts are loaded."""
        from src.drift.baseline import BaselinePromptsManager, DEFAULT_BASELINE_PROMPTS

        manager = BaselinePromptsManager()

        self.assertEqual(len(manager.prompts), len(DEFAULT_BASELINE_PROMPTS))
        self.assertFalse(manager.has_baselines())

    def test_add_prompt(self):
        """Test adding custom prompts."""
        from src.drift.baseline import BaselinePromptsManager

        manager = BaselinePromptsManager(prompts=["Test prompt 1"])
        manager.add_prompt("Test prompt 2")

        self.assertEqual(len(manager.prompts), 2)


class TestDriftController(unittest.TestCase):
    """Test drift controller functionality."""

    def test_controller_initialization(self):
        """Test DriftController initialization."""
        from src.drift.controller import DriftController
        from src.drift.baseline import BaselinePromptsManager

        manager = BaselinePromptsManager()
        controller = DriftController(
            baseline_manager=manager,
            drift_threshold=0.1,
            device="cpu"
        )

        self.assertEqual(controller.drift_threshold, 0.1)
        self.assertEqual(controller.device, "cpu")


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""

    def test_plot_drift_comparison(self):
        """Test drift comparison plotting."""
        from src.evaluation.visualization import plot_drift_comparison
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        results = {
            "baseline_results": {
                "alpha": 1.0,
                "drift": {"mean_kl": 0.15, "max_kl": 0.3}
            },
            "stabilized_results": {
                "alpha": 0.7,
                "drift": {"mean_kl": 0.08, "max_kl": 0.15},
                "search_history": [
                    {"iteration": 0, "alpha": 0.5, "mean_kl": 0.1},
                    {"iteration": 1, "alpha": 0.75, "mean_kl": 0.12},
                ]
            }
        }

        fig = plot_drift_comparison(results)
        self.assertIsNotNone(fig)


class TestHyperparamsLoading(unittest.TestCase):
    """Test hyperparameter loading."""

    def test_rome_hparams(self):
        """Test loading ROME hyperparameters."""
        from rome.rome_hparams import ROMEHyperParams

        hparams_path = Path(__file__).parent.parent / "memit" / "hparams" / "ROME" / "Llama-2-7b.json"

        if hparams_path.exists():
            hparams = ROMEHyperParams.from_json(hparams_path)
            self.assertIn(5, hparams.layers)
            self.assertIn("down_proj", hparams.rewrite_module_tmp)

    def test_memit_hparams(self):
        """Test loading MEMIT hyperparameters."""
        from memit.memit_hparams import MEMITHyperParams

        hparams_path = Path(__file__).parent.parent / "memit" / "hparams" / "MEMIT" / "Llama-2-7b.json"

        if hparams_path.exists():
            hparams = MEMITHyperParams.from_json(hparams_path)
            self.assertGreater(len(hparams.layers), 1)
            self.assertIn("down_proj", hparams.rewrite_module_tmp)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MODULE TESTS")
    print("=" * 60)

    unittest.main(verbosity=2)
