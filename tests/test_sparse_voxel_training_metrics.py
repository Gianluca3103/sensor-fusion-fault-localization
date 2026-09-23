"""Regression checks for sparse diffusion epoch metric aggregation."""

from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from scripts.train_sparse_voxel_diffusion import _run_epoch


class _Batch:
    def to(self, device):
        return self


class _Model:
    def __init__(self):
        self.calls = 0

    def train(self, enabled):
        pass

    def __call__(self, batch, *, compute_metrics):
        self.calls += 1
        value = 0.25 if self.calls == 1 else 0.75
        return {
            "loss": torch.tensor(1.0),
            "diffusion_loss": torch.tensor(0.5),
            "bce_loss": torch.tensor(0.5),
            "chamfer_loss": torch.tensor(0.0),
            "iou_at_0_2m": torch.tensor(value),
            "f1_at_0_2m": torch.tensor(value),
        }


class SparseVoxelTrainingMetricTests(unittest.TestCase):
    def test_validation_metrics_are_counted_once_per_batch(self):
        with patch(
            "scripts.train_sparse_voxel_diffusion._batches",
            return_value=iter((_Batch(), _Batch())),
        ):
            metrics = _run_epoch(
                _Model(), None, [Path("a.npz"), Path("b.npz")],
                device=torch.device("cpu"), batch_size=1,
                loader_kwargs={}, log_every=1, label="test_val",
            )
        self.assertEqual(metrics["batches"], 2)
        self.assertAlmostEqual(metrics["iou_at_0_2m"], 0.5)
        self.assertAlmostEqual(metrics["f1_at_0_2m"], 0.5)


if __name__ == "__main__":
    unittest.main()
