import unittest
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from heatmap_metrics import HeatmapMetricAccumulator, chamfer_distance_m


class HeatmapMetricTests(unittest.TestCase):
    def _metrics(self, pred, target, threshold=0.5):
        acc = HeatmapMetricAccumulator(threshold=threshold, metric_grid_size=None, x_cell_size_m=0.5, y_cell_size_m=0.5)
        acc.update(torch.tensor(pred, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), from_logits=False)
        return acc.compute()

    def test_perfect_prediction(self):
        target = np.array([[[[0, 1], [1, 0]]]], dtype=np.float32)
        metrics = self._metrics(target, target)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["brier_score"], 0.0)
        self.assertEqual(metrics["pixel_mae"], 0.0)

    def test_completely_incorrect_prediction(self):
        pred = np.array([[[[1, 0], [0, 1]]]], dtype=np.float32)
        target = 1.0 - pred
        metrics = self._metrics(pred, target)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["specificity"], 0.0)

    def test_partial_overlap(self):
        pred = np.array([[[[1, 1], [0, 0]]]], dtype=np.float32)
        target = np.array([[[[1, 0], [1, 0]]]], dtype=np.float32)
        metrics = self._metrics(pred, target)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_both_masks_empty(self):
        pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
        target = np.zeros((1, 1, 2, 2), dtype=np.float32)
        metrics = self._metrics(pred, target)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["chamfer_distance_m"], 0.0)
        self.assertEqual(metrics["empty_mask_mismatch_rate"], 0.0)

    def test_prediction_empty_target_nonempty(self):
        pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
        target = np.array([[[[1, 0], [0, 0]]]], dtype=np.float32)
        metrics = self._metrics(pred, target)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["empty_mask_mismatch_rate"], 1.0)

    def test_target_empty_prediction_nonempty(self):
        pred = np.array([[[[1, 0], [0, 0]]]], dtype=np.float32)
        target = np.zeros((1, 1, 2, 2), dtype=np.float32)
        metrics = self._metrics(pred, target)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["empty_mask_mismatch_rate"], 1.0)

    def test_one_cell_shift_chamfer(self):
        pred = np.zeros((3, 3), dtype=bool)
        target = np.zeros((3, 3), dtype=bool)
        pred[1, 1] = True
        target[1, 2] = True
        chamfer, mismatch = chamfer_distance_m(pred, target, x_cell_size_m=0.5, y_cell_size_m=0.25)
        self.assertFalse(mismatch)
        self.assertAlmostEqual(chamfer, 0.25)

    def test_probability_brier_and_mae(self):
        pred = np.array([[[[0.25, 0.75]]]], dtype=np.float32)
        target = np.array([[[[0.0, 1.0]]]], dtype=np.float32)
        metrics = self._metrics(pred, target)
        self.assertAlmostEqual(metrics["brier_score"], 0.0625)
        self.assertAlmostEqual(metrics["pixel_mae"], 0.25)

    def test_bhw_and_bchw_shapes(self):
        pred = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
        target = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
        metrics_bhw = self._metrics(pred, target)
        metrics_bchw = self._metrics(pred[:, None], target[:, None])
        self.assertEqual(metrics_bhw["f1"], metrics_bchw["f1"])


if __name__ == "__main__":
    unittest.main()
