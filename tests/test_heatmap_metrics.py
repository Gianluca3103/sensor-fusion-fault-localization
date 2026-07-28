import unittest

import numpy as np
import torch


from Fault_Localization_Model.heatmap_metrics import (
    HeatmapMetricAccumulator,
    chamfer_distance_m,
    one_to_one_match_masks,
)


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

    def test_fixed_target_threshold_is_independent_from_prediction_threshold(self):
        pred = np.array([[[[0.2, 0.8]]]], dtype=np.float32)
        target = np.array([[[[0.1, 0.4]]]], dtype=np.float32)

        low = HeatmapMetricAccumulator(
            threshold=0.5,
            target_threshold=0.0,
            metric_grid_size=None,
            compute_chamfer=False,
        )
        high = HeatmapMetricAccumulator(
            threshold=0.9,
            target_threshold=0.0,
            metric_grid_size=None,
            compute_chamfer=False,
        )
        tensor_pred = torch.tensor(pred)
        tensor_target = torch.tensor(target)
        low.update(tensor_pred, tensor_target, from_logits=False)
        high.update(tensor_pred, tensor_target, from_logits=False)

        self.assertEqual(low.compute()["localization_target_total"], 2.0)
        self.assertEqual(high.compute()["localization_target_total"], 2.0)
        self.assertEqual(low.compute()["localization_pred_total"], 1.0)
        self.assertEqual(high.compute()["localization_pred_total"], 0.0)

    def test_degenerate_prediction_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            HeatmapMetricAccumulator(threshold=0.0)
        with self.assertRaises(ValueError):
            HeatmapMetricAccumulator(threshold=1.0)

    def test_metadata_count_must_match_batch(self):
        accumulator = HeatmapMetricAccumulator(
            threshold=0.5,
            metric_grid_size=None,
            compute_chamfer=False,
        )
        values = torch.zeros((2, 1, 2, 2))
        with self.assertRaises(ValueError):
            accumulator.update(
                values,
                values,
                metadata=[{}],
                from_logits=False,
            )

    def test_non_finite_values_are_rejected(self):
        accumulator = HeatmapMetricAccumulator(
            threshold=0.5,
            metric_grid_size=None,
            compute_chamfer=False,
        )
        prediction = torch.tensor([[[[float("nan")]]]])
        target = torch.zeros_like(prediction)
        with self.assertRaises(ValueError):
            accumulator.update(prediction, target, from_logits=False)

    def test_dense_chamfer_uses_metric_distance_without_pairwise_allocation(self):
        pred = np.zeros((320, 320), dtype=bool)
        target = np.zeros((320, 320), dtype=bool)
        pred[:, :160] = True
        target[:, 1:161] = True
        chamfer, mismatch = chamfer_distance_m(
            pred,
            target,
            x_cell_size_m=0.2,
            y_cell_size_m=0.2,
        )
        self.assertFalse(mismatch)
        self.assertGreaterEqual(chamfer, 0.0)
        self.assertLess(chamfer, 0.01)

    def test_localization_matching_does_not_reuse_one_target(self):
        prediction = np.zeros((3, 3), dtype=bool)
        target = np.zeros((3, 3), dtype=bool)
        prediction[1, 0:3] = True
        target[1, 1] = True

        prediction_matched, target_matched = one_to_one_match_masks(
            prediction,
            target,
            x_cell_size_m=0.2,
            y_cell_size_m=0.2,
            tolerance_m=0.2,
        )
        self.assertEqual(int(prediction_matched.sum()), 1)
        self.assertEqual(int(target_matched.sum()), 1)

        accumulator = HeatmapMetricAccumulator(
            threshold=0.5,
            target_threshold=0.0,
            metric_grid_size=None,
            x_cell_size_m=0.2,
            y_cell_size_m=0.2,
            compute_chamfer=False,
            localization_tolerance_m=0.2,
        )
        accumulator.update(
            torch.from_numpy(prediction.astype(np.float32))[None, None],
            torch.from_numpy(target.astype(np.float32))[None, None],
            from_logits=False,
            update_groups=False,
        )
        metrics = accumulator.compute()
        self.assertAlmostEqual(metrics["localization_precision"], 1.0 / 3.0)
        self.assertEqual(metrics["localization_recall"], 1.0)
        self.assertAlmostEqual(metrics["localization_iou"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
