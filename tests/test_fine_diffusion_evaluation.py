import unittest

import torch

from models.two_stage_reconstruction_head.diffusion_process.evaluate_fine_diffusion_by_fault import (
    _diffusion_config_from_checkpoint,
    _occupancy_transition_counts,
    _summarize_threshold_records,
    _threshold_sweep_record,
)
from models.two_stage_reconstruction_head.diffusion_process.diffusion_metrics import (
    occupancy_metrics,
    tolerant_metrics_from_counts,
    tolerant_occupancy_counts,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.coarse_loss import (
    _dilate_with_metric_disk,
)


class FineDiffusionEvaluationTests(unittest.TestCase):
    def test_exact_metric_is_unchanged(self):
        prediction = torch.tensor([[[[1.0, 1.0, 0.0]]]])
        target = torch.tensor([[[[1.0, 0.0, 1.0]]]])
        metrics = occupancy_metrics(
            prediction, target, torch.ones_like(target)
        )
        self.assertAlmostEqual(float(metrics["iou"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(metrics["f1"]), 0.5)

    def test_physical_0p2m_tolerance_uses_grid_cell_sizes(self):
        occupied = torch.zeros(1, 1, 5, 5, dtype=torch.bool)
        predicted = torch.zeros_like(occupied)
        occupied[:, :, 2, 2] = True
        predicted[:, :, 2, 3] = True
        valid = torch.ones_like(occupied)
        counts = tolerant_occupancy_counts(
            predicted,
            occupied,
            valid,
            tolerance_m=0.2,
            meters_per_cell_x=0.1,
            meters_per_cell_y=0.2,
        )
        metrics = tolerant_metrics_from_counts(counts)
        self.assertAlmostEqual(metrics["iou"], 1.0, places=6)

    def test_physical_0p5m_matches_previous_square_grid_implementation(self):
        occupied = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        predicted = torch.zeros_like(occupied)
        occupied[:, :, 3, 3] = True
        occupied[:, :, 6, 6] = True
        predicted[:, :, 3, 5] = True
        predicted[:, :, 0, 0] = True
        valid = torch.ones_like(occupied)
        counts = tolerant_occupancy_counts(
            predicted,
            occupied,
            valid,
            tolerance_m=0.5,
            meters_per_cell_x=0.2,
            meters_per_cell_y=0.2,
        )
        old_target_neighborhood = _dilate_with_metric_disk(
            occupied, 0.5, 0.2
        )
        old_prediction_neighborhood = _dilate_with_metric_disk(
            predicted, 0.5, 0.2
        )
        self.assertEqual(
            counts["matched_predictions"],
            float((predicted & old_target_neighborhood).sum()),
        )
        self.assertEqual(
            counts["matched_targets"],
            float((occupied & old_prediction_neighborhood).sum()),
        )
    def test_v10_checkpoint_selects_legacy_current_lidar_input_for_evaluation(self):
        config = _diffusion_config_from_checkpoint(
            {"enabled": True}, {"version": 10}
        )
        self.assertEqual(
            config.transformer_spatial_input_mode, "current_lidar"
        )

    def test_occupancy_transition_categories_are_exact_and_masked(self):
        # Cells: beneficial add, harmful add, beneficial remove, harmful remove,
        # unchanged, and a masked-out beneficial addition.
        clean = torch.tensor([[[[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]])
        coarse = torch.tensor([[[[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]]])
        fine = torch.tensor([[[[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]]]])
        mask = torch.tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]]])

        counts = _occupancy_transition_counts(
            clean, coarse, fine, mask, threshold=0.5
        )

        self.assertEqual(
            counts,
            {
                "beneficial_additions": 1,
                "harmful_additions": 1,
                "beneficial_removals": 1,
                "harmful_removals": 1,
            },
        )

    def test_threshold_sweep_varies_only_fine_decision_threshold(self):
        clean = torch.tensor([[[[1.0, 0.0]]]])
        coarse = torch.tensor([[[[0.6, 0.4]]]])
        fine = torch.tensor([[[[0.4, 0.4]]]])
        mask = torch.ones_like(clean)

        permissive = _threshold_sweep_record(
            clean,
            coarse,
            fine,
            mask,
            fine_threshold=0.3,
            tolerance_m=0.0,
        )
        conservative = _threshold_sweep_record(
            clean,
            coarse,
            fine,
            mask,
            fine_threshold=0.5,
            tolerance_m=0.0,
        )

        self.assertEqual(permissive["coarse_tp"], 1)
        self.assertEqual(conservative["coarse_tp"], 1)
        self.assertEqual(permissive["fine_tp"], 1)
        self.assertEqual(permissive["fine_fp"], 1)
        self.assertEqual(conservative["fine_fn"], 1)

    def test_threshold_summary_uses_global_counts(self):
        clean = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        coarse = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
        fine = clean.clone()
        mask = torch.ones_like(clean)
        record = _threshold_sweep_record(
            clean,
            coarse,
            fine,
            mask,
            fine_threshold=0.5,
            tolerance_m=0.0,
        )

        summary = _summarize_threshold_records([record])

        self.assertAlmostEqual(summary["coarse_exact_iou"], 1.0 / 3.0)
        self.assertEqual(summary["fine_exact_iou"], 1.0)
        self.assertEqual(summary["beneficial_additions"], 1)
        self.assertEqual(summary["beneficial_removals"], 1)
        self.assertEqual(summary["missing_cell_recovery_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
