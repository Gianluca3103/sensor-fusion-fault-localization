import unittest

import torch

from models.two_stage_reconstruction_head.diffusion_process.evaluate_fine_diffusion_by_fault import (
    _diffusion_config_from_checkpoint,
    _occupancy_transition_counts,
)


class FineDiffusionEvaluationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
