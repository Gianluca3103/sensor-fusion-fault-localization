import unittest

import torch

from models.two_stage_reconstruction_head import (
    CoarseLossConfig,
    MaskedBEVReconstructionLoss,
    OccupancyLossConfig,
    build_configs,
    tolerance_radius_cells,
)


def _outputs(logits: torch.Tensor, mask: torch.Tensor):
    zeros = torch.zeros_like(logits)
    raw = torch.cat((logits, zeros, zeros), dim=1)
    replacement = torch.cat((torch.sigmoid(logits), zeros, zeros), dim=1)
    return {
        "replacement_raw": raw,
        "replacement_bev": replacement,
        "occupancy_logits": logits,
        "coarse_lidar_bev": mask * replacement,
        "reconstruction_mask": mask,
    }


def _clean(target: torch.Tensor):
    zeros = torch.zeros_like(target)
    return torch.cat((target, zeros, zeros), dim=1)


def _loss(**overrides):
    settings = {
        "type": "tolerance_aware",
        "exact_weight": 0.25,
        "tolerant_recall_weight": 1.0,
        "far_fp_weight": 0.5,
        "tolerance_radius_m": 0.5,
    }
    settings.update(overrides)
    return MaskedBEVReconstructionLoss(
        CoarseLossConfig(occupancy=OccupancyLossConfig(**settings)),
        bev_resolution_m=0.2,
    )


class ToleranceAwareOccupancyLossTests(unittest.TestCase):
    def test_meter_to_cell_conversion(self):
        self.assertEqual(tolerance_radius_cells(0.5, 0.2), 2)
        self.assertEqual(2 * tolerance_radius_cells(0.5, 0.2) + 1, 5)

    def test_config_defaults_to_existing_and_accepts_tolerance_mode(self):
        _, baseline, _ = build_configs({})
        self.assertEqual(baseline.occupancy.type, "existing")
        _, configured, _ = build_configs(
            {
                "coarse_reconstruction": {
                    "loss": {
                        "positive_occupancy_weight": 1.1,
                        "occupancy": {
                            "type": "tolerance_aware",
                            "exact_weight": 0.35,
                            "tolerant_recall_weight": 0.6,
                            "far_fp_weight": 1.0,
                            "tolerance_radius_m": 0.5,
                        },
                    }
                }
            }
        )
        self.assertEqual(configured.occupancy.type, "tolerance_aware")
        self.assertEqual(configured.positive_occupancy_weight, 1.1)

    def test_existing_mode_is_numerically_unchanged(self):
        torch.manual_seed(8)
        logits = torch.randn(2, 1, 5, 6, requires_grad=True)
        target = (torch.rand_like(logits) > 0.7).float()
        mask = (torch.rand_like(logits) > 0.2).float()
        losses = MaskedBEVReconstructionLoss(CoarseLossConfig())(
            _outputs(logits, mask), _clean(target)
        )
        self.assertTrue(
            torch.equal(
                losses["loss_occupancy"],
                losses["loss_occupancy_bce"]
                + losses["loss_occupancy_dice"],
            )
        )
        self.assertEqual(losses["loss_occupancy_tolerant_recall"].item(), 0.0)
        self.assertEqual(losses["loss_occupancy_far_fp"].item(), 0.0)

    def test_near_prediction_is_rewarded_but_far_prediction_is_not(self):
        target = torch.zeros(1, 1, 11, 11)
        target[..., 5, 5] = 1.0
        mask = torch.ones_like(target)
        near = torch.full_like(target, -10.0)
        near[..., 5, 7] = 10.0  # 0.4 m
        far = torch.full_like(target, -10.0)
        far[..., 5, 9] = 10.0  # 0.8 m

        near_loss = _loss()(_outputs(near, mask), _clean(target))
        far_loss = _loss()(_outputs(far, mask), _clean(target))
        self.assertLess(near_loss["loss_occupancy_tolerant_recall"], 1.0e-3)
        self.assertGreater(far_loss["loss_occupancy_tolerant_recall"], 5.0)
        self.assertLess(near_loss["loss_occupancy_far_fp"], 1.0e-3)
        self.assertGreater(
            far_loss["loss_occupancy_far_fp"],
            100.0 * near_loss["loss_occupancy_far_fp"],
        )

    def test_tolerant_terms_have_gradients(self):
        target = torch.zeros(1, 1, 9, 9)
        target[..., 4, 4] = 1.0
        mask = torch.ones_like(target)
        for weights in (
            {"exact_weight": 0.0, "far_fp_weight": 0.0},
            {"exact_weight": 0.0, "tolerant_recall_weight": 0.0},
        ):
            logits = torch.zeros_like(target, requires_grad=True)
            losses = _loss(**weights)(_outputs(logits, mask), _clean(target))
            losses["loss"].backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_degenerate_targets_are_finite(self):
        mask = torch.ones(1, 1, 3, 3)
        logits = torch.zeros_like(mask, requires_grad=True)
        empty = _loss()(_outputs(logits, mask), _clean(torch.zeros_like(mask)))
        full = _loss()(_outputs(logits, mask), _clean(torch.ones_like(mask)))
        self.assertTrue(torch.isfinite(empty["loss"]))
        self.assertTrue(torch.isfinite(full["loss"]))
        self.assertEqual(empty["loss_occupancy_tolerant_recall"].item(), 0.0)
        self.assertEqual(full["loss_occupancy_far_fp"].item(), 0.0)

    def test_outside_mask_prediction_cannot_satisfy_recall(self):
        target = torch.zeros(1, 1, 7, 7)
        target[..., 3, 3] = 1.0
        mask = torch.zeros_like(target)
        mask[..., 3, 3] = 1.0
        logits = torch.full_like(target, -10.0)
        logits[..., 3, 4] = 10.0
        losses = _loss()(_outputs(logits, mask), _clean(target))
        self.assertGreater(losses["loss_occupancy_tolerant_recall"], 5.0)


if __name__ == "__main__":
    unittest.main()
