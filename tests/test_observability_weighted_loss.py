from dataclasses import asdict
import json
import unittest

import torch
import torch.nn.functional as F

from models.two_stage_reconstruction_head import (
    CoarseLossConfig,
    MaskedBEVReconstructionLoss,
    ObservabilityWeightingConfig,
    build_configs,
    coarse_reconstruction_metrics,
    occupancy_bce_weights,
)


def _outputs(raw, mask, faulty=None):
    faulty = torch.zeros_like(raw) if faulty is None else faulty
    replacement = torch.cat(
        (torch.sigmoid(raw[:, 0:1]), raw[:, 1:]), dim=1
    )
    return {
        "replacement_raw": raw,
        "replacement_bev": replacement,
        "occupancy_logits": raw[:, 0:1],
        "coarse_lidar_bev": (1.0 - mask) * faulty + mask * replacement,
        "reconstruction_mask": mask,
    }


def _loss(enabled, min_empty_weight=0.1):
    return MaskedBEVReconstructionLoss(
        CoarseLossConfig(
            observability_weighting=ObservabilityWeightingConfig(
                enabled=enabled,
                min_empty_weight=min_empty_weight,
            )
        )
    )


class ObservabilityWeightedLossTests(unittest.TestCase):
    def test_nested_loss_configuration_is_json_serializable(self):
        config = CoarseLossConfig(
            observability_weighting=ObservabilityWeightingConfig(
                enabled=False,
                min_empty_weight=0.0,
            )
        )

        serialized = json.loads(json.dumps(asdict(config)))

        self.assertFalse(serialized["observability_weighting"]["enabled"])
        self.assertEqual(
            serialized["observability_weighting"]["min_empty_weight"],
            0.0,
        )

    def test_occupied_cells_always_have_unit_weight(self):
        target = torch.ones(1, 1, 1, 3)
        confidence = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        weights = occupancy_bce_weights(target, confidence, 0.1)

        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))

    def test_positive_occupancy_weight_biases_only_occupied_cells(self):
        target = torch.tensor([[[[1.0, 0.0]]]])
        confidence = torch.ones_like(target)
        weights = occupancy_bce_weights(
            target,
            confidence,
            0.1,
            positive_occupancy_weight=1.1,
        )

        self.assertTrue(
            torch.allclose(weights, torch.tensor([[[[1.1, 1.0]]]]))
        )

    def test_fully_observable_empty_cell_matches_ordinary_bce(self):
        raw = torch.tensor([[[[0.7]], [[0.0]], [[0.0]]]], requires_grad=True)
        clean = torch.zeros_like(raw)
        mask = torch.ones(1, 1, 1, 1)
        confidence = torch.ones_like(mask)
        weighted = _loss(True)(_outputs(raw, mask), clean, confidence)
        ordinary = _loss(False)(_outputs(raw, mask), clean)

        self.assertEqual(
            occupancy_bce_weights(clean[:, 0:1], confidence, 0.1).item(),
            1.0,
        )
        self.assertTrue(
            torch.allclose(
                weighted["loss_occupancy_bce"],
                ordinary["loss_occupancy_bce"],
            )
        )

    def test_unobserved_empty_cell_has_minimum_weight(self):
        target = torch.zeros(1, 1, 1, 1)
        confidence = torch.zeros_like(target)

        self.assertAlmostEqual(
            occupancy_bce_weights(target, confidence, 0.1).item(), 0.1
        )

    def test_intermediate_empty_observability_has_interpolated_weight(self):
        target = torch.zeros(1, 1, 1, 1)
        confidence = torch.full_like(target, 0.5)

        self.assertAlmostEqual(
            occupancy_bce_weights(target, confidence, 0.1).item(), 0.55
        )

    def test_weighted_bce_uses_sum_of_valid_weights_for_normalization(self):
        raw = torch.zeros(1, 3, 1, 2, requires_grad=True)
        raw.data[:, 0] = torch.tensor([[0.0, 2.0]])
        clean = torch.zeros_like(raw)
        mask = torch.ones(1, 1, 1, 2)
        confidence = torch.tensor([[[[0.0, 1.0]]]])
        actual = _loss(True)(_outputs(raw, mask), clean, confidence)[
            "loss_occupancy_bce"
        ]
        per_cell = F.binary_cross_entropy_with_logits(
            raw[:, 0:1], clean[:, 0:1], reduction="none"
        )
        expected = (per_cell[..., 0] * 0.1 + per_cell[..., 1]).sum() / 1.1

        self.assertTrue(torch.allclose(actual, expected))

    def test_observability_outside_reconstruction_mask_has_no_effect(self):
        raw = torch.zeros(1, 3, 1, 2, requires_grad=True)
        clean = torch.zeros_like(raw)
        mask = torch.tensor([[[[1.0, 0.0]]]])
        first = torch.tensor([[[[0.5, 0.0]]]])
        second = torch.tensor([[[[0.5, 1.0]]]])

        loss_a = _loss(True)(_outputs(raw, mask), clean, first)["loss"]
        loss_b = _loss(True)(_outputs(raw, mask), clean, second)["loss"]
        self.assertTrue(torch.equal(loss_a, loss_b))

    def test_dice_density_and_height_are_identical_to_baseline(self):
        raw = torch.tensor(
            [[[[0.2, -0.4]], [[0.1, 0.8]], [[0.7, 0.2]]]],
            requires_grad=True,
        )
        clean = torch.tensor(
            [[[[1.0, 0.0]], [[0.5, 0.0]], [[0.25, 0.0]]]]
        )
        mask = torch.ones(1, 1, 1, 2)
        confidence = torch.tensor([[[[0.0, 0.4]]]])
        baseline = _loss(False)(_outputs(raw, mask), clean)
        weighted = _loss(True)(_outputs(raw, mask), clean, confidence)

        for name in ("loss_occupancy_dice", "loss_density", "loss_height"):
            self.assertTrue(torch.equal(baseline[name], weighted[name]))

    def test_disabled_mode_exactly_reproduces_baseline_with_or_without_map(self):
        torch.manual_seed(4)
        raw = torch.randn(2, 3, 3, 4, requires_grad=True)
        clean = torch.rand_like(raw)
        clean[:, 0] = (clean[:, 0] > 0.5).float()
        mask = torch.ones(2, 1, 3, 4)
        confidence = torch.rand_like(mask)
        loss_fn = _loss(False)
        without = loss_fn(_outputs(raw, mask), clean)
        with_map = loss_fn(_outputs(raw, mask), clean, confidence)

        for name in (
            "loss",
            "loss_occupancy",
            "loss_occupancy_bce",
            "loss_occupancy_dice",
            "loss_density",
            "loss_height",
        ):
            self.assertTrue(torch.equal(without[name], with_map[name]))

    def test_enabled_mode_requires_observability_map(self):
        raw = torch.zeros(1, 3, 2, 2, requires_grad=True)
        clean = torch.zeros_like(raw)
        mask = torch.ones(1, 1, 2, 2)

        with self.assertRaisesRegex(ValueError, "does not contain observability_confidence"):
            _loss(True)(_outputs(raw, mask), clean)

    def test_invalid_observability_values_and_shapes_are_rejected(self):
        raw = torch.zeros(1, 3, 2, 2, requires_grad=True)
        clean = torch.zeros_like(raw)
        mask = torch.ones(1, 1, 2, 2)
        invalid_maps = (
            torch.full_like(mask, torch.nan),
            torch.full_like(mask, torch.inf),
            torch.full_like(mask, -0.01),
            torch.full_like(mask, 1.01),
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 1, 3, 2),
        )
        for confidence in invalid_maps:
            with self.subTest(shape=tuple(confidence.shape), value=str(confidence.flatten()[0])):
                with self.assertRaises((ValueError, TypeError)):
                    _loss(True)(_outputs(raw, mask), clean, confidence)

    def test_total_loss_backpropagates_when_weighting_is_enabled(self):
        raw = torch.zeros(1, 3, 2, 2, requires_grad=True)
        clean = torch.ones_like(raw)
        clean[:, 0, 0, 0] = 0.0
        mask = torch.ones(1, 1, 2, 2)
        confidence = torch.full_like(mask, 0.5)
        losses = _loss(True)(_outputs(raw, mask), clean, confidence)

        self.assertTrue(losses["loss"].requires_grad)
        losses["loss"].backward()
        self.assertGreater(raw.grad.abs().sum().item(), 0.0)

    def test_diagnostics_and_stratified_hallucination_rates(self):
        logits = torch.tensor([[[[2.0, 2.0, -2.0]]]])
        raw = torch.cat((logits, torch.zeros(1, 2, 1, 3)), dim=1)
        clean = torch.zeros_like(raw)
        mask = torch.ones(1, 1, 1, 3)
        confidence = torch.tensor([[[[0.1, 0.5, 0.9]]]])
        losses = _loss(True)(_outputs(raw, mask), clean, confidence)
        metrics = coarse_reconstruction_metrics(
            _outputs(raw, mask),
            torch.zeros_like(raw),
            clean,
            observability_confidence=confidence,
        )

        self.assertAlmostEqual(losses["mean_observability_repair"].item(), 0.5)
        self.assertEqual(metrics["hallucination_rate_low_observability"].item(), 1.0)
        self.assertEqual(metrics["hallucination_rate_medium_observability"].item(), 1.0)
        self.assertEqual(metrics["hallucination_rate_high_observability"].item(), 0.0)

    def test_nested_configuration_is_validated(self):
        _model, loss_config, _selector = build_configs(
            {
                "coarse_reconstruction": {
                    "loss": {
                        "observability_weighting": {
                            "enabled": True,
                            "min_empty_weight": 0.2,
                        }
                    }
                }
            }
        )
        self.assertTrue(loss_config.observability_weighting.enabled)
        self.assertEqual(loss_config.observability_weighting.min_empty_weight, 0.2)
        _model, recall_biased, _selector = build_configs(
            {
                "coarse_reconstruction": {
                    "loss": {"positive_occupancy_weight": 1.1}
                }
            }
        )
        self.assertEqual(recall_biased.positive_occupancy_weight, 1.1)
        with self.assertRaisesRegex(ValueError, "must be a bool"):
            build_configs(
                {
                    "coarse_reconstruction": {
                        "loss": {
                            "observability_weighting": {"enabled": "yes"}
                        }
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
