import unittest

import torch

from models.reconstruction_head import (
    CoarseLossConfig,
    MaskedBEVReconstructionLoss,
    build_configs,
    coarse_reconstruction_metrics,
)


def _raw(occupancy_logits, density, height):
    return torch.cat((occupancy_logits, density, height), dim=1)


def _model_outputs(replacement_raw, mask, faulty=None):
    faulty = torch.zeros_like(replacement_raw) if faulty is None else faulty
    replacement_bev = torch.cat(
        (torch.sigmoid(replacement_raw[:, 0:1]), replacement_raw[:, 1:]),
        dim=1,
    )
    return {
        "replacement_raw": replacement_raw,
        "replacement_bev": replacement_bev,
        "occupancy_logits": replacement_raw[:, 0:1],
        "reconstruction_mask": mask,
        "coarse_lidar_bev": (1.0 - mask) * faulty + mask * replacement_bev,
    }


class CoarseBaselineLossTests(unittest.TestCase):
    def setUp(self):
        self.loss_fn = MaskedBEVReconstructionLoss(CoarseLossConfig())

    def test_perfect_reconstruction_and_metrics(self):
        occupancy = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        density = torch.tensor([[[[0.8, 0.0], [0.3, 0.0]]]])
        height = torch.tensor([[[[0.6, 0.0], [0.2, 0.0]]]])
        clean = torch.cat((occupancy, density, height), dim=1)
        logits = torch.where(occupancy > 0, 20.0, -20.0)
        raw = _raw(logits, density.clone(), height.clone()).requires_grad_()
        mask = torch.ones(1, 1, 2, 2)
        outputs = _model_outputs(raw, mask)

        losses = self.loss_fn(outputs, clean)
        metrics = coarse_reconstruction_metrics(outputs, clean, clean)

        self.assertLess(losses["loss_occupancy_bce"].item(), 1.0e-7)
        self.assertLess(losses["loss_occupancy_dice"].item(), 1.0e-7)
        self.assertEqual(losses["loss_density"].item(), 0.0)
        self.assertEqual(losses["loss_height"].item(), 0.0)
        for name in ("precision", "recall", "f1", "iou"):
            self.assertEqual(metrics[f"coarse_occupancy_{name}"].item(), 1.0)

    def test_completely_wrong_occupancy(self):
        occupancy = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        clean = torch.cat((occupancy, occupancy, occupancy), dim=1)
        logits = torch.where(occupancy > 0, -20.0, 20.0)
        raw = _raw(logits, occupancy, occupancy).requires_grad_()
        mask = torch.ones(1, 1, 2, 2)
        outputs = _model_outputs(raw, mask)

        losses = self.loss_fn(outputs, clean)
        metrics = coarse_reconstruction_metrics(outputs, torch.zeros_like(clean), clean)

        self.assertGreater(losses["loss_occupancy"].item(), 1.0)
        self.assertEqual(metrics["coarse_occupancy_f1"].item(), 0.0)
        self.assertEqual(metrics["coarse_occupancy_iou"].item(), 0.0)

    def test_continuous_errors_on_clean_empty_cells_do_not_contribute(self):
        clean = torch.zeros(1, 3, 2, 2)
        clean[:, 0, 0, 0] = 1.0
        raw = torch.zeros_like(clean)
        raw[:, 1:, 0, 1:] = 100.0
        raw[:, 1:, 1, :] = -100.0
        outputs = _model_outputs(raw.requires_grad_(), torch.ones(1, 1, 2, 2))

        losses = self.loss_fn(outputs, clean)

        self.assertEqual(losses["loss_density"].item(), 0.0)
        self.assertEqual(losses["loss_height"].item(), 0.0)

    def test_errors_outside_reconstruction_mask_do_not_contribute(self):
        clean = torch.zeros(1, 3, 2, 2)
        mask = torch.zeros(1, 1, 2, 2)
        mask[:, :, 0, 0] = 1.0
        clean[:, 0, 0, 0] = 1.0
        logits = torch.full((1, 1, 2, 2), 20.0)
        logits[:, :, 1, :] = -20.0
        density = torch.zeros(1, 1, 2, 2)
        height = torch.zeros_like(density)
        density[:, :, 1, :] = 100.0
        height[:, :, 1, :] = -100.0
        raw = _raw(logits, density, height).requires_grad_()

        losses = self.loss_fn(_model_outputs(raw, mask), clean)

        self.assertLess(losses["loss"].item(), 1.0e-7)

    def test_empty_reconstruction_mask_is_finite_and_differentiable(self):
        raw = torch.randn(2, 3, 3, 3, requires_grad=True)
        clean = torch.rand_like(raw)
        clean[:, 0] = (clean[:, 0] > 0.5).float()
        losses = self.loss_fn(
            _model_outputs(raw, torch.zeros(2, 1, 3, 3)), clean
        )

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(losses["loss"].item(), 0.0)
        losses["loss"].backward()
        self.assertIsNotNone(raw.grad)

    def test_no_clean_occupied_cells_gives_zero_continuous_losses(self):
        raw = torch.randn(1, 3, 3, 3, requires_grad=True)
        clean = torch.zeros_like(raw)
        losses = self.loss_fn(
            _model_outputs(raw, torch.ones(1, 1, 3, 3)), clean
        )

        self.assertEqual(losses["loss_density"].item(), 0.0)
        self.assertEqual(losses["loss_height"].item(), 0.0)

    def test_dice_is_averaged_per_valid_sample_despite_mask_size(self):
        clean = torch.ones(2, 3, 3, 3)
        raw = torch.zeros_like(clean, requires_grad=True)
        masks = torch.zeros(2, 1, 3, 3)
        masks[0, :, 0, 0] = 1.0
        masks[1] = 1.0
        batch_dice = self.loss_fn(_model_outputs(raw, masks), clean)[
            "loss_occupancy_dice"
        ]
        individual = []
        for index in range(2):
            individual.append(
                self.loss_fn(
                    _model_outputs(raw[index : index + 1], masks[index : index + 1]),
                    clean[index : index + 1],
                )["loss_occupancy_dice"]
            )

        self.assertTrue(torch.allclose(batch_dice, torch.stack(individual).mean()))

    def test_total_loss_backpropagates_to_all_three_channels(self):
        raw = torch.zeros(1, 3, 2, 2, requires_grad=True)
        clean = torch.ones_like(raw)
        loss = self.loss_fn(
            _model_outputs(raw, torch.ones(1, 1, 2, 2)), clean
        )["loss"]

        self.assertTrue(loss.requires_grad)
        loss.backward()
        for channel in range(3):
            self.assertGreater(raw.grad[:, channel].abs().sum().item(), 0.0)

    def test_loss_configuration_rejects_invalid_and_obsolete_values(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CoarseLossConfig(lambda_density=-1.0).validate()
        with self.assertRaisesRegex(ValueError, "positive"):
            CoarseLossConfig(epsilon=0.0).validate()
        with self.assertRaisesRegex(ValueError, "obsolete"):
            build_configs(
                {
                    "coarse_reconstruction": {
                        "loss": {"lambda_reconstruction": 1.0}
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
