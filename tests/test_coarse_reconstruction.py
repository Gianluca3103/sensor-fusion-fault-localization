import unittest

import torch
import torch.nn.functional as F

from models.reconstruction_head import (
    CoarseLossConfig,
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    MaskedBEVReconstructionLoss,
    coarse_reconstruction_metrics,
)


def _config():
    return CoarseReconstructionConfig(
        unet_base_channels=4,
        unet_depth=3,
        global_base_channels=4,
        global_channel_multipliers=(1, 2, 4),
        attention_dim=16,
        num_heads=4,
    )


def _inputs(batch=2, size=32):
    faulty = torch.randn(batch, 3, size, size)
    radar = torch.randn(batch, 4, size, size)
    clean = torch.randn(batch, 3, size, size)
    reconstruction = torch.zeros(batch, 1, size, size)
    reconstruction[:, :, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
    halo = torch.zeros_like(reconstruction)
    halo[:, :, 2 : size - 2, 2 : size - 2] = 1
    halo *= 1 - reconstruction
    healthy_context = torch.zeros_like(reconstruction)
    healthy_context[:, :, 3 : size - 3 : 2, 3 : size - 3 : 2] = 1
    healthy_context *= halo
    return faulty, radar, reconstruction, healthy_context, halo, clean


class CoarseReconstructionTests(unittest.TestCase):
    def test_local_attention_projection_is_used_only_when_channels_differ(self):
        matching = CoarseReconstructionModel(_config())
        self.assertIsInstance(
            matching.cross_attention.local_projection, torch.nn.Identity
        )

        different = CoarseReconstructionModel(
            CoarseReconstructionConfig(
                unet_base_channels=2,
                unet_depth=5,
                global_base_channels=2,
                global_channel_multipliers=(1, 2, 4, 8, 16),
                attention_dim=8,
                num_heads=2,
            )
        )
        self.assertIsInstance(
            different.cross_attention.local_projection, torch.nn.Conv2d
        )

    def test_direct_bev_erasure_replacement_attention_and_preservation(self):
        model = CoarseReconstructionModel(_config()).eval()
        faulty, radar, reconstruction, healthy, halo, _clean = _inputs()
        with torch.no_grad():
            output = model(
                faulty,
                radar,
                reconstruction,
                healthy,
                halo,
                return_attention_weights=True,
            )

        self.assertEqual(output["local_input"].shape, (2, 9, 32, 32))
        self.assertEqual(output["replacement_raw"].shape, (2, 3, 32, 32))
        self.assertEqual(output["coarse_lidar_bev"].shape, (2, 3, 32, 32))
        self.assertTrue(torch.all(output["erased_lidar_bev"] * reconstruction == 0))
        self.assertTrue(
            torch.equal(
                output["erased_lidar_bev"] * (1 - reconstruction),
                faulty * (1 - reconstruction),
            )
        )
        self.assertTrue(
            torch.equal(
                output["coarse_lidar_bev"] * (1 - reconstruction),
                faulty * (1 - reconstruction),
            )
        )
        # Every selected cell is replaced, regardless of whether it was originally healthy.
        self.assertTrue(
            torch.equal(
                output["coarse_lidar_bev"] * reconstruction,
                output["replacement_raw"] * reconstruction,
            )
        )
        self.assertEqual(output["local_bottleneck"].shape, (2, 16, 8, 8))
        self.assertEqual(output["global_context_map"].shape, (2, 16, 8, 8))
        self.assertEqual(output["query_tokens"].shape, (2, 64, 16))
        self.assertEqual(output["context_tokens"].shape, (2, 64, 16))
        self.assertEqual(output["attention_context"].shape, (2, 16, 8, 8))
        self.assertEqual(output["attention_weights"].shape, (2, 4, 64, 64))
        self.assertEqual(
            output["fused_bottleneck"].shape, output["local_bottleneck"].shape
        )

    def test_default_spatial_progression_reaches_twenty_by_twenty(self):
        config = CoarseReconstructionConfig(
            unet_base_channels=2,
            unet_depth=5,
            global_base_channels=2,
            global_channel_multipliers=(1, 2, 4, 8, 16),
            attention_dim=8,
            num_heads=2,
        )
        model = CoarseReconstructionModel(config).eval()
        faulty, radar, reconstruction, healthy, halo, _clean = _inputs(1, 320)
        with torch.no_grad():
            output = model(faulty, radar, reconstruction, healthy, halo)
        self.assertEqual(output["local_bottleneck"].shape[-2:], (20, 20))
        self.assertEqual(output["global_context_map"].shape[-2:], (20, 20))
        self.assertEqual(output["query_tokens"].shape[1], 400)
        self.assertEqual(output["context_tokens"].shape[1], 400)
        self.assertEqual(output["replacement_raw"].shape, (1, 3, 320, 320))

    def test_global_lidar_encoder_receives_erased_bev(self):
        model = CoarseReconstructionModel(_config()).eval()
        faulty, radar, reconstruction, healthy, halo, _clean = _inputs(1)
        observed = []

        def capture(_module, arguments):
            observed.append(arguments[0].detach().clone())

        handle = model.global_lidar_encoder.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                model(faulty, radar, reconstruction, healthy, halo)
        finally:
            handle.remove()
        self.assertEqual(len(observed), 1)
        self.assertTrue(torch.all(observed[0] * reconstruction == 0))

    def test_loss_covers_complete_reconstruction_region_and_empty_mask(self):
        config = CoarseLossConfig(reconstruction_loss_type="smooth_l1")
        loss_fn = MaskedBEVReconstructionLoss(config)
        replacement = torch.zeros(1, 3, 4, 4, requires_grad=True)
        clean = torch.zeros_like(replacement)
        clean[:, :, 1, 1] = 1.0  # Faulty cell target.
        clean[:, :, 1, 2] = 0.5  # Deliberately sacrificed healthy cell target.
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, 1, 1:3] = 1
        outputs = {"replacement_raw": replacement, "reconstruction_mask": mask}
        value = loss_fn(outputs, clean)["reconstruction_loss"]
        expected = (
            F.smooth_l1_loss(replacement, clean, reduction="none") * mask
        ).sum() / (3 * mask.sum())
        self.assertTrue(torch.equal(value, expected))
        value.backward()
        self.assertNotEqual(replacement.grad[:, :, 1, 2].abs().sum().item(), 0.0)

        empty_replacement = torch.randn(1, 3, 4, 4, requires_grad=True)
        empty_mask = torch.zeros(1, 1, 4, 4)
        empty_loss = loss_fn(
            {
                "replacement_raw": empty_replacement,
                "reconstruction_mask": empty_mask,
            },
            torch.randn_like(empty_replacement),
        )["loss"]
        self.assertTrue(torch.isfinite(empty_loss))
        self.assertEqual(empty_loss.item(), 0.0)
        empty_loss.backward()
        self.assertIsNotNone(empty_replacement.grad)

    def test_metrics_compare_erased_and_reconstructed_bev(self):
        mask = torch.ones(1, 1, 2, 2)
        faulty = torch.full((1, 3, 2, 2), 0.5)
        clean = torch.ones_like(faulty)
        outputs = {
            "reconstruction_mask": mask,
            "erased_lidar_bev": torch.zeros_like(faulty),
            "coarse_lidar_bev": torch.full_like(faulty, 0.75),
        }
        metrics = coarse_reconstruction_metrics(outputs, faulty, clean)
        self.assertAlmostEqual(metrics["erased_masked_mae"].item(), 1.0)
        self.assertAlmostEqual(metrics["coarse_masked_mae"].item(), 0.25)
        self.assertAlmostEqual(metrics["relative_improvement"].item(), 0.75)

    def test_gradients_reach_every_required_component_without_latent_encoder(self):
        model = CoarseReconstructionModel(_config()).train()
        faulty, radar, reconstruction, healthy, halo, clean = _inputs(1)
        outputs = model(faulty, radar, reconstruction, healthy, halo)
        loss = MaskedBEVReconstructionLoss()(outputs, clean)["loss"]
        loss.backward()
        required_modules = (
            model.local_unet_encoder,
            model.global_lidar_encoder,
            model.global_radar_encoder,
            model.global_fusion,
            model.cross_attention,
            model.bottleneck_fusion,
            model.local_unet_decoder,
            model.replacement_head,
        )
        for module in required_modules:
            self.assertTrue(
                any(parameter.grad is not None for parameter in module.parameters()),
                module.__class__.__name__,
            )
        self.assertFalse(hasattr(model, "lidar_representation_encoder"))

    def test_mixed_precision_forward_and_backward(self):
        model = CoarseReconstructionModel(_config()).train()
        faulty, radar, reconstruction, healthy, halo, clean = _inputs(1)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            outputs = model(faulty, radar, reconstruction, healthy, halo)
            loss = MaskedBEVReconstructionLoss()(outputs, clean)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.replacement_head.head.weight.grad)


if __name__ == "__main__":
    unittest.main()
