from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from models.reconstruction_head import (
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    DiffusionProcessConfig,
    FrozenCoarseDiffusionPipeline,
    GaussianNoiseSchedule,
    MaskedEpsilonMSELoss,
    MaskedResidualDiffusion,
    ResidualDiffusionSampler,
    ResidualDiffusionUNetConfig,
    load_frozen_coarse_model,
    occupancy_metrics,
    per_channel_continuous_metrics,
    reconstruction_stage_metrics,
    residual_target,
)


def _diffusion():
    return MaskedResidualDiffusion(
        ResidualDiffusionUNetConfig(
            base_channels=4,
            channel_multipliers=(1, 2),
            residual_blocks_per_level=1,
            time_embedding_dim=16,
        ),
        DiffusionProcessConfig(num_train_timesteps=8, noise_schedule="cosine"),
    )


def _inputs(batch=2, size=16):
    clean = torch.randn(batch, 3, size, size)
    coarse = torch.randn_like(clean)
    mask = torch.zeros(batch, 1, size, size)
    mask[:, :, 4:12, 3:13] = 1
    return clean, coarse, mask


class DummyCoarse(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, faulty, radar, reconstruction_mask, healthy, halo):
        erased = (1 - reconstruction_mask) * faulty
        coarse = (1 - reconstruction_mask) * faulty + reconstruction_mask * self.scale
        return {"erased_lidar_bev": erased, "coarse_lidar_bev": coarse}


class ResidualDiffusionTests(unittest.TestCase):
    def test_masked_target_forward_process_input_and_prediction_shapes(self):
        diffusion = _diffusion()
        clean, coarse, mask = _inputs()
        timestep = torch.tensor([0, 7])
        epsilon = torch.randn_like(clean)
        output = diffusion(clean, coarse, mask, timestep=timestep, epsilon=epsilon)
        self.assertEqual(output["residual_gt"].shape, clean.shape)
        self.assertTrue(torch.all(output["residual_gt"] * (1 - mask) == 0))
        self.assertTrue(torch.all(output["epsilon_masked"] * (1 - mask) == 0))
        self.assertTrue(torch.all(output["residual_t"] * (1 - mask) == 0))
        self.assertEqual(output["diffusion_input"].shape, (2, 7, 16, 16))
        self.assertEqual(output["epsilon_pred"].shape, clean.shape)
        self.assertTrue(torch.isfinite(output["diffusion_loss"]))

    def test_residual_target_uses_clean_minus_coarse_only_inside_mask(self):
        clean, coarse, mask = _inputs(1)
        expected = mask * (clean - coarse)
        self.assertTrue(torch.equal(residual_target(clean, coarse, mask), expected))

    def test_mask_normalized_loss_and_empty_mask_are_differentiable(self):
        loss_fn = MaskedEpsilonMSELoss()
        for width in (1, 5, 0):
            prediction = torch.randn(1, 3, 8, 8, requires_grad=True)
            target = torch.randn_like(prediction)
            mask = torch.zeros(1, 1, 8, 8)
            mask[:, :, :width, :width] = 1
            loss = loss_fn(prediction, target, mask)
            self.assertTrue(torch.isfinite(loss))
            if width == 0:
                self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertIsNotNone(prediction.grad)

    def test_diffusion_gradients_and_no_attention_or_external_conditioning(self):
        diffusion = _diffusion().train()
        clean, coarse, mask = _inputs(1)
        output = diffusion(clean, coarse, mask)
        output["diffusion_loss"].backward()
        self.assertTrue(any(p.grad is not None for p in diffusion.unet.parameters()))
        module_names = {module.__class__.__name__.lower() for module in diffusion.unet.modules()}
        self.assertFalse(any("attention" in name for name in module_names))
        signature_names = diffusion.predict_epsilon.__code__.co_varnames
        for forbidden in ("radar", "halo", "global_context_map", "attention_context"):
            self.assertNotIn(forbidden, signature_names)

    def test_frozen_coarse_pipeline_receives_no_gradients(self):
        coarse = DummyCoarse()
        pipeline = FrozenCoarseDiffusionPipeline(coarse, _diffusion()).train()
        clean, faulty, mask = _inputs(1)
        radar = torch.randn(1, 4, 16, 16)
        healthy = torch.zeros_like(mask)
        halo = torch.zeros_like(mask)
        output = pipeline(clean, faulty, radar, mask, healthy, halo)
        output["diffusion_loss"].backward()
        self.assertTrue(all(p.grad is None for p in pipeline.coarse_model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in pipeline.coarse_model.parameters()))

    def test_ddpm_sampling_is_masked_preserving_and_reproducible(self):
        diffusion = _diffusion().eval()
        sampler = ResidualDiffusionSampler(diffusion)
        _clean, coarse, mask = _inputs(1)
        faulty = coarse.clone()
        first = sampler.sample(
            coarse,
            mask,
            faulty_lidar_bev=faulty,
            generator=torch.Generator().manual_seed(123),
            save_intermediate_steps=True,
            intermediate_stride=1,
        )
        second = sampler.sample(
            coarse,
            mask,
            generator=torch.Generator().manual_seed(123),
        )
        self.assertTrue(torch.equal(first["final_lidar_bev"], second["final_lidar_bev"]))
        self.assertTrue(torch.all(first["residual_pred"] * (1 - mask) == 0))
        self.assertTrue(torch.equal(first["final_lidar_bev"] * (1 - mask), coarse * (1 - mask)))
        for _step, residual in first["intermediate_steps"]:
            self.assertTrue(torch.all(residual * (1 - mask.cpu()) == 0))

    def test_oracle_residual_merge_and_zero_residual(self):
        clean, coarse, mask = _inputs(1)
        oracle = residual_target(clean, coarse, mask)
        final = coarse + mask * oracle
        self.assertTrue(torch.allclose(final * mask, clean * mask, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.equal(final * (1 - mask), coarse * (1 - mask)))
        self.assertTrue(torch.all(residual_target(clean, clean, mask) == 0))

    def test_occupancy_and_continuous_metrics_handle_edge_cases(self):
        mask = torch.ones(1, 1, 2, 2)
        target = torch.zeros(1, 3, 2, 2)
        prediction = torch.zeros_like(target)
        target[:, 2, 0, 0] = 1
        prediction[:, 2, 0, 0] = 1
        prediction[:, 2, 0, 1] = 1
        values = occupancy_metrics(prediction, target, mask)
        self.assertEqual(values["tp"].item(), 1)
        self.assertEqual(values["fp"].item(), 1)
        self.assertAlmostEqual(values["iou"].item(), 0.5)
        self.assertAlmostEqual(values["precision"].item(), 0.5)
        no_positive = occupancy_metrics(torch.zeros_like(target), target, mask)
        self.assertEqual(no_positive["recall"].item(), 0.0)
        empty_target = occupancy_metrics(prediction, torch.zeros_like(target), mask)
        self.assertEqual(empty_target["target_positive_cells"].item(), 0)
        empty_mask = occupancy_metrics(prediction, target, torch.zeros_like(mask))
        self.assertEqual(empty_mask["iou"].item(), 0.0)
        continuous = per_channel_continuous_metrics(prediction, target, mask)
        self.assertEqual(continuous["mae_per_channel"].shape, (3,))
        self.assertAlmostEqual(continuous["mae_per_channel"][2].item(), 0.25)

        stages = reconstruction_stage_metrics(
            prediction,
            prediction,
            prediction,
            torch.zeros_like(target),
            target,
            mask,
        )
        self.assertIn("full_scene", stages)
        self.assertIn("diagnostic_subregions", stages)
        self.assertIn("actual_fault", stages["diagnostic_subregions"])
        self.assertIn("sacrificed_healthy", stages["diagnostic_subregions"])

    def test_mixed_precision_forward_backward(self):
        diffusion = _diffusion().train()
        clean, coarse, mask = _inputs(1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = diffusion(clean, coarse, mask)
        self.assertTrue(torch.isfinite(output["diffusion_loss"]))
        output["diffusion_loss"].backward()
        self.assertIsNotNone(diffusion.unet.output_projection.weight.grad)

    def test_coarse_checkpoint_loader(self):
        config = CoarseReconstructionConfig(
            unet_base_channels=2,
            unet_depth=2,
            global_base_channels=2,
            global_channel_multipliers=(1, 2),
            attention_dim=4,
            num_heads=1,
        )
        model = CoarseReconstructionModel(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coarse.pt"
            torch.save(
                {"model_config": config.to_dict(), "model_state_dict": model.state_dict()},
                path,
            )
            loaded, _checkpoint = load_frozen_coarse_model(path)
        self.assertFalse(loaded.training)
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))


if __name__ == "__main__":
    unittest.main()
