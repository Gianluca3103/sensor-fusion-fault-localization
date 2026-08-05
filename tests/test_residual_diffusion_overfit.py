"""Small fixed-noise overfit proofs for masked residual diffusion."""

import unittest

import torch

from models.reconstruction_head import (
    DiffusionProcessConfig,
    MaskedResidualDiffusion,
    ResidualDiffusionUNetConfig,
)


class ResidualDiffusionOverfitTests(unittest.TestCase):
    def _run_case(self, masks):
        torch.manual_seed(91)
        batch = masks.shape[0]
        diffusion = MaskedResidualDiffusion(
            ResidualDiffusionUNetConfig(
                base_channels=4,
                channel_multipliers=(1,),
                residual_blocks_per_level=1,
                time_embedding_dim=8,
            ),
            DiffusionProcessConfig(num_train_timesteps=4, noise_schedule="linear"),
        )
        coarse = torch.zeros(batch, 3, 8, 8)
        clean = masks * (0.2 + 0.6 * torch.rand_like(coarse))
        timestep = torch.full((batch,), 2, dtype=torch.long)
        epsilon = torch.randn_like(clean)
        optimizer = torch.optim.Adam(diffusion.unet.parameters(), lr=2e-2)

        with torch.no_grad():
            initial = diffusion(
                clean, coarse, masks, timestep=timestep, epsilon=epsilon
            )
            initial_loss = float(initial["diffusion_loss"])
        for _ in range(180):
            optimizer.zero_grad(set_to_none=True)
            output = diffusion(
                clean, coarse, masks, timestep=timestep, epsilon=epsilon
            )
            output["diffusion_loss"].backward()
            optimizer.step()

        with torch.no_grad():
            fitted = diffusion(
                clean, coarse, masks, timestep=timestep, epsilon=epsilon
            )
            schedule = diffusion.schedule
            residual_prediction = masks * (
                schedule.extract(schedule.sqrt_recip_alpha_bars, timestep, fitted["residual_t"])
                * fitted["residual_t"]
                - schedule.extract(schedule.sqrt_recipm1_alpha_bars, timestep, fitted["residual_t"])
                * fitted["epsilon_pred"]
            )
            final = coarse + residual_prediction
            denominator = (3 * masks.sum()).clamp_min(1e-8)
            coarse_mae = float((masks * (coarse - clean)).abs().sum() / denominator)
            final_mae = float((masks * (final - clean)).abs().sum() / denominator)
            outside_change = float(((1 - masks) * (final - coarse)).abs().max())

        self.assertLess(float(fitted["diffusion_loss"]), initial_loss * 0.05)
        self.assertLess(final_mae, coarse_mae * 0.25)
        self.assertEqual(outside_change, 0.0)

    def test_one_sample_overfits_and_improves_final_masked_mae(self):
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        self._run_case(mask)

    def test_two_different_mask_sizes_overfit_stably(self):
        masks = torch.zeros(2, 1, 8, 8)
        masks[0, :, 1:4, 1:4] = 1
        masks[1, :, 1:7, 2:6] = 1
        self._run_case(masks)


if __name__ == "__main__":
    unittest.main()
