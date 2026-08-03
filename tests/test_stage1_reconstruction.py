import unittest

import torch

from models.reconstruction_head.coarse_reconstructor import CoarseLiDARRadarReconstructor
from models.reconstruction_head.diffusion_scheduler import DiffusionSchedule
from models.reconstruction_head.losses import diffusion_noise_loss
from models.reconstruction_head.residual_diffusion_unet import ResidualDiffusionUNet
from models.reconstruction_head.stage1_pipeline import Stage1ReconstructionPipeline


class Stage1ReconstructionTests(unittest.TestCase):
    def _tensors(self):
        torch.manual_seed(7)
        lidar = torch.randn(2, 3, 64, 64)
        clean = lidar.clone()
        clean[:, :, 20:36, 24:40] += 0.5
        radar = torch.rand(2, 4, 64, 64)
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 20:36, 24:40] = 1.0
        occ = (lidar.abs().sum(dim=1, keepdim=True) > 0).float()
        return lidar, clean, radar, mask, occ

    def test_stage1_shapes(self):
        lidar, _, radar, mask, occ = self._tensors()
        coarse = CoarseLiDARRadarReconstructor(
            lidar_channels=3,
            radar_channels=4,
            base_channels=4,
            levels=3,
            conditioning_channels=8,
        )
        out = coarse(lidar, radar, mask, occ)
        self.assertEqual(out["delta_coarse"].shape, lidar.shape)
        self.assertEqual(out["coarse_features"].shape, lidar.shape)
        diffusion = ResidualDiffusionUNet(
            residual_channels=3,
            coarse_channels=3,
            conditioning_channels=out["conditioning_features"].shape[1],
            base_channels=4,
            levels=3,
        )
        timesteps = torch.tensor([1, 2])
        pred = diffusion(
            torch.randn_like(lidar) * mask,
            out["coarse_features"],
            out["conditioning_features"],
            mask,
            timesteps,
        )
        self.assertEqual(pred.shape, lidar.shape)

    def test_healthy_cells_are_preserved_by_construction(self):
        lidar, _, radar, mask, occ = self._tensors()
        coarse = CoarseLiDARRadarReconstructor(3, 4, base_channels=4, levels=3)
        diffusion = ResidualDiffusionUNet(3, 3, 4, base_channels=4, levels=3)
        pipeline = Stage1ReconstructionPipeline(coarse, diffusion, DiffusionSchedule(num_train_timesteps=10))
        coarse_out = coarse(lidar, radar, mask, occ)
        self.assertLess(float(((1.0 - mask) * (coarse_out["coarse_features"] - lidar)).abs().max().detach()), 1e-6)
        final = lidar * (1.0 - mask) + (coarse_out["coarse_features"] + mask * torch.randn_like(lidar)) * mask
        self.assertLess(float(((1.0 - mask) * (final - lidar)).abs().max().detach()), 1e-6)

    def test_residual_construction_zero_when_coarse_matches_clean(self):
        lidar, clean, _, mask, _ = self._tensors()
        coarse_features = lidar.clone()
        coarse_features = coarse_features * (1.0 - mask) + clean * mask
        residual = Stage1ReconstructionPipeline.residual_target(clean, coarse_features, mask)
        self.assertLess(float((mask * residual).abs().max()), 1e-6)

    def test_forward_diffusion_masks_noise(self):
        lidar, clean, _, mask, _ = self._tensors()
        residual = mask * (clean - lidar)
        noise = torch.randn_like(residual)
        schedule = DiffusionSchedule(num_train_timesteps=10)
        noisy = schedule.add_noise(residual, torch.tensor([3, 4]), noise, mask)
        outside = (1.0 - mask) * noisy
        self.assertLess(float(outside.abs().max()), 1e-6)

    def test_frozen_coarse_model_keeps_no_gradients(self):
        lidar, clean, radar, mask, occ = self._tensors()
        coarse = CoarseLiDARRadarReconstructor(3, 4, base_channels=4, levels=3)
        for parameter in coarse.parameters():
            parameter.requires_grad_(False)
        diffusion = ResidualDiffusionUNet(3, 3, 4, base_channels=4, levels=3)
        with torch.no_grad():
            coarse_out = coarse(lidar, radar, mask, occ)
            residual = mask * (clean - coarse_out["coarse_features"])
        prediction = diffusion(
            residual,
            coarse_out["coarse_features"].detach(),
            coarse_out["conditioning_features"].detach(),
            mask,
            torch.tensor([1, 2]),
        )
        loss = diffusion_noise_loss(torch.zeros_like(prediction), prediction, mask)
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in coarse.parameters()))


if __name__ == "__main__":
    unittest.main()
