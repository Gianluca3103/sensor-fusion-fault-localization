from __future__ import annotations

import torch
from torch import nn

from .diffusion_scheduler import DiffusionSchedule
from .losses import diffusion_noise_loss, residual_l1_loss


class Stage1ReconstructionPipeline(nn.Module):
    """Wrapper coordinating coarse reconstruction and residual diffusion."""

    def __init__(self, coarse_model: nn.Module, diffusion_model: nn.Module | None = None, schedule: DiffusionSchedule | None = None):
        super().__init__()
        self.coarse_model = coarse_model
        self.diffusion_model = diffusion_model
        self.schedule = schedule or DiffusionSchedule()

    def forward_coarse(self, lidar_corrupt, radar, mask_gt, occupancy=None):
        return self.coarse_model(lidar_corrupt, radar, mask_gt, occupancy)

    @staticmethod
    def residual_target(lidar_clean: torch.Tensor, coarse_features: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        return mask_gt * (lidar_clean - coarse_features)

    def forward_diffusion_train(
        self,
        noisy_residual: torch.Tensor,
        coarse_features: torch.Tensor,
        conditioning_features: torch.Tensor,
        mask_gt: torch.Tensor,
        timestep: torch.Tensor,
    ):
        if self.diffusion_model is None:
            raise RuntimeError("diffusion_model is required for diffusion training")
        return self.diffusion_model(noisy_residual, coarse_features, conditioning_features, mask_gt, timestep)

    @torch.no_grad()
    def sample_residual(
        self,
        coarse_features: torch.Tensor,
        conditioning_features: torch.Tensor,
        mask_gt: torch.Tensor,
        *,
        num_inference_steps: int = 50,
        seed: int | None = None,
    ) -> torch.Tensor:
        if self.diffusion_model is None:
            raise RuntimeError("diffusion_model is required for sampling")
        device = coarse_features.device
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
            residual = torch.randn(coarse_features.shape, device=device, generator=generator) * mask_gt
        else:
            residual = torch.randn_like(coarse_features) * mask_gt
        timesteps = self.schedule.to(device).inference_timesteps(num_inference_steps, device)
        for index, timestep_value in enumerate(timesteps):
            timestep = torch.full((coarse_features.shape[0],), int(timestep_value.item()), device=device, dtype=torch.long)
            noise_prediction = self.diffusion_model(residual, coarse_features, conditioning_features, mask_gt, timestep)
            if index == len(timesteps) - 1:
                residual = self.schedule.reconstruct_x0(residual, timestep, noise_prediction)
            else:
                prev = torch.full_like(timestep, int(timesteps[index + 1].item()))
                residual = self.schedule.ddim_step(residual, timestep, prev, noise_prediction)
            residual = residual * mask_gt
        return residual

    @torch.no_grad()
    def reconstruct(self, lidar_corrupt, radar, mask_gt, occupancy=None, *, num_inference_steps: int = 50, seed: int | None = None):
        coarse_output = self.forward_coarse(lidar_corrupt, radar, mask_gt, occupancy)
        if self.diffusion_model is None:
            final = coarse_output["coarse_features"]
            residual = torch.zeros_like(final)
        else:
            residual = self.sample_residual(
                coarse_output["coarse_features"],
                coarse_output["conditioning_features"],
                mask_gt,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            final = coarse_output["coarse_features"] + mask_gt * residual
        final = lidar_corrupt * (1.0 - mask_gt) + final * mask_gt
        return {"coarse": coarse_output, "predicted_residual": residual, "final_features": final}

