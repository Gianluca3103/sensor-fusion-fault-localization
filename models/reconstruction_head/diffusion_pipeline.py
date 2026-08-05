"""Frozen-coarse orchestration and masked DDPM sampling."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .coarse_model import CoarseReconstructionConfig, CoarseReconstructionModel
from .residual_diffusion import MaskedResidualDiffusion


def load_frozen_coarse_model(checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Coarse checkpoint requires model_config and model_state_dict")
    config_payload = dict(checkpoint["model_config"])
    if "global_channel_multipliers" in config_payload:
        config_payload["global_channel_multipliers"] = tuple(
            config_payload["global_channel_multipliers"]
        )
    model = CoarseReconstructionModel(CoarseReconstructionConfig(**config_payload))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


class FrozenCoarseDiffusionPipeline(nn.Module):
    def __init__(self, coarse_model, diffusion: MaskedResidualDiffusion):
        super().__init__()
        self.coarse_model = coarse_model.eval().requires_grad_(False)
        self.diffusion = diffusion

    def train(self, mode: bool = True):
        super().train(mode)
        self.coarse_model.eval()
        self.diffusion.train(mode)
        return self

    def coarse_forward(
        self, faulty_lidar_bev, radar_bev, reconstruction_mask, healthy_context_mask, halo_mask
    ):
        self.coarse_model.eval()
        with torch.no_grad():
            output = self.coarse_model(
                faulty_lidar_bev,
                radar_bev,
                reconstruction_mask,
                healthy_context_mask,
                halo_mask,
            )
        return output["coarse_lidar_bev"].detach(), output["erased_lidar_bev"].detach()

    def forward(
        self,
        clean_lidar_bev,
        faulty_lidar_bev,
        radar_bev,
        reconstruction_mask,
        healthy_context_mask,
        halo_mask,
        **diffusion_options,
    ):
        coarse, erased = self.coarse_forward(
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            healthy_context_mask,
            halo_mask,
        )
        output = self.diffusion(
            clean_lidar_bev,
            coarse,
            reconstruction_mask,
            **diffusion_options,
        )
        output.update(
            {
                "faulty_lidar_bev": faulty_lidar_bev,
                "erased_lidar_bev": erased,
            }
        )
        return output


class ResidualDiffusionSampler:
    """Correct masked ancestral DDPM sampler; DDIM can share this interface later."""

    def __init__(self, diffusion: MaskedResidualDiffusion):
        self.diffusion = diffusion

    @torch.no_grad()
    def sample(
        self,
        coarse_lidar_bev,
        reconstruction_mask,
        *,
        faulty_lidar_bev=None,
        generator=None,
        save_intermediate_steps=False,
        intermediate_stride=100,
    ):
        self.diffusion.eval()
        residual_t = reconstruction_mask * torch.randn(
            coarse_lidar_bev.shape,
            device=coarse_lidar_bev.device,
            dtype=coarse_lidar_bev.dtype,
            generator=generator,
        )
        intermediates = []
        total = self.diffusion.schedule.config.num_train_timesteps
        for index in reversed(range(total)):
            timestep = torch.full(
                (coarse_lidar_bev.shape[0],),
                index,
                device=coarse_lidar_bev.device,
                dtype=torch.long,
            )
            epsilon_pred, _input = self.diffusion.predict_epsilon(
                residual_t, coarse_lidar_bev, reconstruction_mask, timestep
            )
            residual_t = self.diffusion.schedule.ddpm_step(
                residual_t,
                epsilon_pred,
                timestep,
                reconstruction_mask,
                generator=generator,
            )
            if save_intermediate_steps and (
                index == 0 or index % max(intermediate_stride, 1) == 0
            ):
                intermediates.append((index, residual_t.detach().cpu()))
        residual_pred = reconstruction_mask * self.diffusion.normalization.denormalize_residual(
            residual_t
        )
        final_lidar_bev = coarse_lidar_bev + reconstruction_mask * residual_pred
        output = {
            "coarse_lidar_bev": coarse_lidar_bev,
            "residual_pred": residual_pred,
            "final_lidar_bev": final_lidar_bev,
            "reconstruction_mask": reconstruction_mask,
        }
        if faulty_lidar_bev is not None:
            output["faulty_lidar_bev"] = faulty_lidar_bev
        if save_intermediate_steps:
            output["intermediate_steps"] = intermediates
        return output

