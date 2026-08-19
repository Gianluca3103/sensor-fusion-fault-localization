"""Frozen-coarse orchestration and masked DDPM sampling."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..coarse_reconstruction.coarse_config import CoarseReconstructionConfig
from ..coarse_reconstruction.coarse_model import CoarseReconstructionModel
from .residual_diffusion import MaskedResidualDiffusion
from .local_diffusion import FineDiffusionRefiner


def load_frozen_coarse_model(
    checkpoint_path, device="cpu", *, allow_pointpillars: bool = False
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise KeyError("Coarse checkpoint requires model_config and model_state_dict")
    config_payload = dict(checkpoint["model_config"])
    if "global_channel_multipliers" in config_payload:
        config_payload["global_channel_multipliers"] = tuple(
            config_payload["global_channel_multipliers"]
        )
    config = CoarseReconstructionConfig.from_dict(config_payload)
    if config.pointpillars_enabled and not allow_pointpillars:
        raise ValueError(
            "PointPillars coarse checkpoints require the local fine-diffusion "
            "pipeline with raw point-cloud inputs"
        )
    model = CoarseReconstructionModel(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


class FrozenCoarseFineDiffusionPipeline(nn.Module):
    """Online frozen coarse reconstruction followed by local fine diffusion."""

    def __init__(self, coarse_model: nn.Module, diffusion: FineDiffusionRefiner):
        super().__init__()
        self.coarse_model = coarse_model.eval().requires_grad_(False)
        self.diffusion = diffusion

    def train(self, mode: bool = True):
        super().train(mode)
        self.coarse_model.eval()
        self.diffusion.train(mode)
        return self

    def coarse_forward(
        self,
        faulty_lidar_bev,
        radar_bev,
        reconstruction_mask,
        healthy_context_mask,
        halo_mask,
        *,
        faulty_lidar_points=None,
        radar_points=None,
    ):
        self.coarse_model.eval()
        with torch.no_grad():
            output = self.coarse_model(
                faulty_lidar_bev,
                radar_bev,
                reconstruction_mask,
                healthy_context_mask,
                halo_mask,
                faulty_lidar_points=faulty_lidar_points,
                radar_points=radar_points,
            )
        return output["coarse_lidar_bev"].detach(), output

    def forward(
        self,
        clean_lidar_bev,
        faulty_lidar_bev,
        radar_bev,
        reconstruction_mask,
        healthy_context_mask,
        halo_mask,
        *,
        coarse_lidar_bev=None,
        faulty_lidar_points=None,
        radar_points=None,
        **diffusion_options,
    ):
        if coarse_lidar_bev is None:
            coarse_lidar_bev, coarse_output = self.coarse_forward(
                faulty_lidar_bev,
                radar_bev,
                reconstruction_mask,
                healthy_context_mask,
                halo_mask,
                faulty_lidar_points=faulty_lidar_points,
                radar_points=radar_points,
            )
        else:
            coarse_lidar_bev = coarse_lidar_bev.detach()
            coarse_output = None
        output = self.diffusion(
            clean_lidar_bev,
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
            **diffusion_options,
        )
        output["coarse_output"] = coarse_output
        return output

    @torch.no_grad()
    def sample(
        self,
        faulty_lidar_bev,
        radar_bev,
        reconstruction_mask,
        healthy_context_mask,
        halo_mask,
        *,
        coarse_lidar_bev=None,
        faulty_lidar_points=None,
        radar_points=None,
        **sampling_options,
    ):
        if coarse_lidar_bev is None:
            coarse_lidar_bev, _coarse_output = self.coarse_forward(
                faulty_lidar_bev,
                radar_bev,
                reconstruction_mask,
                healthy_context_mask,
                halo_mask,
                faulty_lidar_points=faulty_lidar_points,
                radar_points=radar_points,
            )
        return self.diffusion.sample(
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
            **sampling_options,
        )


def validate_diffusion_checkpoint_compatibility(checkpoint, diffusion):
    """Reject legacy 7-channel checkpoints before state loading."""
    state = checkpoint.get("diffusion_state_dict")
    if not isinstance(state, dict):
        raise KeyError("Diffusion checkpoint requires diffusion_state_dict")
    key = "unet.input_projection.weight"
    weight = state.get(key)
    if weight is None or weight.ndim != 4:
        raise KeyError(f"Diffusion checkpoint is missing a valid {key}")
    checkpoint_channels = int(weight.shape[1])
    expected_channels = diffusion.unet.config.input_channels
    if checkpoint_channels != expected_channels:
        legacy = "legacy 7-channel " if checkpoint_channels == 7 else ""
        raise ValueError(
            "Incompatible residual-diffusion checkpoint: "
            f"the {legacy}checkpoint expects {checkpoint_channels} input channels, "
            f"but local radar conditioning requires {expected_channels}. Start a "
            "fresh diffusion run with an 11-channel checkpoint."
        )


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
            radar_bev,
            reconstruction_mask,
            halo_mask,
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
        radar_bev,
        reconstruction_mask,
        halo_mask,
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
            epsilon_pred, _input, local_radar, active_mask = self.diffusion.predict_epsilon(
                residual_t,
                coarse_lidar_bev,
                radar_bev,
                reconstruction_mask,
                halo_mask,
                timestep,
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
            "local_radar": local_radar,
            "active_mask": active_mask,
        }
        if faulty_lidar_bev is not None:
            output["faulty_lidar_bev"] = faulty_lidar_bev
        if save_intermediate_steps:
            output["intermediate_steps"] = intermediates
        return output
