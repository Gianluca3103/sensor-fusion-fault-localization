"""Frozen-coarse orchestration and masked DDPM sampling."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..coarse_reconstruction.coarse_config import CoarseReconstructionConfig
from ..coarse_reconstruction.coarse_model import CoarseReconstructionModel
from ..pointpillars import BEVGridGeometry
from .local_diffusion import FineDiffusionRefiner


def load_frozen_coarse_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    allow_pointpillars: bool = False,
) -> tuple[CoarseReconstructionModel, dict]:
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
    grid_geometry: BEVGridGeometry | None = None
    if config.pointpillars_enabled:
        if "grid_geometry" not in checkpoint:
            raise KeyError(
                "PointPillars coarse checkpoint requires saved grid_geometry"
            )
        grid_geometry = BEVGridGeometry(**checkpoint["grid_geometry"])
    model = CoarseReconstructionModel(config, grid_geometry=grid_geometry)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


class FrozenCoarseFineDiffusionPipeline(nn.Module):
    """Online frozen coarse reconstruction followed by local fine diffusion."""

    def __init__(
        self,
        coarse_model: CoarseReconstructionModel,
        diffusion: FineDiffusionRefiner,
    ) -> None:
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

