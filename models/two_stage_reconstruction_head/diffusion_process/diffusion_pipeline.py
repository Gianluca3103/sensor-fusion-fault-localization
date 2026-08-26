"""Frozen-coarse orchestration and masked DDPM sampling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from ..coarse_reconstruction.coarse_config import CoarseReconstructionConfig
from ..coarse_reconstruction.coarse_model import CoarseReconstructionModel
from ..pointpillars import BEVGridGeometry
from ..reconstruction_inputs import ReconstructionInputs
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
        coarse_model: CoarseReconstructionModel | None,
        diffusion: FineDiffusionRefiner,
    ) -> None:
        super().__init__()
        if coarse_model is None and not diffusion.config.bypass_coarse_reconstruction:
            raise ValueError(
                "A coarse model is required unless coarse reconstruction is bypassed"
            )
        if diffusion.config.use_pointpillars_conditioning:
            if coarse_model is None:
                raise ValueError(
                    "Fine PointPillars conditioning requires a frozen coarse model"
                )
            if not coarse_model.config.pointpillars_enabled:
                raise ValueError(
                    "Fine PointPillars conditioning requires a PointPillars "
                    "coarse checkpoint"
                )
        self.coarse_model = (
            coarse_model.eval().requires_grad_(False)
            if coarse_model is not None
            else None
        )
        self.diffusion = diffusion

    def train(self, mode: bool = True):
        super().train(mode)
        if self.coarse_model is not None:
            self.coarse_model.eval()
        self.diffusion.train(mode)
        return self

    @property
    def bypasses_coarse_reconstruction(self) -> bool:
        return self.diffusion.config.bypass_coarse_reconstruction

    @staticmethod
    def _erased_faulty_base(faulty_lidar_bev, reconstruction_mask):
        return faulty_lidar_bev * (1.0 - reconstruction_mask)

    @staticmethod
    def _shared_inputs(
        faulty_lidar_bev,
        radar_bev,
        reconstruction_mask,
        healthy_context_mask,
        halo_mask,
        *,
        faulty_lidar_points=None,
        radar_points=None,
    ) -> ReconstructionInputs:
        return ReconstructionInputs(
            faulty_lidar_bev=faulty_lidar_bev,
            radar_bev=radar_bev,
            reconstruction_mask=reconstruction_mask,
            healthy_context_mask=healthy_context_mask,
            halo_mask=halo_mask,
            faulty_lidar_points=faulty_lidar_points,
            radar_points=radar_points,
        )

    def coarse_forward_inputs(self, inputs: ReconstructionInputs):
        if self.bypasses_coarse_reconstruction:
            base = self._erased_faulty_base(
                inputs.faulty_lidar_bev, inputs.reconstruction_mask
            )
            occupancy = base[:, 0:1].clamp(1.0e-6, 1.0 - 1.0e-6)
            return base, {
                "coarse_lidar_bev": base,
                "occupancy_logits": torch.logit(occupancy),
                "bypassed_coarse_reconstruction": True,
            }
        if self.coarse_model is None:
            raise RuntimeError("Coarse reconstruction model is unavailable")
        self.coarse_model.eval()
        with torch.no_grad():
            output = self.coarse_model(
                inputs.faulty_lidar_bev,
                inputs.radar_bev,
                inputs.reconstruction_mask,
                inputs.healthy_context_mask,
                inputs.halo_mask,
                faulty_lidar_points=inputs.faulty_lidar_points,
                radar_points=inputs.radar_points,
                shared_inputs=inputs,
            )
        return output["coarse_lidar_bev"].detach(), output

    def _with_pointpillar_conditioning(
        self,
        inputs: ReconstructionInputs,
        coarse_output: dict | None,
    ) -> ReconstructionInputs:
        """Attach frozen LiDAR/radar post-scatter tensors for fine conditioning."""

        if not self.diffusion.config.use_pointpillars_conditioning:
            return inputs
        if self.coarse_model is None:
            raise RuntimeError("Frozen coarse PointPillars model is unavailable")
        if coarse_output is not None:
            lidar_pillars = coarse_output.get("lidar_pillar_bev")
            radar_pillars = coarse_output.get("radar_pillar_bev")
        else:
            lidar_pillars = radar_pillars = None
        if lidar_pillars is None or radar_pillars is None:
            with torch.no_grad():
                (
                    lidar_pillars,
                    radar_pillars,
                    _lidar_statistics,
                    _radar_statistics,
                ) = self.coarse_model._sensor_features(
                    inputs.faulty_lidar_bev,
                    inputs.radar_bev,
                    inputs.faulty_lidar_points,
                    inputs.radar_points,
                    radar_enabled=True,
                )
        expected_lidar = self.diffusion.config.lidar_pillar_channels
        expected_radar = self.diffusion.config.radar_pillar_channels
        if lidar_pillars.shape[1] != expected_lidar:
            raise ValueError(
                "LiDAR PointPillars channel mismatch: fine diffusion expects "
                f"{expected_lidar}, frozen encoder produced {lidar_pillars.shape[1]}"
            )
        if radar_pillars.shape[1] != expected_radar:
            raise ValueError(
                "Radar PointPillars channel mismatch: fine diffusion expects "
                f"{expected_radar}, frozen encoder produced {radar_pillars.shape[1]}"
            )
        return replace(
            inputs,
            lidar_pillar_bev=lidar_pillars.detach(),
            radar_pillar_bev=radar_pillars.detach(),
        )

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
        inputs = self._shared_inputs(
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            healthy_context_mask,
            halo_mask,
            faulty_lidar_points=faulty_lidar_points,
            radar_points=radar_points,
        )
        return self.coarse_forward_inputs(inputs)

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
        coarse_output=None,
        faulty_lidar_points=None,
        radar_points=None,
        **diffusion_options,
    ):
        shared_inputs = self._shared_inputs(
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            healthy_context_mask,
            halo_mask,
            faulty_lidar_points=faulty_lidar_points,
            radar_points=radar_points,
        )
        if coarse_lidar_bev is None:
            if coarse_output is not None:
                raise ValueError(
                    "coarse_output cannot be supplied without coarse_lidar_bev"
                )
            coarse_lidar_bev, coarse_output = self.coarse_forward_inputs(
                shared_inputs
            )
        else:
            coarse_lidar_bev = coarse_lidar_bev.detach()
        shared_inputs = self._with_pointpillar_conditioning(
            shared_inputs, coarse_output
        )
        output = self.diffusion(
            clean_lidar_bev,
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
            shared_inputs=shared_inputs,
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
        coarse_output=None,
        faulty_lidar_points=None,
        radar_points=None,
        **sampling_options,
    ):
        shared_inputs = self._shared_inputs(
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            healthy_context_mask,
            halo_mask,
            faulty_lidar_points=faulty_lidar_points,
            radar_points=radar_points,
        )
        if coarse_lidar_bev is None:
            if coarse_output is not None:
                raise ValueError(
                    "coarse_output cannot be supplied without coarse_lidar_bev"
                )
            coarse_lidar_bev, coarse_output = self.coarse_forward_inputs(
                shared_inputs
            )
        shared_inputs = self._with_pointpillar_conditioning(
            shared_inputs, coarse_output
        )
        return self.diffusion.sample(
            coarse_lidar_bev,
            faulty_lidar_bev,
            radar_bev,
            reconstruction_mask,
            halo_mask,
            shared_inputs=shared_inputs,
            **sampling_options,
        )
