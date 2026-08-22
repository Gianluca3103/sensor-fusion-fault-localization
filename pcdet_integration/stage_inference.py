"""Load frozen thesis reconstruction stages for OpenPCDet condition export."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch

from models.two_stage_reconstruction_head import (
    BEVChannelNormalization,
    FineDiffusionConfig,
    FineDiffusionRefiner,
    FrozenCoarseFineDiffusionPipeline,
    ResidualChannelNormalization,
    load_frozen_coarse_model,
    validate_fine_diffusion_checkpoint_compatibility,
)


def load_frozen_reconstruction_pipeline(
    coarse_checkpoint_path: str | Path,
    fine_checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[FrozenCoarseFineDiffusionPipeline, dict, dict]:
    """Load the exact frozen coarse and fine models used for evaluation."""

    coarse, coarse_checkpoint = load_frozen_coarse_model(
        coarse_checkpoint_path,
        device,
        allow_pointpillars=True,
    )
    fine_checkpoint = torch.load(
        fine_checkpoint_path, map_location="cpu", weights_only=False
    )
    required = {"diffusion_state_dict", "diffusion_config"}
    missing = sorted(required - set(fine_checkpoint))
    if missing:
        raise KeyError(f"Fine checkpoint is missing: {', '.join(missing)}")
    valid = {field.name for field in fields(FineDiffusionConfig)}
    raw_config = dict(fine_checkpoint["diffusion_config"])
    unknown = sorted(set(raw_config) - valid)
    if unknown:
        raise ValueError("Unknown fine checkpoint settings: " + ", ".join(unknown))
    diffusion_config = FineDiffusionConfig(**raw_config)
    diffusion_config.validate()
    validate_fine_diffusion_checkpoint_compatibility(
        fine_checkpoint, diffusion_config
    )

    bev_meta = fine_checkpoint.get("bev_normalization") or {}
    bev_normalizer = BEVChannelNormalization(
        means=bev_meta.get("means", (0.0,) * diffusion_config.lidar_channels),
        stds=bev_meta.get("stds", (1.0,) * diffusion_config.lidar_channels),
        epsilon=float(bev_meta.get("epsilon", 1.0e-6)),
        source=bev_meta.get("source", "fine_checkpoint"),
    )
    residual_meta = fine_checkpoint.get("residual_normalization")
    if residual_meta is None:
        raise KeyError(
            "Fine checkpoint lacks train-only residual_normalization metadata"
        )
    residual_normalizer = ResidualChannelNormalization(
        means=residual_meta["means"],
        stds=residual_meta["stds"],
        epsilon=float(residual_meta.get("epsilon", 1.0e-6)),
        source=residual_meta.get("source", "fine_checkpoint"),
    )
    diffusion = FineDiffusionRefiner(
        diffusion_config, bev_normalizer, residual_normalizer
    ).to(device)
    diffusion.load_state_dict(fine_checkpoint["diffusion_state_dict"], strict=True)
    pipeline = FrozenCoarseFineDiffusionPipeline(coarse, diffusion).to(device)
    pipeline.eval().requires_grad_(False)
    return pipeline, coarse_checkpoint, fine_checkpoint
