"""Noise prediction with observed-surface geometry and abstention supervision."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.radar_lidar_stage2_diffusion.losses import training_state
from models.radar_lidar_stage2_residual.targets import ResidualTargets
from models.radar_lidar_stage2_residual.coverage import FaultyCoverage

from .config import BottleneckDiffusionConfig


def _mean_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[0] != len(mask):
        raise ValueError("Loss mask differs from the candidate support")
    if not bool(mask.any()):
        return values.sum() * 0
    return values[mask].mean()


def reconstruction_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor,
                        occupancy_logits: torch.Tensor, offset_raw: torch.Tensor,
                        target: ResidualTargets, coverage: FaultyCoverage,
                        config: BottleneckDiffusionConfig) -> dict[str, torch.Tensor]:
    n = len(target.addition)
    if (predicted_noise.shape != (n, 4) or target_noise.shape != (n, 4)
            or occupancy_logits.shape != (n,) or offset_raw.shape != (n, 3)
            or coverage.may_add.shape != (n,)):
        raise ValueError("Diffusion predictions and supervision have different candidate shapes")
    positive, free, observed = target.addition, target.known_free, target.observed
    if bool((positive & free).any()) or bool((positive & observed).any()):
        raise ValueError("Positive and negative reconstruction labels overlap")
    noise_error = (predicted_noise.float() - target_noise.float()).square()
    positive_noise = _mean_masked(noise_error, positive)
    free_noise = _mean_masked(noise_error[:, 0], free)
    observed_noise = _mean_masked(noise_error[:, 0], observed)
    active = (int(bool(positive.any())) + int(bool(free.any())) + int(bool(observed.any())))
    diffusion = (config.residual.positive_weight * positive_noise + free_noise
                 + config.residual.observed_weight * observed_noise) / max(active, 1)
    occupancy = (_mean_masked(F.softplus(-occupancy_logits.float()), positive)
                 + _mean_masked(F.softplus(occupancy_logits.float()), free))
    offsets = 0.5 * torch.tanh(offset_raw.float())
    geometry = _mean_masked(F.smooth_l1_loss(
        offsets, target.offsets_normalized.float(), reduction="none"), positive)
    # These candidate cells are already measured or outside the repair mask.
    # Inference also hard-preserves them; the loss discourages a competing add.
    preservation = _mean_masked(F.softplus(occupancy_logits.float()),
                                ~coverage.may_add)
    total = (config.diffusion_weight * diffusion
             + config.occupancy_weight * occupancy
             + config.geometry_weight * geometry
             + config.preservation_weight * preservation)
    return {"total": total, "diffusion": diffusion,
            "noise_positive": positive_noise, "noise_free": free_noise,
            "noise_observed": observed_noise,
            "occupancy": occupancy, "geometry": geometry,
            "preservation": preservation,
            "positive_sites": positive.sum().detach(),
            "free_sites": free.sum().detach(),
            "observed_sites": observed.sum().detach(),
            "unknown_sites": (~target.supervised).sum().detach()}


__all__ = ["training_state", "reconstruction_loss"]
