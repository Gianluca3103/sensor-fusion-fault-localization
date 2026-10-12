"""Masked denoising without treating unobserved clean space as free."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from models.radar_lidar_stage2_residual.targets import ResidualTargets

from .config import DiffusionConfig


def clean_state(target: ResidualTargets) -> torch.Tensor:
    n = len(target.addition)
    state = target.offsets_normalized.new_zeros((n, 4))
    negative = target.observed | target.known_free
    state[negative, 0] = -1
    state[target.addition, 0] = 1
    state[target.addition, 1:] = target.offsets_normalized[target.addition] * 2
    return state


def training_state(target: ResidualTargets, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Impute unknown sites independently of clean LiDAR before adding noise.

    Zero-filling unknowns would expose the training-only clean visibility mask
    through the noisy input at low timesteps. These sites remain loss-masked.
    """
    state = clean_state(target)
    unknown = ~target.supervised
    # Draw on the fixed candidate shape, avoiding a GPU-to-CPU count read on
    # every batch. Only the unknown positions are copied into the clean state.
    random = torch.rand(state.shape, dtype=state.dtype, device=state.device,
                        generator=generator)
    pseudo_positive = random[:, 0] >= .5
    state[:, 0] = torch.where(unknown, torch.where(pseudo_positive, 1., -1.), state[:, 0])
    offsets = torch.where(pseudo_positive[:, None], random[:, 1:] * 2 - 1, 0.)
    state[:, 1:] = torch.where(unknown[:, None], offsets, state[:, 1:])
    return state


def denoising_loss(predicted: torch.Tensor, clean: torch.Tensor,
                   target: ResidualTargets, config: DiffusionConfig) -> dict[str, torch.Tensor]:
    if predicted.shape != clean.shape or predicted.shape != (len(target.addition), 4):
        raise ValueError("Predicted and target diffusion states differ")
    positive = target.addition
    observed = target.observed
    free = target.known_free
    # Balance categories. Unknown cells have no denoising target and no loss.
    occupancy = predicted.sum() * 0
    classification = predicted.sum() * 0
    active_groups = predicted.new_zeros(())
    occupancy_error = (predicted[:, 0] - clean[:, 0]).square()
    for mask, weight, sign in ((positive, config.residual.positive_weight, 1),
                               (observed, config.residual.observed_weight, -1),
                               (free, 1., -1)):
        count = mask.sum()
        scale = mask.to(predicted.dtype) / count.clamp_min(1)
        active_groups = active_groups + (count > 0).to(predicted.dtype)
        occupancy = occupancy + weight * (occupancy_error * scale).sum()
        classification = classification + weight * (F.softplus(-4 * sign * predicted[:, 0]) * scale).sum()
    occupancy = occupancy / active_groups.clamp_min(1)
    classification = classification / active_groups.clamp_min(1)
    offset_scale = positive.to(predicted.dtype)[:, None] / (positive.sum() * 3).clamp_min(1)
    offset = (((predicted[:, 1:] - clean[:, 1:]).square()) * offset_scale).sum()
    total = (config.occupancy_denoise_weight * occupancy
             + config.offset_denoise_weight * offset
             + config.classification_weight * classification)
    return {"total": total, "occupancy_denoise": occupancy, "offset_denoise": offset,
            "classification": classification, "positive_sites": positive.sum().detach(),
            "observed_sites": observed.sum().detach(), "known_free_sites": free.sum().detach(),
            "unknown_sites": (~target.supervised).sum().detach()}
