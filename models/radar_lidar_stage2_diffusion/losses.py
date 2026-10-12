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
    count = int(unknown.sum())
    if count:
        random = torch.rand((count, 4), dtype=state.dtype, device=state.device,
                            generator=generator)
        pseudo_positive = random[:, 0] >= .5
        state[unknown, 0] = torch.where(pseudo_positive, 1., -1.)
        offsets = random[:, 1:] * 2 - 1
        state[unknown, 1:] = torch.where(pseudo_positive[:, None], offsets,
                                        torch.zeros_like(offsets))
    return state


def denoising_loss(predicted: torch.Tensor, clean: torch.Tensor,
                   target: ResidualTargets, config: DiffusionConfig) -> dict[str, torch.Tensor]:
    if predicted.shape != clean.shape or predicted.shape != (len(target.addition), 4):
        raise ValueError("Predicted and target diffusion states differ")
    zero = predicted.sum() * 0
    positive = target.addition
    observed = target.observed
    free = target.known_free
    # Balance categories. Unknown cells have no denoising target and no loss.
    terms = []
    cls_terms = []
    for mask, weight, sign in ((positive, config.residual.positive_weight, 1),
                               (observed, config.residual.observed_weight, -1),
                               (free, 1., -1)):
        if bool(mask.any()):
            terms.append(weight * F.mse_loss(predicted[mask, 0], clean[mask, 0]))
            cls_terms.append(weight * F.softplus(-4 * sign * predicted[mask, 0]).mean())
    occupancy = sum(terms) / len(terms) if terms else zero
    classification = sum(cls_terms) / len(cls_terms) if cls_terms else zero
    offset = (F.mse_loss(predicted[positive, 1:], clean[positive, 1:])
              if bool(positive.any()) else zero)
    total = (config.occupancy_denoise_weight * occupancy
             + config.offset_denoise_weight * offset
             + config.classification_weight * classification)
    return {"total": total, "occupancy_denoise": occupancy, "offset_denoise": offset,
            "classification": classification, "positive_sites": positive.sum().detach(),
            "observed_sites": observed.sum().detach(), "known_free_sites": free.sum().detach(),
            "unknown_sites": (~target.supervised).sum().detach()}
