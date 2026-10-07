"""Reward justified additions; penalize duplicate and clean free-space returns."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import ResidualStage2Config
from .model import ResidualOutput
from .targets import ResidualTargets


def residual_loss(output: ResidualOutput, target: ResidualTargets,
                  config: ResidualStage2Config) -> dict[str, torch.Tensor]:
    logits = output.addition_logits
    if len(logits) != len(target.addition):
        raise ValueError("Residual output and target domains differ")
    zero = logits.sum() * 0
    # Normalize by category. Otherwise the many observed/empty voxels drown
    # out the comparatively rare useful additions.
    terms = []
    if bool(target.addition.any()):
        terms.append(config.positive_weight * F.softplus(-logits[target.addition]).mean())
    if bool(target.observed.any()):
        terms.append(config.observed_weight * F.softplus(logits[target.observed]).mean())
    if bool(target.known_free.any()):
        terms.append(F.softplus(logits[target.known_free]).mean())
    occupancy = sum(terms) / len(terms) if terms else zero
    if bool(target.addition.any()):
        offset = F.smooth_l1_loss(output.predicted_offsets[target.addition],
                                  target.offsets_normalized[target.addition], beta=.1)
    else:
        offset = zero
    return {"total": config.occupancy_weight * occupancy + config.offset_weight * offset,
            "occupancy": occupancy, "offset": offset,
            "addition_sites": target.addition.sum().detach(),
            "observed_sites": target.observed.sum().detach(),
            "known_free_sites": target.known_free.sum().detach(),
            "unknown_sites": (~target.supervised).sum().detach()}
