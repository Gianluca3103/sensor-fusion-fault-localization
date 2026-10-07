"""Supervise measured surfaces and conservatively visible free cells only."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import Stage2Config
from .reconstruction_model import Stage2Output
from .voxel_target import VoxelTargets


def reconstruction_loss(output: Stage2Output, target: VoxelTargets,
                        config: Stage2Config) -> dict[str, torch.Tensor]:
    if len(output.occupancy_logits) != len(target.occupied):
        raise ValueError("Output and target candidate domains differ")
    supervised = target.occupied | target.known_free
    zero = output.occupancy_logits.sum() * 0
    if bool(supervised.any()):
        occupancy = F.binary_cross_entropy_with_logits(
            output.occupancy_logits[supervised], target.occupied[supervised].float(),
            pos_weight=output.occupancy_logits.new_tensor(config.positive_weight))
    else:
        occupancy = zero
    if bool(target.occupied.any()):
        offset = F.smooth_l1_loss(output.predicted_offsets[target.occupied],
                                  target.offsets_normalized[target.occupied], beta=0.1)
    else:
        offset = zero
    total = config.occupancy_weight * occupancy + config.offset_weight * offset
    return {"total": total, "occupancy": occupancy, "offset": offset,
            "weighted_occupancy": config.occupancy_weight * occupancy,
            "weighted_offset": config.offset_weight * offset,
            "positive_sites": target.occupied.sum().detach(),
            "known_free_sites": target.known_free.sum().detach(),
            "unknown_sites": (~supervised).sum().detach()}
