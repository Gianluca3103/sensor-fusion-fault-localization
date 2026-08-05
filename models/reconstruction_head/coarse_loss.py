"""Mask-normalized direct-BEV objectives and metrics for coarse reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CoarseLossConfig:
    reconstruction_loss_type: str = "smooth_l1"
    lambda_reconstruction: float = 1.0
    epsilon: float = 1.0e-8

    def validate(self) -> None:
        if self.reconstruction_loss_type not in {"l1", "smooth_l1"}:
            raise ValueError("reconstruction_loss_type must be 'l1' or 'smooth_l1'")
        if self.lambda_reconstruction < 0:
            raise ValueError("lambda_reconstruction must be non-negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


def masked_bev_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    expected_mask = (prediction.shape[0], 1, *prediction.shape[-2:])
    if tuple(reconstruction_mask.shape) != expected_mask:
        raise ValueError(f"reconstruction_mask must have shape {expected_mask}")
    numerator = (
        reconstruction_mask * (prediction - target).abs()
    ).sum()
    denominator = prediction.shape[1] * reconstruction_mask.sum()
    return numerator / denominator.clamp_min(epsilon)


class MaskedBEVReconstructionLoss(nn.Module):
    """Supervise replacement content over every cell in reconstruction_mask."""

    def __init__(self, config: CoarseLossConfig | None = None):
        super().__init__()
        self.config = config or CoarseLossConfig()
        self.config.validate()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        clean_lidar_bev: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        replacement = outputs["replacement_raw"]
        mask = outputs["reconstruction_mask"]
        if replacement.shape != clean_lidar_bev.shape:
            raise ValueError("replacement_raw and clean_lidar_bev must have identical shapes")
        if self.config.reconstruction_loss_type == "smooth_l1":
            elementwise = F.smooth_l1_loss(
                replacement, clean_lidar_bev, reduction="none"
            )
        else:
            elementwise = (replacement - clean_lidar_bev).abs()
        numerator = (mask * elementwise).sum()
        denominator = replacement.shape[1] * mask.sum()
        reconstruction_loss = numerator / denominator.clamp_min(self.config.epsilon)
        total = self.config.lambda_reconstruction * reconstruction_loss
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite coarse BEV reconstruction loss")
        return {"loss": total, "reconstruction_loss": reconstruction_loss}


def coarse_reconstruction_metrics(
    outputs: dict[str, torch.Tensor],
    faulty_lidar_bev: torch.Tensor,
    clean_lidar_bev: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    mask = outputs["reconstruction_mask"]
    erased_mae = masked_bev_mae(
        outputs["erased_lidar_bev"], clean_lidar_bev, mask, epsilon
    )
    coarse_mae = masked_bev_mae(
        outputs["coarse_lidar_bev"], clean_lidar_bev, mask, epsilon
    )
    improvement = erased_mae - coarse_mae
    relative = improvement / erased_mae.clamp_min(epsilon)
    outside_change = (
        (1.0 - mask) * (outputs["coarse_lidar_bev"] - faulty_lidar_bev)
    ).abs().max()
    return {
        "erased_masked_mae": erased_mae,
        "coarse_masked_mae": coarse_mae,
        "reconstruction_improvement": improvement,
        "relative_improvement": relative,
        "outside_mask_max_change": outside_change,
    }

