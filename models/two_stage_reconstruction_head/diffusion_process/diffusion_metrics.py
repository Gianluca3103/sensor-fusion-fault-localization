"""Repair-region occupancy and continuous metrics for BEV reconstruction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def metric_disk_kernel(
    tolerance_m: float,
    meters_per_cell_x: float,
    meters_per_cell_y: float,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Build an anisotropic-grid disk using physical cell-center distances."""

    if tolerance_m < 0.0:
        raise ValueError("tolerance_m must be non-negative")
    if meters_per_cell_x <= 0.0 or meters_per_cell_y <= 0.0:
        raise ValueError("BEV metres-per-cell values must be positive")
    # Ceil establishes a numerically safe bounding box. The physical-distance
    # comparison below removes offsets outside the requested metric disk.
    row_radius = int(math.ceil(tolerance_m / meters_per_cell_x))
    column_radius = int(math.ceil(tolerance_m / meters_per_cell_y))
    row_offsets = torch.arange(
        -row_radius, row_radius + 1, device=device, dtype=torch.float32
    )
    column_offsets = torch.arange(
        -column_radius, column_radius + 1, device=device, dtype=torch.float32
    )
    rows, columns = torch.meshgrid(
        row_offsets, column_offsets, indexing="ij"
    )
    kernel = (
        torch.sqrt(
            (rows * meters_per_cell_x).square()
            + (columns * meters_per_cell_y).square()
        )
        <= tolerance_m + 1.0e-6
    ).to(dtype=torch.float32)[None, None]
    return kernel, (row_radius, column_radius)


def tolerant_occupancy_counts(
    predicted: torch.Tensor,
    occupied: torch.Tensor,
    valid: torch.Tensor,
    *,
    tolerance_m: float,
    meters_per_cell_x: float,
    meters_per_cell_y: float,
) -> dict[str, float]:
    """Return bidirectional tolerant-matching sufficient statistics."""

    predicted = predicted.bool() & valid.bool()
    occupied = occupied.bool() & valid.bool()
    kernel, padding = metric_disk_kernel(
        tolerance_m,
        meters_per_cell_x,
        meters_per_cell_y,
        device=predicted.device,
    )

    def dilate(values: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            values.float(), kernel, padding=padding
        ) > 0

    target_neighborhood = dilate(occupied)
    prediction_neighborhood = dilate(predicted)
    return {
        "matched_predictions": float((predicted & target_neighborhood).sum()),
        "matched_targets": float((occupied & prediction_neighborhood).sum()),
        "prediction_count": float(predicted.sum()),
        "target_count": float(occupied.sum()),
    }


def tolerant_metrics_from_counts(
    counts: dict[str, float], epsilon: float = 1.0e-8
) -> dict[str, float]:
    precision = counts["matched_predictions"] / (
        counts["prediction_count"] + epsilon
    )
    recall = counts["matched_targets"] / (counts["target_count"] + epsilon)
    f1 = 2.0 * precision * recall / (precision + recall + epsilon)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": f1 / (2.0 - f1 + epsilon),
    }


def bev_occupancy(bev: torch.Tensor) -> torch.Tensor:
    """Return occupancy from the canonical first LiDAR BEV channel."""
    if bev.ndim != 4 or bev.shape[1] < 1:
        raise ValueError("LiDAR BEV must have shape [B,C>=1,H,W]")
    return bev[:, 0:1] >= 0.5


def occupancy_metrics(prediction, target, reconstruction_mask, epsilon=1e-8):
    predicted = bev_occupancy(prediction)
    expected = bev_occupancy(target)
    selected = reconstruction_mask > 0.5
    tp = (predicted & expected & selected).sum().float()
    fp = (predicted & ~expected & selected).sum().float()
    fn = (~predicted & expected & selected).sum().float()
    tn = (~predicted & ~expected & selected).sum().float()
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "target_positive_cells": (expected & selected).sum().float(),
        "selected_cells": selected.sum().float(),
        "iou": tp / (tp + fp + fn + epsilon),
        "f1": 2 * tp / (2 * tp + fp + fn + epsilon),
        "precision": tp / (tp + fp + epsilon),
        "recall": tp / (tp + fn + epsilon),
    }


def per_channel_continuous_metrics(prediction, target, reconstruction_mask, epsilon=1e-8):
    error = reconstruction_mask * (prediction - target)
    denominator = reconstruction_mask.sum() + epsilon
    dimensions = (0, 2, 3)
    mae = error.abs().sum(dim=dimensions) / denominator
    rmse = (error.square().sum(dim=dimensions) / denominator).sqrt()
    smooth = (
        F.smooth_l1_loss(prediction, target, reduction="none") * reconstruction_mask
    ).sum(dim=dimensions) / denominator
    return {
        "mae_per_channel": mae,
        "rmse_per_channel": rmse,
        "smooth_l1_per_channel": smooth,
        "mae_aggregate": mae.mean(),
        "rmse_aggregate": rmse.mean(),
        "smooth_l1_aggregate": smooth.mean(),
    }


def _stage_metrics(stages, target, region, epsilon):
    output = {}
    for name, value in stages.items():
        output[name] = {
            "occupancy": occupancy_metrics(value, target, region, epsilon),
            "continuous": per_channel_continuous_metrics(
                value, target, region, epsilon
            ),
        }
    return output


def reconstruction_stage_metrics(
    erased_lidar_bev,
    coarse_lidar_bev,
    final_lidar_bev,
    faulty_lidar_bev,
    clean_lidar_bev,
    reconstruction_mask,
    epsilon=1e-8,
):
    stages = {
        "erased": erased_lidar_bev,
        "coarse": coarse_lidar_bev,
        "final": final_lidar_bev,
    }
    output = _stage_metrics(stages, clean_lidar_bev, reconstruction_mask, epsilon)
    full_scene_mask = torch.ones_like(reconstruction_mask)
    output["full_scene"] = _stage_metrics(
        stages, clean_lidar_bev, full_scene_mask, epsilon
    )
    actual_fault_subset = reconstruction_mask * (
        (clean_lidar_bev - faulty_lidar_bev).abs().amax(dim=1, keepdim=True)
        > epsilon
    ).to(reconstruction_mask.dtype)
    sacrificed_healthy_subset = reconstruction_mask * (1 - actual_fault_subset)
    output["diagnostic_subregions"] = {
        "actual_fault": _stage_metrics(
            stages, clean_lidar_bev, actual_fault_subset, epsilon
        ),
        "sacrificed_healthy": _stage_metrics(
            stages, clean_lidar_bev, sacrificed_healthy_subset, epsilon
        ),
    }
    coarse_mae = output["coarse"]["continuous"]["mae_per_channel"].mean()
    final_mae = output["final"]["continuous"]["mae_per_channel"].mean()
    output["diffusion_improvement"] = coarse_mae - final_mae
    output["relative_diffusion_improvement"] = (
        coarse_mae - final_mae
    ) / (coarse_mae + epsilon)
    output["coarse_masked_mae"] = coarse_mae
    output["final_masked_mae"] = final_mae
    outside = 1 - reconstruction_mask
    output["outside_mask_max_change"] = (
        outside * (final_lidar_bev - faulty_lidar_bev)
    ).abs().max()
    output["coarse_outside_mask_max_change"] = (
        outside * (coarse_lidar_bev - faulty_lidar_bev)
    ).abs().max()
    return output
