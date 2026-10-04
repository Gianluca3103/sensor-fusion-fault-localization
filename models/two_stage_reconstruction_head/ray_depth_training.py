"""Training-only clean-LiDAR targets for ray-depth blueprints.

Clean scans enter here after the inference path has produced its blueprint.
The teacher feature term is opt-in: a randomly initialized teacher offers no
useful geometric target. Freeze or EMA-train that encoder before enabling it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .cross_modal_encoders import EncoderGrid
from .range_view.geometry import RangeGeometry
from .ray_depth_attention import RayDepthBlueprint


def clean_first_return_targets(geometry: RangeGeometry, rows: torch.Tensor,
                               cols: torch.Tensor, clean_lidar: torch.Tensor,
                               clean_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project clean returns onto calibrated rays and keep the nearest hit."""
    if (clean_lidar.ndim != 3 or clean_lidar.shape[-1] != 4 or
            clean_valid.shape != clean_lidar.shape[:2] or
            clean_valid.dtype != torch.bool or rows.shape != cols.shape or
            rows.shape[0] != clean_lidar.shape[0]):
        raise ValueError("Incompatible clean scan, point mask, or ray indices")
    height, width = geometry.shape
    beams = clean_lidar.new_tensor(geometry.beam_elevations_rad)
    if geometry.max_beam_error_rad is None:
        spacing = beams[1:] - beams[:-1]
        tolerance = torch.empty_like(beams)
        tolerance[0], tolerance[-1] = spacing[0] / 2, spacing[-1] / 2
        if height > 2:
            tolerance[1:-1] = torch.maximum(spacing[:-1], spacing[1:]) / 2
    else:
        tolerance = beams.new_full((height,), geometry.max_beam_error_rad)
    target = clean_lidar.new_full(rows.shape, torch.nan)
    returned = torch.zeros_like(rows, dtype=torch.bool)
    for scene in range(clean_lidar.shape[0]):
        xyz = clean_lidar[scene, clean_valid[scene], :3]
        if not len(xyz):
            continue
        radius = xyz.norm(dim=-1)
        elevation = torch.atan2(xyz[:, 2], xyz[:, :2].norm(dim=-1))
        row = (elevation[:, None] - beams[None, :]).abs().argmin(-1)
        relative_azimuth = torch.remainder(
            torch.atan2(xyz[:, 1], xyz[:, 0]) - geometry.azimuth_offset_rad,
            2 * math.pi,
        )
        col = torch.floor(relative_azimuth * width / geometry.azimuth_span_rad).long()
        keep = (torch.isfinite(xyz).all(-1) &
                (radius >= geometry.min_range_m) & (radius <= geometry.max_range_m) &
                ((elevation - beams[row]).abs() <= tolerance[row]) &
                (relative_azimuth < geometry.azimuth_span_rad) &
                (col >= 0) & (col < width))
        if not bool(keep.any()):
            continue
        flat = row[keep] * width + col[keep]
        depth_map = radius.new_full((height * width,), torch.inf)
        depth_map.scatter_reduce_(0, flat, radius[keep], reduce="amin", include_self=True)
        value = depth_map[rows[scene] * width + cols[scene]]
        returned[scene] = torch.isfinite(value)
        target[scene] = value.masked_fill(~returned[scene], torch.nan)
    return target, returned


def ray_depth_blueprint_loss(
    blueprint: RayDepthBlueprint, geometry: RangeGeometry,
    clean_lidar: torch.Tensor, clean_valid: torch.Tensor, *,
    grid: EncoderGrid | None = None, depth_weight: float = 1.0,
    teacher_weight: float = 0.0, max_residual_m: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Supervise first return, no return, metric depth and optional patch feature.

    Returns reported ``coverage``: fraction of clean positive rays with an
    evidence-supported candidate within max_residual_m. Unsupported clean
    returns are trained to abstain rather than hallucinate from scene priors.
    """
    if teacher_weight and (blueprint.clean_teacher is None or grid is None):
        raise ValueError("Teacher feature loss needs a clean teacher and grid")
    target_depth, clean_return = clean_first_return_targets(
        geometry, blueprint.queries.rows, blueprint.queries.cols,
        clean_lidar, clean_valid,
    )
    candidate_depth = blueprint.queries.depths_m
    supported = blueprint.queries.valid & (blueprint.evidence_weights[..., :2].sum(-1) > 0)
    error = (candidate_depth - target_depth[..., None]).abs()
    error = error.masked_fill(~supported | ~clean_return[..., None], torch.inf)
    nearest_error, nearest_index = error.min(-1)
    covered = clean_return & (nearest_error <= max_residual_m)
    no_return_index = candidate_depth.shape[-1]
    labels = torch.where(covered, nearest_index,
                         torch.full_like(nearest_index, no_return_index))
    classification = F.cross_entropy(
        blueprint.first_return_logits.reshape(-1, no_return_index + 1),
        labels.reshape(-1),
    )
    predicted_depth = (candidate_depth + blueprint.depth_residual_m).gather(
        -1, nearest_index[..., None]).squeeze(-1)
    if bool(covered.any()):
        depth_loss = F.smooth_l1_loss(predicted_depth[covered], target_depth[covered])
    else:
        depth_loss = blueprint.features.sum() * 0
    teacher_loss = depth_loss * 0
    if teacher_weight and bool(covered.any()):
        assert grid is not None and blueprint.clean_teacher is not None
        selected = blueprint.features.gather(
            2, nearest_index[..., None, None].expand(-1, -1, 1,
                                                      blueprint.features.shape[-1])
        ).squeeze(2)
        clean_xyz = blueprint.queries.directions * target_depth.nan_to_num()[..., None]
        minimum = clean_xyz.new_tensor(grid.minimum_xyz)
        step = clean_xyz.new_tensor(grid.voxel_size_xyz)
        xyz_index = torch.floor((clean_xyz - minimum) / step).long()
        zyx_size = grid.shape_zyx
        in_grid = (xyz_index[..., 0].ge(0) & xyz_index[..., 0].lt(zyx_size[2]) &
                   xyz_index[..., 1].ge(0) & xyz_index[..., 1].lt(zyx_size[1]) &
                   xyz_index[..., 2].ge(0) & xyz_index[..., 2].lt(zyx_size[0]))
        x = xyz_index[..., 0].clamp(0, zyx_size[2] - 1)
        y = xyz_index[..., 1].clamp(0, zyx_size[1] - 1)
        z = xyz_index[..., 2].clamp(0, zyx_size[0] - 1)
        batch = torch.arange(len(x), device=x.device)[:, None]
        teacher = blueprint.clean_teacher.features[batch, :, z, y, x]
        occupied = blueprint.clean_teacher.occupied[batch, 0, z, y, x]
        use = covered & in_grid & occupied
        if bool(use.any()):
            teacher_loss = F.smooth_l1_loss(selected[use], teacher[use].detach())
    total = classification + depth_weight * depth_loss + teacher_weight * teacher_loss
    coverage = covered.sum().float() / clean_return.sum().clamp_min(1)
    return {"loss": total, "classification": classification,
            "depth": depth_loss, "teacher": teacher_loss,
            "coverage": coverage.detach()}
