"""LiDAR ray/depth proposals made only from inference-available measurements.

The calibrated directions belong to the sensor, not to a clean scan.  Clean
LiDAR must never be passed to this module: it is a training target only.
Callers pass a bounded tile of ray indices at a time for full-scan inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .range_view.geometry import RangeGeometry


@dataclass
class RayDepthQueries:
    rows: torch.Tensor                 # [B, Q]
    cols: torch.Tensor                 # [B, Q]
    directions: torch.Tensor           # [B, Q, 3], unit LiDAR rays
    depths_m: torch.Tensor             # [B, Q, D]
    valid: torch.Tensor                # [B, Q, D]
    source: torch.Tensor               # 0=uniform, 1=radar, 2=observed LiDAR

    @property
    def xyz(self) -> torch.Tensor:
        return self.directions[:, :, None, :] * self.depths_m[..., None]


def ray_tile_indices(
    geometry: RangeGeometry, *, row_start: int = 0, row_stop: int | None = None,
    col_start: int = 0, col_stop: int | None = None,
    batch_size: int = 1, device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one rectangular tile of calibrated ray indices, flattened."""
    height, width = geometry.shape
    row_stop = height if row_stop is None else row_stop
    col_stop = width if col_stop is None else col_stop
    if not (0 <= row_start < row_stop <= height and
            0 <= col_start < col_stop <= width and batch_size > 0):
        raise ValueError("Invalid ray tile bounds or batch size")
    rows, cols = torch.meshgrid(
        torch.arange(row_start, row_stop, device=device),
        torch.arange(col_start, col_stop, device=device), indexing="ij",
    )
    return (rows.reshape(1, -1).expand(batch_size, -1),
            cols.reshape(1, -1).expand(batch_size, -1))


def _angular_proposals(
    directions: torch.Tensor, points: torch.Tensor, valid: torch.Tensor,
    *, count: int, cosine_limit: float, min_depth: float, max_depth: float,
    separation_m: float, chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find distinct measured depths near each ray using bounded Q-by-N tiles."""
    batch, rays, _ = directions.shape
    depths = directions.new_full((batch, rays, count), min_depth)
    supported = torch.zeros((batch, rays, count), dtype=torch.bool,
                            device=directions.device)
    if count == 0:
        return depths, supported
    for scene in range(batch):
        xyz = points[scene, valid[scene], :3]
        if xyz.numel() == 0:
            continue
        radius = torch.linalg.vector_norm(xyz, dim=-1)
        keep = torch.isfinite(xyz).all(-1) & (radius > 1e-6)
        xyz, radius = xyz[keep], radius[keep]
        if xyz.numel() == 0:
            continue
        unit = xyz / radius[:, None]
        for start in range(0, rays, chunk_size):
            stop = min(start + chunk_size, rays)
            cosine = directions[scene, start:stop] @ unit.T
            projected = directions[scene, start:stop] @ xyz.T
            eligible = ((cosine >= cosine_limit) &
                        (projected >= min_depth) & (projected <= max_depth))
            # Search the complete local angular window. A 20-scan stack can
            # contain many repeats at one depth; a small top-k by angle could
            # otherwise hide a second, physically distinct depth entirely.
            scores = cosine.masked_fill(~eligible, -2.0)
            candidate_depths = projected
            available = eligible
            for slot in range(count):
                best_score, best_index = scores.masked_fill(~available, -2.0).max(dim=1)
                chosen = best_score >= cosine_limit
                chosen_depth = candidate_depths.gather(1, best_index[:, None]).squeeze(1)
                depths[scene, start:stop, slot] = torch.where(
                    chosen, chosen_depth, depths[scene, start:stop, slot]
                )
                supported[scene, start:stop, slot] = chosen
                available = available & ((candidate_depths - chosen_depth[:, None]).abs() >= separation_m)
    return depths, supported


def propose_ray_depth_queries(
    geometry: RangeGeometry,
    rows: torch.Tensor,
    cols: torch.Tensor,
    radar: torch.Tensor,
    radar_valid: torch.Tensor,
    observed_lidar: torch.Tensor,
    observed_valid: torch.Tensor,
    *,
    radar_slots: int = 2,
    lidar_slots: int = 1,
    uniform_slots: int = 2,
    radar_angle_rad: float = 0.08,
    lidar_angle_rad: float = 0.025,
    separation_m: float = 1.0,
    chunk_size: int = 128,
) -> RayDepthQueries:
    """Propose depths from radar, faulty LiDAR, and uniform fallback anchors.

    Uniform anchors keep a query available even in a total sensor dropout;
    they are not evidence. The later attention block may choose its null path.
    No clean depth or clean-return mask enters proposal construction.
    """
    if rows.ndim != 2 or rows.shape != cols.shape or rows.dtype != torch.long or cols.dtype != torch.long:
        raise ValueError("rows and cols must be matching [B,Q] long tensors")
    if rows.device != radar.device or cols.device != radar.device:
        raise ValueError("Ray indices and sensor tensors must share a device")
    batch, rays = rows.shape
    if rays == 0 or radar.shape[:2] != radar_valid.shape or observed_lidar.shape[:2] != observed_valid.shape:
        raise ValueError("Invalid ray or point-mask shape")
    if (radar.shape[0] != batch or radar.shape[-1] != 7 or
            observed_lidar.shape[0] != batch or observed_lidar.shape[-1] != 4):
        raise ValueError("Expected aligned VoD radar [B,N,7] and LiDAR [B,M,4]")
    if radar_valid.dtype != torch.bool or observed_valid.dtype != torch.bool:
        raise ValueError("Point masks must be boolean")
    if any(value < 0 for value in (radar_slots, lidar_slots, uniform_slots)) or not (radar_slots + lidar_slots + uniform_slots):
        raise ValueError("At least one nonnegative proposal slot is required")
    if not (0 < radar_angle_rad < math.pi / 2 and 0 < lidar_angle_rad < math.pi / 2 and
            separation_m > 0 and chunk_size > 0):
        raise ValueError("Invalid angular window, depth separation, or chunk size")
    height, width = geometry.shape
    if ((rows < 0) | (rows >= height) | (cols < 0) | (cols >= width)).any():
        raise ValueError("Ray indices outside calibrated geometry")
    lookup = torch.as_tensor(geometry.ray_directions().copy(),
                             device=radar.device, dtype=radar.dtype)
    directions = lookup[rows, cols]
    pieces = []
    masks = []
    sources = []
    for points, point_valid, count, angular_window, source_id in (
        (radar, radar_valid, radar_slots, radar_angle_rad, 1),
        (observed_lidar, observed_valid, lidar_slots, lidar_angle_rad, 2),
    ):
        if not count:
            continue
        proposed, supported = _angular_proposals(
            directions, points, point_valid, count=count,
            cosine_limit=math.cos(angular_window),
            min_depth=geometry.min_range_m, max_depth=geometry.max_range_m,
            separation_m=separation_m, chunk_size=chunk_size,
        )
        pieces.append(proposed)
        masks.append(supported)
        sources.append(torch.full_like(proposed, source_id, dtype=torch.long))
    if uniform_slots:
        anchors = torch.linspace(
            geometry.min_range_m, geometry.max_range_m,
            uniform_slots + 2, device=radar.device, dtype=radar.dtype,
        )[1:-1]
        fallback = anchors.reshape(1, 1, -1).expand(batch, rays, -1)
        pieces.append(fallback)
        masks.append(torch.ones_like(fallback, dtype=torch.bool))
        sources.append(torch.zeros_like(fallback, dtype=torch.long))
    return RayDepthQueries(rows, cols, directions, torch.cat(pieces, -1),
                           torch.cat(masks, -1), torch.cat(sources, -1))
