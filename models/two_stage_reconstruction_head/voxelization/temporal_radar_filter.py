"""Temporal-consistency filtering for aligned 3D radar voxels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import VoxelGridConfig
from .hard_voxelizer import HardVoxelizer


@dataclass(frozen=True)
class VoxelTemporalConsistencyConfig:
    """Require local 3D voxel support from multiple distinct radar scans."""

    min_scans: int = 3
    min_scan_fraction: float = 0.15
    neighbor_radius_cells: int = 1
    preserve_current_scan: bool = False

    def validate(self) -> None:
        if self.min_scans < 1:
            raise ValueError("min_scans must be at least one")
        if not 0.0 <= self.min_scan_fraction <= 1.0:
            raise ValueError("min_scan_fraction must lie in [0, 1]")
        if self.neighbor_radius_cells < 0:
            raise ValueError("neighbor_radius_cells must be non-negative")


def filter_temporally_consistent_radar_voxels(
    radar_points: np.ndarray,
    grid: VoxelGridConfig,
    config: VoxelTemporalConsistencyConfig,
    *,
    time_index_column: int = 6,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Remove points in voxels without support from enough distinct scans.

    Points are expected to have physical XYZ in their first three columns and
    an integer-valued relative scan index in ``time_index_column``. Support is
    accumulated over the Chebyshev neighborhood configured in voxel cells,
    including height; thus XY-coincident but vertically inconsistent clutter
    does not automatically survive.
    """
    config.validate()
    points = np.asarray(radar_points)
    if points.ndim != 2 or points.shape[1] <= time_index_column:
        raise ValueError(
            f"radar_points must have shape [N,C] with C>{time_index_column}"
        )
    if not np.issubdtype(points.dtype, np.number):
        raise TypeError("radar_points must use a numeric dtype")
    if len(points) == 0:
        return points.copy(), {
            "input_points": 0,
            "in_grid_points": 0,
            "out_of_grid_points": 0,
            "distinct_scans": 0,
            "required_scans": 0,
            "input_occupied_voxels": 0,
            "supported_occupied_voxels": 0,
            "rejected_occupied_voxels": 0,
            "rejected_points": 0,
            "output_points": 0,
            "neighbor_radius_cells": config.neighbor_radius_cells,
            "preserve_current_scan": config.preserve_current_scan,
        }

    time_values = points[:, time_index_column]
    finite_time = np.isfinite(time_values)
    integer_time = finite_time & (
        np.abs(time_values - np.rint(time_values)) <= 1.0e-4
    )
    voxelizer = HardVoxelizer(grid)
    coordinates, spatial_valid = voxelizer.point_indices(points)
    valid = spatial_valid & integer_time
    scans = np.zeros(len(points), dtype=np.int64)
    scans[integer_time] = np.rint(time_values[integer_time]).astype(np.int64)
    distinct_scans = np.unique(scans[valid])
    scan_count = len(distinct_scans)
    if scan_count == 0:
        return points[:0].copy(), {
            "input_points": len(points),
            "in_grid_points": 0,
            "out_of_grid_points": len(points),
            "distinct_scans": 0,
            "required_scans": 0,
            "input_occupied_voxels": 0,
            "supported_occupied_voxels": 0,
            "rejected_occupied_voxels": 0,
            "rejected_points": len(points),
            "output_points": 0,
            "neighbor_radius_cells": config.neighbor_radius_cells,
            "preserve_current_scan": config.preserve_current_scan,
        }
    required_scans = min(
        scan_count,
        max(config.min_scans, int(math.ceil(config.min_scan_fraction * scan_count))),
    )

    nz, ny, nx = grid.dimensions_zyx
    valid_coordinates = coordinates[valid]
    valid_scans = scans[valid]
    occupied_by_scan = {
        (int(scan), int(z), int(y), int(x))
        for scan, (z, y, x) in zip(valid_scans, valid_coordinates)
    }
    support: dict[int, set[int]] = {}
    radius = config.neighbor_radius_cells
    for scan, z, y, x in occupied_by_scan:
        for dz in range(-radius, radius + 1):
            target_z = z + dz
            if not 0 <= target_z < nz:
                continue
            for dy in range(-radius, radius + 1):
                target_y = y + dy
                if not 0 <= target_y < ny:
                    continue
                for dx in range(-radius, radius + 1):
                    target_x = x + dx
                    if not 0 <= target_x < nx:
                        continue
                    flat = (target_z * ny + target_y) * nx + target_x
                    support.setdefault(flat, set()).add(scan)

    valid_flat = (
        (valid_coordinates[:, 0].astype(np.int64) * ny + valid_coordinates[:, 1])
        * nx
        + valid_coordinates[:, 2]
    )
    supported_valid = np.fromiter(
        (len(support[int(flat)]) >= required_scans for flat in valid_flat),
        dtype=bool,
        count=len(valid_flat),
    )
    if config.preserve_current_scan:
        supported_valid |= valid_scans == 0
    keep = np.zeros(len(points), dtype=bool)
    keep[np.flatnonzero(valid)] = supported_valid
    input_voxels = np.unique(valid_flat)
    supported_voxels = np.unique(valid_flat[supported_valid])
    return points[keep].copy(), {
        "input_points": len(points),
        "in_grid_points": int(valid.sum()),
        "out_of_grid_points": int((~valid).sum()),
        "distinct_scans": scan_count,
        "required_scans": required_scans,
        "input_occupied_voxels": len(input_voxels),
        "supported_occupied_voxels": len(supported_voxels),
        "rejected_occupied_voxels": len(input_voxels) - len(supported_voxels),
        "rejected_points": len(points) - int(keep.sum()),
        "output_points": int(keep.sum()),
        "neighbor_radius_cells": radius,
        "preserve_current_scan": config.preserve_current_scan,
    }
