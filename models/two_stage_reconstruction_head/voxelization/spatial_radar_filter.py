"""Purely spatial 3D occupied-voxel filtering for radar point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import VoxelGridConfig
from .hard_voxelizer import HardVoxelizer


@dataclass(frozen=True)
class SpatialRadarVoxelFilterConfig:
    """Reject isolated occupied voxels without using scan timestamps."""

    neighbor_radius_cells: int = 2
    min_neighbor_voxels: int = 1

    def validate(self) -> None:
        if self.neighbor_radius_cells < 1:
            raise ValueError("neighbor_radius_cells must be at least one")
        if self.min_neighbor_voxels < 1:
            raise ValueError("min_neighbor_voxels must be at least one")


def filter_spatially_isolated_radar_voxels(
    radar_points: np.ndarray,
    grid: VoxelGridConfig,
    config: SpatialRadarVoxelFilterConfig,
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep points whose occupied 3D voxel has enough occupied neighbors.

    Neighbor support is computed over the combined aligned stack and does not
    inspect or require a time-index column. The central voxel is excluded from
    its own support count, so an isolated dense voxel remains isolated.
    """
    config.validate()
    points = np.asarray(radar_points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("radar_points must have shape [N,C] with C >= 3")
    if len(points) == 0:
        return points.copy(), {
            "input_points": 0,
            "in_grid_points": 0,
            "out_of_grid_points": 0,
            "input_occupied_voxels": 0,
            "supported_occupied_voxels": 0,
            "rejected_occupied_voxels": 0,
            "rejected_points": 0,
            "output_points": 0,
            "neighbor_radius_cells": config.neighbor_radius_cells,
            "min_neighbor_voxels": config.min_neighbor_voxels,
        }

    voxelizer = HardVoxelizer(grid)
    coordinates, valid = voxelizer.point_indices(points)
    nz, ny, nx = grid.dimensions_zyx
    valid_coordinates = coordinates[valid]
    valid_flat = (
        (valid_coordinates[:, 0].astype(np.int64) * ny + valid_coordinates[:, 1])
        * nx
        + valid_coordinates[:, 2]
    )
    occupied = set(int(value) for value in np.unique(valid_flat))
    supported: set[int] = set()
    radius = config.neighbor_radius_cells
    for flat in occupied:
        z = flat // (ny * nx)
        remainder = flat % (ny * nx)
        y = remainder // nx
        x = remainder % nx
        neighbor_count = 0
        stop = False
        for dz in range(-radius, radius + 1):
            target_z = z + dz
            if not 0 <= target_z < nz:
                continue
            for dy in range(-radius, radius + 1):
                target_y = y + dy
                if not 0 <= target_y < ny:
                    continue
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    target_x = x + dx
                    if not 0 <= target_x < nx:
                        continue
                    neighbor = (target_z * ny + target_y) * nx + target_x
                    if neighbor in occupied:
                        neighbor_count += 1
                        if neighbor_count >= config.min_neighbor_voxels:
                            supported.add(flat)
                            stop = True
                            break
                if stop:
                    break
            if stop:
                break

    keep_valid = np.fromiter(
        (int(flat) in supported for flat in valid_flat),
        dtype=bool,
        count=len(valid_flat),
    )
    keep = np.zeros(len(points), dtype=bool)
    keep[np.flatnonzero(valid)] = keep_valid
    return points[keep].copy(), {
        "input_points": len(points),
        "in_grid_points": int(valid.sum()),
        "out_of_grid_points": int((~valid).sum()),
        "input_occupied_voxels": len(occupied),
        "supported_occupied_voxels": len(supported),
        "rejected_occupied_voxels": len(occupied) - len(supported),
        "rejected_points": len(points) - int(keep.sum()),
        "output_points": int(keep.sum()),
        "neighbor_radius_cells": radius,
        "min_neighbor_voxels": config.min_neighbor_voxels,
    }
