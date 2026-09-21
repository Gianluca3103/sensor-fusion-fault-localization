"""Exact provenance-derived 3D voxel targets for fault selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import VoxelGridConfig
from .hard_voxelizer import HardVoxelizer


@dataclass(frozen=True)
class VoxelFaultTargets:
    """Dense ``zyx`` targets and point-count evidence.

    ``repair_mask`` and ``remove_mask`` are intentionally independent.  A
    voxel may contain both missing clean evidence and an unreliable faulty
    return, so forcing these targets into one exclusive class would discard
    real supervision.
    """

    clean_count: np.ndarray
    faulty_count: np.ndarray
    stable_count: np.ndarray
    damaged_clean_count: np.ndarray
    unreliable_faulty_count: np.ndarray
    clean_occupancy: np.ndarray
    faulty_occupancy: np.ndarray
    preserve_mask: np.ndarray
    repair_mask: np.ndarray
    remove_mask: np.ndarray
    change_mask: np.ndarray
    repair_fraction: np.ndarray
    removal_fraction: np.ndarray
    clean_points_in_grid: int
    faulty_points_in_grid: int
    stable_points: int
    damaged_clean_points: int
    unreliable_faulty_points: int


def _counts(flat_indices: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(flat_indices, minlength=size).astype(np.uint32, copy=False)


def build_voxel_fault_targets(
    clean_points: np.ndarray,
    faulty_points: np.ndarray,
    faulty_source_ids: np.ndarray,
    grid: VoxelGridConfig,
    *,
    movement_tolerance_m: float = 0.05,
    feature_tolerance: float = 1e-4,
) -> VoxelFaultTargets:
    """Build exact 3D targets from clean-to-faulty point provenance.

    Source IDs must index rows in ``clean_points``; ``-1`` identifies a
    synthetic return.  A clean point is stable only when a derived faulty
    return remains in the same voxel, moves no farther than the movement
    tolerance, and preserves its non-XYZ features within ``feature_tolerance``.
    """

    clean = np.asarray(clean_points)
    faulty = np.asarray(faulty_points)
    source_ids = np.asarray(faulty_source_ids, dtype=np.int64)
    if clean.ndim != 2 or clean.shape[1] < 3:
        raise ValueError("clean_points must have shape [N, C] with C >= 3")
    if faulty.ndim != 2 or faulty.shape[1] < 3:
        raise ValueError("faulty_points must have shape [M, C] with C >= 3")
    if source_ids.shape != (len(faulty),):
        raise ValueError("faulty_source_ids must contain one ID per faulty point")
    if movement_tolerance_m < 0 or feature_tolerance < 0:
        raise ValueError("movement and feature tolerances must be non-negative")
    derived = source_ids >= 0
    if np.any(source_ids[derived] >= len(clean)):
        raise ValueError("faulty_source_ids contain an out-of-range clean row")

    voxelizer = HardVoxelizer(grid)
    clean_coords, clean_valid = voxelizer.point_indices(clean)
    faulty_coords, faulty_valid = voxelizer.point_indices(faulty)
    nz, ny, nx = grid.dimensions_zyx
    size = nz * ny * nx

    def flatten(coords: np.ndarray) -> np.ndarray:
        return (coords[:, 0] * ny + coords[:, 1]) * nx + coords[:, 2]

    clean_flat = flatten(clean_coords)
    faulty_flat = flatten(faulty_coords)
    clean_count = _counts(clean_flat[clean_valid], size)
    faulty_count = _counts(faulty_flat[faulty_valid], size)

    stable_faulty = np.zeros(len(faulty), dtype=bool)
    stable_source = np.zeros(len(clean), dtype=bool)
    derived_indices = np.flatnonzero(derived)
    if len(derived_indices):
        sources = source_ids[derived_indices]
        spatially_stable = (
            clean_valid[sources]
            & faulty_valid[derived_indices]
            & np.all(clean_coords[sources] == faulty_coords[derived_indices], axis=1)
            & (
                np.linalg.norm(
                    clean[sources, :3] - faulty[derived_indices, :3], axis=1
                )
                <= movement_tolerance_m
            )
        )
        shared_features = min(clean.shape[1], faulty.shape[1]) - 3
        if shared_features > 0:
            feature_stable = np.all(
                np.abs(
                    clean[sources, 3 : 3 + shared_features]
                    - faulty[derived_indices, 3 : 3 + shared_features]
                )
                <= feature_tolerance,
                axis=1,
            )
            spatially_stable &= feature_stable
        stable_faulty[derived_indices] = spatially_stable
        stable_source[sources[spatially_stable]] = True

    damaged_clean = clean_valid & ~stable_source
    unreliable_faulty = faulty_valid & ~stable_faulty
    stable_in_grid = faulty_valid & stable_faulty
    stable_count = _counts(faulty_flat[stable_in_grid], size)
    damaged_count = _counts(clean_flat[damaged_clean], size)
    unreliable_count = _counts(faulty_flat[unreliable_faulty], size)

    shape = grid.dimensions_zyx
    clean_count = clean_count.reshape(shape)
    faulty_count = faulty_count.reshape(shape)
    stable_count = stable_count.reshape(shape)
    damaged_count = damaged_count.reshape(shape)
    unreliable_count = unreliable_count.reshape(shape)
    clean_occupancy = clean_count > 0
    faulty_occupancy = faulty_count > 0
    repair_mask = damaged_count > 0
    remove_mask = unreliable_count > 0
    preserve_mask = clean_occupancy & faulty_occupancy & ~repair_mask & ~remove_mask
    change_mask = repair_mask | remove_mask
    repair_fraction = np.divide(
        damaged_count,
        clean_count,
        out=np.zeros(shape, dtype=np.float32),
        where=clean_count > 0,
    )
    removal_fraction = np.divide(
        unreliable_count,
        faulty_count,
        out=np.zeros(shape, dtype=np.float32),
        where=faulty_count > 0,
    )
    return VoxelFaultTargets(
        clean_count=clean_count,
        faulty_count=faulty_count,
        stable_count=stable_count,
        damaged_clean_count=damaged_count,
        unreliable_faulty_count=unreliable_count,
        clean_occupancy=clean_occupancy,
        faulty_occupancy=faulty_occupancy,
        preserve_mask=preserve_mask,
        repair_mask=repair_mask,
        remove_mask=remove_mask,
        change_mask=change_mask,
        repair_fraction=repair_fraction,
        removal_fraction=removal_fraction,
        clean_points_in_grid=int(clean_valid.sum()),
        faulty_points_in_grid=int(faulty_valid.sum()),
        stable_points=int(stable_in_grid.sum()),
        damaged_clean_points=int(damaged_clean.sum()),
        unreliable_faulty_points=int(unreliable_faulty.sum()),
    )
