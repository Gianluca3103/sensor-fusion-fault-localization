"""Deterministic oracle selection of 3D voxel fault regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import VoxelGridConfig


@dataclass(frozen=True)
class OracleFaultSelector3DConfig:
    halo_m: float = 1.0
    grouping_radius_m: float = 0.4
    connectivity: int = 26
    min_component_voxels: int = 1
    min_crop_shape_zyx: tuple[int, int, int] = (4, 10, 10)

    def validate(self) -> None:
        if self.halo_m < 0 or self.grouping_radius_m < 0:
            raise ValueError("halo and grouping radii must be non-negative")
        if self.connectivity not in {6, 18, 26}:
            raise ValueError("connectivity must be 6, 18, or 26")
        if self.min_component_voxels < 1:
            raise ValueError("min_component_voxels must be positive")
        if len(self.min_crop_shape_zyx) != 3 or any(
            int(value) < 1 for value in self.min_crop_shape_zyx
        ):
            raise ValueError("min_crop_shape_zyx must contain three positive values")


@dataclass(frozen=True)
class OracleFaultComponent3D:
    component_id: int
    operation_voxels: int
    repair_voxels: int
    remove_voxels: int
    core_min_zyx: tuple[int, int, int]
    core_max_exclusive_zyx: tuple[int, int, int]
    crop_min_zyx: tuple[int, int, int]
    crop_max_exclusive_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class OracleFaultSelection3D:
    repair_core: np.ndarray
    remove_core: np.ndarray
    operation_mask: np.ndarray
    context_mask: np.ndarray
    context_halo: np.ndarray
    grouping_mask: np.ndarray
    component_labels: np.ndarray
    components: tuple[OracleFaultComponent3D, ...]


def _distance_dilation(mask: np.ndarray, radius_m: float, grid: VoxelGridConfig) -> np.ndarray:
    if radius_m <= 0 or not np.any(mask):
        return mask.copy()
    vx, vy, vz = grid.voxel_size
    distance = ndimage.distance_transform_edt(~mask, sampling=(vz, vy, vx))
    return distance <= radius_m


def _connectivity_structure(connectivity: int) -> np.ndarray:
    rank = {6: 1, 18: 2, 26: 3}[connectivity]
    return ndimage.generate_binary_structure(3, rank)


def _expand_interval(
    lower: int,
    upper: int,
    requested_size: int,
    limit: int,
) -> tuple[int, int]:
    requested_size = min(int(requested_size), int(limit))
    current = upper - lower
    missing = max(0, requested_size - current)
    lower -= missing // 2
    upper += missing - missing // 2
    if lower < 0:
        upper = min(limit, upper - lower)
        lower = 0
    if upper > limit:
        lower = max(0, lower - (upper - limit))
        upper = limit
    return int(lower), int(upper)


def _expanded_crop(
    core_min: np.ndarray,
    core_max: np.ndarray,
    shape: tuple[int, int, int],
    grid: VoxelGridConfig,
    config: OracleFaultSelector3DConfig,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    vx, vy, vz = grid.voxel_size
    padding = np.ceil(
        config.halo_m / np.asarray((vz, vy, vx), dtype=np.float64)
    ).astype(int)
    lower = np.maximum(0, core_min - padding)
    upper = np.minimum(np.asarray(shape), core_max + padding)
    for axis in range(3):
        lower[axis], upper[axis] = _expand_interval(
            int(lower[axis]),
            int(upper[axis]),
            int(config.min_crop_shape_zyx[axis]),
            int(shape[axis]),
        )
    return tuple(int(value) for value in lower), tuple(int(value) for value in upper)


def select_oracle_fault_regions_3d(
    repair_mask: np.ndarray,
    remove_mask: np.ndarray,
    grid: VoxelGridConfig,
    config: OracleFaultSelector3DConfig = OracleFaultSelector3DConfig(),
) -> OracleFaultSelection3D:
    """Convert exact operation masks into context-aware 3D selector output.

    Halo and grouping never alter the operation targets.  ``context_halo`` is
    explicitly disjoint from ``operation_mask`` so a consumer cannot
    accidentally learn to reconstruct contextual voxels.
    """

    config.validate()
    grid.validate()
    repair = np.asarray(repair_mask, dtype=bool)
    remove = np.asarray(remove_mask, dtype=bool)
    expected_shape = grid.dimensions_zyx
    if repair.shape != expected_shape or remove.shape != expected_shape:
        raise ValueError(
            f"repair/remove masks must both have shape {expected_shape}; "
            f"got {repair.shape} and {remove.shape}"
        )
    operation = repair | remove
    if not np.any(operation):
        empty_labels = np.zeros(expected_shape, dtype=np.int32)
        return OracleFaultSelection3D(
            repair_core=repair,
            remove_core=remove,
            operation_mask=operation,
            context_mask=operation.copy(),
            context_halo=np.zeros(expected_shape, dtype=bool),
            grouping_mask=operation.copy(),
            component_labels=empty_labels,
            components=(),
        )

    grouping = _distance_dilation(operation, config.grouping_radius_m, grid)
    raw_labels, raw_count = ndimage.label(
        grouping, structure=_connectivity_structure(config.connectivity)
    )
    keep_raw_ids = []
    for raw_id in range(1, raw_count + 1):
        core_count = int(np.sum(operation & (raw_labels == raw_id)))
        if core_count >= config.min_component_voxels:
            keep_raw_ids.append(raw_id)
    kept_grouping = np.isin(raw_labels, keep_raw_ids)
    filtered_operation = operation & kept_grouping
    repair_core = repair & filtered_operation
    remove_core = remove & filtered_operation
    context = _distance_dilation(filtered_operation, config.halo_m, grid)
    context_halo = context & ~filtered_operation

    component_labels = np.zeros(expected_shape, dtype=np.int32)
    components = []
    for new_id, raw_id in enumerate(keep_raw_ids, start=1):
        grouped_component = raw_labels == raw_id
        core = operation & grouped_component
        component_labels[core] = new_id
        coordinates = np.argwhere(core)
        core_min = coordinates.min(axis=0)
        core_max = coordinates.max(axis=0) + 1
        crop_min, crop_max = _expanded_crop(
            core_min, core_max, expected_shape, grid, config
        )
        components.append(
            OracleFaultComponent3D(
                component_id=new_id,
                operation_voxels=int(core.sum()),
                repair_voxels=int((repair & core).sum()),
                remove_voxels=int((remove & core).sum()),
                core_min_zyx=tuple(int(value) for value in core_min),
                core_max_exclusive_zyx=tuple(int(value) for value in core_max),
                crop_min_zyx=crop_min,
                crop_max_exclusive_zyx=crop_max,
            )
        )
    return OracleFaultSelection3D(
        repair_core=repair_core,
        remove_core=remove_core,
        operation_mask=filtered_operation,
        context_mask=context,
        context_halo=context_halo,
        grouping_mask=kept_grouping,
        component_labels=component_labels,
        components=tuple(components),
    )
