"""Sparse, selector-local 3D examples for diffusion reconstruction.

The 3D refiner operates on a fixed candidate lattice inside one selector crop.
This is important: missing LiDAR voxels are candidates even though they are not
present in the faulty sparse cloud, so a sparse model can add them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..voxelization.config import VoxelGridConfig
from ..voxelization.fault_selector_3d import (
    OracleFaultComponent3D,
    OracleFaultSelection3D,
)
from ..voxelization.fault_targets import VoxelFaultTargets
from ..voxelization.hard_voxelizer import VoxelizedPointCloud


@dataclass(frozen=True)
class SparseVoxelExample:
    """One unpadded 3D fault-component example.

    Coordinates are global integer ``zyx`` indices.  Condition features are
    ``[log(1+faulty_count), log(1+radar_count), repair, remove, halo,
    trusted_faulty]``.  The target is clean occupancy; it deliberately does
    not encode a variable-length point list.
    """

    coords_zyx: torch.Tensor
    coords_xyz_m: torch.Tensor
    condition_features: torch.Tensor
    target_occupancy: torch.Tensor
    faulty_occupancy: torch.Tensor
    editable_mask: torch.Tensor


@dataclass(frozen=True)
class SparseVoxelBatch:
    """Padded batch consumed by :class:`SparseVoxelDiffusionBaseline`."""

    coords_zyx: torch.Tensor
    coords_xyz_m: torch.Tensor
    condition_features: torch.Tensor
    target_occupancy: torch.Tensor
    faulty_occupancy: torch.Tensor
    editable_mask: torch.Tensor
    valid_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.coords_zyx.shape[0])

    def to(self, *args, **kwargs) -> "SparseVoxelBatch":
        """Move every batched tensor while preserving integer coordinates."""

        target = self.target_occupancy.to(*args, **kwargs)
        return SparseVoxelBatch(
            coords_zyx=self.coords_zyx.to(device=target.device),
            coords_xyz_m=self.coords_xyz_m.to(*args, **kwargs),
            condition_features=self.condition_features.to(*args, **kwargs),
            target_occupancy=target,
            faulty_occupancy=self.faulty_occupancy.to(*args, **kwargs),
            editable_mask=self.editable_mask.to(*args, **kwargs),
            valid_mask=self.valid_mask.to(*args, **kwargs),
        )


def _crop_slices(component: OracleFaultComponent3D) -> tuple[slice, slice, slice]:
    lower = component.crop_min_zyx
    upper = component.crop_max_exclusive_zyx
    return tuple(slice(start, stop) for start, stop in zip(lower, upper))


def _occupancy_counts_in_crop(
    cloud: VoxelizedPointCloud,
    component: OracleFaultComponent3D,
) -> np.ndarray:
    """Return sparse-cloud point counts aligned to the selector crop."""

    lower = np.asarray(component.crop_min_zyx, dtype=np.int32)
    upper = np.asarray(component.crop_max_exclusive_zyx, dtype=np.int32)
    shape = tuple((upper - lower).tolist())
    output = np.zeros(shape, dtype=np.float32)
    coords = np.asarray(cloud.voxel_coords, dtype=np.int32)
    keep = np.all((coords >= lower) & (coords < upper), axis=1)
    if not np.any(keep):
        return output
    local = coords[keep] - lower
    output[local[:, 0], local[:, 1], local[:, 2]] = np.asarray(
        cloud.original_num_points[keep], dtype=np.float32
    )
    return output


def _coords_xyz_m(coords_zyx: np.ndarray, grid: VoxelGridConfig) -> np.ndarray:
    # Coordinates are integer zyx, but physical coordinates are xyz.
    xyz_indices = coords_zyx[:, ::-1].astype(np.float32, copy=False)
    return np.asarray(grid.mins_xyz, dtype=np.float32) + (
        xyz_indices + 0.5
    ) * np.asarray(grid.voxel_size, dtype=np.float32)


def build_sparse_voxel_example(
    *,
    faulty_lidar: VoxelizedPointCloud,
    radar: VoxelizedPointCloud,
    targets: VoxelFaultTargets,
    selection: OracleFaultSelection3D,
    component: OracleFaultComponent3D,
    grid: VoxelGridConfig,
) -> SparseVoxelExample:
    """Build a selector-local sparse candidate lattice.

    The context mask supplies empty candidates around faults; unioning the two
    sensor occupancies means that usable evidence is never discarded when a
    selector crop is tight.
    """

    grid.validate()
    slices = _crop_slices(component)
    # Sensor evidence comes from the sparse cache rather than from target
    # arrays, preventing clean-label information from leaking into inference.
    faulty_count = _occupancy_counts_in_crop(faulty_lidar, component)
    radar_count = _occupancy_counts_in_crop(radar, component)
    repair = np.asarray(selection.repair_core[slices], dtype=bool)
    remove = np.asarray(selection.remove_core[slices], dtype=bool)
    halo = np.asarray(selection.context_halo[slices], dtype=bool)
    context = np.asarray(selection.context_mask[slices], dtype=bool)
    faulty_occupancy = faulty_count > 0
    # Candidate positions include all selected empty volume and every sensor
    # return in the crop.  This lets the model generate missing voxels while
    # retaining radar-only conditioning locations.
    candidate = context | faulty_occupancy | (radar_count > 0)
    local_coords = np.argwhere(candidate).astype(np.int64)
    if len(local_coords) == 0:
        raise ValueError("A selected 3D component produced no candidate voxels")
    global_coords = local_coords + np.asarray(component.crop_min_zyx, dtype=np.int64)
    index = tuple(local_coords[:, axis] for axis in range(3))
    editable = repair[index] | remove[index]
    trusted = faulty_occupancy[index] & ~editable
    features = np.stack(
        (
            np.log1p(faulty_count[index]),
            np.log1p(radar_count[index]),
            repair[index].astype(np.float32),
            remove[index].astype(np.float32),
            halo[index].astype(np.float32),
            trusted.astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    target = np.asarray(targets.clean_occupancy[slices], dtype=np.float32)[index]
    return SparseVoxelExample(
        coords_zyx=torch.from_numpy(global_coords),
        coords_xyz_m=torch.from_numpy(_coords_xyz_m(global_coords, grid)),
        condition_features=torch.from_numpy(features),
        target_occupancy=torch.from_numpy(target[:, None]),
        faulty_occupancy=torch.from_numpy(
            faulty_occupancy[index].astype(np.float32)[:, None]
        ),
        editable_mask=torch.from_numpy(editable.astype(np.float32)[:, None]),
    )


def collate_sparse_voxel_examples(
    examples: Sequence[SparseVoxelExample],
) -> SparseVoxelBatch:
    """Pad variable-size selector components without inventing valid voxels."""

    if not examples:
        raise ValueError("Cannot collate an empty sparse-voxel batch")
    feature_width = examples[0].condition_features.shape[1]
    if any(item.condition_features.shape[1] != feature_width for item in examples):
        raise ValueError("All sparse examples must use the same feature width")
    maximum = max(len(item.coords_zyx) for item in examples)
    batch_size = len(examples)
    def zeros(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)
    coords = zeros(batch_size, maximum, 3, dtype=torch.long)
    xyz = zeros(batch_size, maximum, 3, dtype=torch.float32)
    condition = zeros(batch_size, maximum, feature_width, dtype=torch.float32)
    target = zeros(batch_size, maximum, 1, dtype=torch.float32)
    faulty = zeros(batch_size, maximum, 1, dtype=torch.float32)
    editable = zeros(batch_size, maximum, 1, dtype=torch.float32)
    valid = zeros(batch_size, maximum, 1, dtype=torch.float32)
    for batch_index, item in enumerate(examples):
        count = len(item.coords_zyx)
        coords[batch_index, :count] = item.coords_zyx
        xyz[batch_index, :count] = item.coords_xyz_m
        condition[batch_index, :count] = item.condition_features
        target[batch_index, :count] = item.target_occupancy
        faulty[batch_index, :count] = item.faulty_occupancy
        editable[batch_index, :count] = item.editable_mask
        valid[batch_index, :count] = 1.0
    return SparseVoxelBatch(coords, xyz, condition, target, faulty, editable, valid)
