"""Deterministic hard voxelization with explicit 3D spatial features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import VoxelGridConfig


DECORATION_NAMES = (
    "x_minus_voxel_centroid",
    "y_minus_voxel_centroid",
    "z_minus_voxel_centroid",
    "x_minus_voxel_center",
    "y_minus_voxel_center",
    "z_minus_voxel_center",
)


@dataclass(frozen=True)
class VoxelizedPointCloud:
    """Sparse hard-voxel result.

    ``voxel_points`` has shape ``[N, K, C_raw + 6]``.  The first ``C_raw``
    channels preserve every input field; the final six are XYZ offsets from
    the full (pre-truncation) voxel centroid and geometric voxel center.
    """

    voxel_coords: np.ndarray
    voxel_points: np.ndarray
    num_points: np.ndarray
    original_num_points: np.ndarray
    raw_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    input_point_count: int
    valid_point_count: int
    nonfinite_point_count: int
    out_of_range_point_count: int
    truncated_point_count: int

    @property
    def occupied_voxel_count(self) -> int:
        return int(self.voxel_coords.shape[0])


class HardVoxelizer:
    """NumPy hard voxelizer using half-open physical bounds.

    Input points are ``[x, y, z, ...]``.  Output sparse coordinates are
    lexicographically sorted ``[z_index, y_index, x_index]``.  Points retain
    their input order inside a voxel, making truncation deterministic.
    """

    def __init__(
        self,
        grid: VoxelGridConfig,
        *,
        max_points_per_voxel: int | None = None,
    ) -> None:
        grid.validate()
        if max_points_per_voxel is not None and max_points_per_voxel < 1:
            raise ValueError("max_points_per_voxel must be positive or null")
        self.grid = grid
        self.max_points_per_voxel = max_points_per_voxel

    def point_indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return integer ``zyx`` indices and a validity mask for input rows."""
        points = self._validate_points(points)
        xyz = points[:, :3].astype(np.float64, copy=False)
        finite = np.isfinite(points).all(axis=1)
        mins = np.asarray(self.grid.mins_xyz, dtype=np.float64)
        maxs = np.asarray(self.grid.maxs_xyz, dtype=np.float64)
        inside = finite & np.all(xyz >= mins, axis=1) & np.all(xyz < maxs, axis=1)
        indices_xyz = np.zeros((len(points), 3), dtype=np.int64)
        if inside.any():
            sizes = np.asarray(self.grid.voxel_size, dtype=np.float64)
            indices_xyz[inside] = np.floor((xyz[inside] - mins) / sizes).astype(
                np.int64
            )
            dimensions = np.asarray(self.grid.dimensions_xyz, dtype=np.int64)
            if np.any(indices_xyz[inside] < 0) or np.any(
                indices_xyz[inside] >= dimensions
            ):
                raise RuntimeError("Internal voxel indexing escaped validated bounds")
        return indices_xyz[:, ::-1], inside

    def voxel_centers(self, voxel_coords: np.ndarray) -> np.ndarray:
        """Convert integer ``[z,y,x]`` coordinates to physical ``[x,y,z]``."""
        coordinates = np.asarray(voxel_coords)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("voxel_coords must have shape [N, 3]")
        if not np.issubdtype(coordinates.dtype, np.integer):
            raise TypeError("voxel_coords must have an integer dtype")
        dimensions = np.asarray(self.grid.dimensions_zyx, dtype=np.int64)
        if np.any(coordinates < 0) or np.any(coordinates >= dimensions):
            raise ValueError("voxel_coords contain out-of-range indices")
        indices_xyz = coordinates[:, ::-1].astype(np.float64, copy=False)
        mins = np.asarray(self.grid.mins_xyz, dtype=np.float64)
        sizes = np.asarray(self.grid.voxel_size, dtype=np.float64)
        return (mins + (indices_xyz + 0.5) * sizes).astype(np.float32)

    def voxelize(
        self,
        points: np.ndarray,
        raw_feature_names: tuple[str, ...] | list[str],
    ) -> VoxelizedPointCloud:
        points = self._validate_points(points)
        raw_names = tuple(str(name) for name in raw_feature_names)
        if len(raw_names) != points.shape[1]:
            raise ValueError(
                "raw_feature_names must describe every input column; "
                f"got {len(raw_names)} names for {points.shape[1]} columns"
            )
        if raw_names[:3] != ("x", "y", "z"):
            raise ValueError("The first three raw feature names must be x, y, z")

        coordinates, valid = self.point_indices(points)
        finite = np.isfinite(points).all(axis=1)
        valid_points = points[valid].astype(np.float32, copy=False)
        valid_coords = coordinates[valid]
        feature_names = raw_names + DECORATION_NAMES
        if len(valid_points) == 0:
            capacity = self.max_points_per_voxel or 0
            return VoxelizedPointCloud(
                voxel_coords=np.empty((0, 3), dtype=np.int32),
                voxel_points=np.zeros(
                    (0, capacity, len(feature_names)), dtype=np.float32
                ),
                num_points=np.empty((0,), dtype=np.int32),
                original_num_points=np.empty((0,), dtype=np.int32),
                raw_feature_names=raw_names,
                feature_names=feature_names,
                input_point_count=len(points),
                valid_point_count=0,
                nonfinite_point_count=int((~finite).sum()),
                out_of_range_point_count=int((finite & ~valid).sum()),
                truncated_point_count=0,
            )

        nx, ny, _ = self.grid.dimensions_xyz
        flat = (
            (valid_coords[:, 0].astype(np.int64) * ny + valid_coords[:, 1]) * nx
            + valid_coords[:, 2]
        )
        order = np.argsort(flat, kind="stable")
        sorted_flat = flat[order]
        sorted_points = valid_points[order]
        starts = np.flatnonzero(
            np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
        )
        ends = np.r_[starts[1:], len(sorted_flat)]
        original_counts = (ends - starts).astype(np.int32)
        capacity = (
            int(original_counts.max())
            if self.max_points_per_voxel is None
            else self.max_points_per_voxel
        )
        retained_counts = np.minimum(original_counts, capacity).astype(np.int32)

        unique_flat = sorted_flat[starts]
        iz = unique_flat // (ny * nx)
        remainder = unique_flat % (ny * nx)
        iy = remainder // nx
        ix = remainder % nx
        voxel_coords = np.stack((iz, iy, ix), axis=1).astype(np.int32)
        centers = self.voxel_centers(voxel_coords)
        output = np.zeros(
            (len(starts), capacity, len(feature_names)), dtype=np.float32
        )
        for voxel_index, (start, end, retained) in enumerate(
            zip(starts, ends, retained_counts)
        ):
            full_group = sorted_points[start:end]
            kept = full_group[:retained]
            centroid = full_group[:, :3].mean(axis=0, dtype=np.float64).astype(
                np.float32
            )
            output[voxel_index, :retained, : points.shape[1]] = kept
            offset = points.shape[1]
            output[voxel_index, :retained, offset : offset + 3] = (
                kept[:, :3] - centroid
            )
            output[voxel_index, :retained, offset + 3 : offset + 6] = (
                kept[:, :3] - centers[voxel_index]
            )

        return VoxelizedPointCloud(
            voxel_coords=voxel_coords,
            voxel_points=output,
            num_points=retained_counts,
            original_num_points=original_counts,
            raw_feature_names=raw_names,
            feature_names=feature_names,
            input_point_count=len(points),
            valid_point_count=len(valid_points),
            nonfinite_point_count=int((~finite).sum()),
            out_of_range_point_count=int((finite & ~valid).sum()),
            truncated_point_count=int((original_counts - retained_counts).sum()),
        )

    @staticmethod
    def _validate_points(points: np.ndarray) -> np.ndarray:
        array = np.asarray(points)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError("points must have shape [N, C] with C >= 3")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("points must use a numeric dtype")
        return array
