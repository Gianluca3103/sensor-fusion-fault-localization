"""Conservative original-preserving merge and generated-point provenance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RangeGeometry, RangeProjection, backproject


@dataclass(frozen=True)
class MergeConfig:
    allow_original_deletion: bool = False
    delete_threshold: float = 0.999
    add_threshold: float = 0.5
    generated_min_radar_support: float = 0.0
    forward_only: bool = False
    enforce_single_return_per_cell: bool = False
    radar_anchor_radius_m: float | None = None

    def __post_init__(self) -> None:
        if not (0 < self.delete_threshold <= 1 and 0 < self.add_threshold <= 1):
            raise ValueError("add/delete thresholds must lie in (0,1]")
        if not 0 <= self.generated_min_radar_support <= 1:
            raise ValueError("generated_min_radar_support must be in [0,1]")
        if self.radar_anchor_radius_m is not None and (
            not np.isfinite(self.radar_anchor_radius_m) or self.radar_anchor_radius_m <= 0
        ):
            raise ValueError("radar_anchor_radius_m must be finite and positive")


@dataclass(frozen=True)
class MergeResult:
    points: np.ndarray
    retained_original_indices: np.ndarray
    deleted_original_indices: np.ndarray
    generated_points: np.ndarray
    generated_rows: np.ndarray
    generated_cols: np.ndarray
    generated_add_probability: np.ndarray
    generated_range_m: np.ndarray
    generated_radar_support: np.ndarray
    output_is_generated: np.ndarray
    same_ray_original_and_generated: int
    blocked_generated_occupied_cells: int
    blocked_generated_radar_distance: int


def merge_reconstruction(
    original_points: np.ndarray,
    original_projection: RangeProjection,
    geometry: RangeGeometry,
    add_probability: np.ndarray,
    add_range_m: np.ndarray,
    delete_probability: np.ndarray,
    *,
    config: MergeConfig = MergeConfig(),
    radar_support: np.ndarray | None = None,
    radar_points: np.ndarray | None = None,
    add_intensity: np.ndarray | None = None,
) -> MergeResult:
    original = np.asarray(original_points, dtype=np.float32)
    if original.ndim != 2 or original.shape[1] < 3 or len(original) != len(original_projection.point_row):
        raise ValueError("original points and projection do not align")
    if config.forward_only and np.any(original[:, 0] < 0):
        raise ValueError("forward-only merge requires front-filtered original LiDAR")
    add_p = np.asarray(add_probability, dtype=np.float32)
    add_r = np.asarray(add_range_m, dtype=np.float32)
    delete_p = np.asarray(delete_probability, dtype=np.float32)
    if add_p.shape != geometry.shape or add_r.shape != geometry.shape or delete_p.shape != geometry.shape:
        raise ValueError("prediction maps must match sensor geometry")
    support = np.zeros(geometry.shape, dtype=np.float32) if radar_support is None else np.asarray(radar_support, dtype=np.float32)
    if support.shape != geometry.shape:
        raise ValueError("radar_support must match sensor geometry")
    intensity = None if add_intensity is None else np.asarray(add_intensity, dtype=np.float32)
    if intensity is not None and (intensity.shape != geometry.shape or not np.isfinite(intensity).all()):
        raise ValueError("add_intensity must be a finite sensor-geometry map")
    point_valid = original_projection.point_valid
    point_rows = original_projection.point_row
    point_cols = original_projection.point_col
    delete = np.zeros(len(original), dtype=bool)
    if config.allow_original_deletion:
        # Skip colliding rays: a ray-level score cannot safely select among
        # multiple original measurements in one angular bin.
        safe = point_valid & (original_projection.collision_count[point_rows, point_cols] == 1)
        delete[safe] = delete_p[point_rows[safe], point_cols[safe]] >= config.delete_threshold
    retained = np.flatnonzero(~delete)
    deleted = np.flatnonzero(delete)
    retained_ray_count = np.zeros(geometry.shape, dtype=np.int32)
    retained_valid = retained[point_valid[retained]]
    if len(retained_valid):
        np.add.at(retained_ray_count,
                  (point_rows[retained_valid], point_cols[retained_valid]), 1)
    generate = (
        np.isfinite(add_p) & np.isfinite(add_r)
        & (add_p >= config.add_threshold)
        & (add_r >= geometry.min_range_m) & (add_r <= geometry.max_range_m)
        & (support >= config.generated_min_radar_support)
    )
    if config.forward_only:
        generate &= geometry.ray_directions()[:, :, 0] >= 0
    blocked = int(np.sum(generate & (retained_ray_count > 0)))
    if config.enforce_single_return_per_cell:
        # The model predicts one range per virtual cell. If an original return
        # survives there, appending either a nearer or farther point would
        # create two first returns for the same represented ray.
        generate &= retained_ray_count == 0
    rows, cols = np.nonzero(generate)
    generated_xyz = backproject(rows, cols, add_r[rows, cols], geometry)
    blocked_radar = 0
    if config.radar_anchor_radius_m is not None:
        if radar_points is None:
            raise ValueError("radar_anchor_radius_m requires aligned radar_points")
        radar_xyz = np.asarray(radar_points, dtype=np.float32)
        if radar_xyz.ndim != 2 or radar_xyz.shape[1] < 3:
            raise ValueError("radar_points must have XYZ columns")
        radar_xyz = radar_xyz[np.isfinite(radar_xyz[:, :3]).all(axis=1), :3]
        if len(radar_xyz) and len(generated_xyz):
            from scipy.spatial import cKDTree
            distances, _ = cKDTree(radar_xyz).query(generated_xyz, k=1)
            near_radar = distances <= config.radar_anchor_radius_m
        else:
            near_radar = np.zeros(len(generated_xyz), dtype=bool)
        blocked_radar = int((~near_radar).sum())
        rows, cols, generated_xyz = rows[near_radar], cols[near_radar], generated_xyz[near_radar]
    generated = np.zeros((len(rows), original.shape[1]), dtype=np.float32)
    generated[:, :3] = generated_xyz
    if intensity is not None and generated.shape[1] > 3:
        generated[:, 3] = np.maximum(intensity[rows, cols], 0)
    points = np.concatenate((original[retained], generated), axis=0)
    same_ray = int(np.sum(retained_ray_count[rows, cols] > 0))
    return MergeResult(
        points=points, retained_original_indices=retained,
        deleted_original_indices=deleted, generated_points=generated,
        generated_rows=rows.astype(np.int32), generated_cols=cols.astype(np.int32),
        generated_add_probability=add_p[rows, cols], generated_range_m=add_r[rows, cols],
        generated_radar_support=support[rows, cols],
        output_is_generated=np.r_[np.zeros(len(retained), dtype=bool), np.ones(len(rows), dtype=bool)],
        same_ray_original_and_generated=same_ray,
        blocked_generated_occupied_cells=blocked if config.enforce_single_return_per_cell else 0,
        blocked_generated_radar_distance=blocked_radar,
    )
