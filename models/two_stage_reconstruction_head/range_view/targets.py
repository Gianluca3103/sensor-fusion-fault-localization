"""Independent ADD and DELETE supervision in LiDAR's angular domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import RangeProjection


@dataclass(frozen=True)
class RangeTargets:
    add: np.ndarray
    add_range_m: np.ndarray
    delete: np.ndarray
    delete_valid: np.ndarray
    keep: np.ndarray
    replace: np.ndarray
    clean_valid: np.ndarray
    clean_range_m: np.ndarray
    healthy_original: np.ndarray
    corrupted_original: np.ndarray

    def counts(self) -> dict[str, int]:
        return {name: int(getattr(self, name).sum()) for name in
                ("keep", "add", "delete", "replace", "delete_valid")}


def build_range_targets(
    faulty: RangeProjection,
    clean: RangeProjection,
    faulty_points: np.ndarray,
    clean_points: np.ndarray,
    faulty_source_ids: np.ndarray,
    *,
    range_tolerance_m: float = 0.2,
    point_tolerance_m: float = 0.05,
) -> RangeTargets:
    if faulty.valid.shape != clean.valid.shape:
        raise ValueError("faulty and clean projections require the same geometry")
    if range_tolerance_m <= 0 or point_tolerance_m < 0:
        raise ValueError("geometric tolerances must be nonnegative")
    faulty_points = np.asarray(faulty_points)
    clean_points = np.asarray(clean_points)
    source = np.asarray(faulty_source_ids, dtype=np.int64)
    if source.shape != (len(faulty_points),):
        raise ValueError("one faulty_source_id is required per original point")
    if np.any(source >= len(clean_points)):
        raise ValueError("faulty_source_ids exceed clean LiDAR row count")
    derived = source >= 0
    healthy = np.zeros(len(source), dtype=bool)
    if np.any(derived):
        indices = np.flatnonzero(derived)
        healthy[indices] = (
            np.linalg.norm(faulty_points[indices, :3] - clean_points[source[indices], :3], axis=1)
            <= point_tolerance_m
        )
    corrupted = ~healthy
    height, width = faulty.valid.shape
    healthy_per_ray = np.zeros((height, width), dtype=bool)
    supported_points = healthy & faulty.point_valid
    if np.any(supported_points):
        healthy_per_ray[faulty.point_row[supported_points], faulty.point_col[supported_points]] = True
    # Provenance distinguishes a real surviving return from a synthetic or
    # moved return that merely happens to land at a similar range.
    range_agrees = np.abs(faulty.range_m - clean.range_m) <= range_tolerance_m
    supported = faulty.valid & clean.valid & healthy_per_ray & range_agrees
    replace = faulty.valid & clean.valid & ~supported
    add = clean.valid & (~faulty.valid | replace)
    delete = faulty.valid & (~clean.valid | replace)
    # A cell with multiple original measurements cannot identify which one to
    # delete from one ray-level prediction. Preserve it unless separately
    # resolved by a future point-level head.
    delete_valid = faulty.valid & (faulty.collision_count == 1)
    return RangeTargets(
        add=add, add_range_m=np.where(add, clean.range_m, 0).astype(np.float32),
        delete=delete, delete_valid=delete_valid, keep=supported,
        replace=replace, clean_valid=clean.valid,
        clean_range_m=clean.range_m, healthy_original=healthy,
        corrupted_original=corrupted,
    )
