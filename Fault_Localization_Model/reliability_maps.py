from __future__ import annotations

import numpy as np

from Fault_Localization_Model.config_utils import require_range


POINT_STATUS_CORRECT = 0
POINT_STATUS_MISSING = 1
POINT_STATUS_MOVED = 2
POINT_STATUS_ADDED = 3
LEGACY_DUPLICATE_MAP_KEYS = {
    "correct_counts",
    "missing_counts",
    "wrong_counts",
}


def point_counts_grid(points, x_min, x_max, y_min, y_max, grid_rows, grid_cols):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"points must have shape [N,>=2], got {points.shape}")
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("points contains non-finite XY coordinates")
    if (
        not isinstance(grid_rows, (int, np.integer))
        or not isinstance(grid_cols, (int, np.integer))
        or grid_rows < 1
        or grid_cols < 1
    ):
        raise ValueError("grid_rows and grid_cols must be positive integers")
    require_range(x_min, x_max, "x range")
    require_range(y_min, y_max, "y range")
    if len(points) == 0:
        return np.zeros((grid_rows, grid_cols), dtype=np.float32)

    x_cell_size = (x_max - x_min) / grid_rows
    y_cell_size = (y_max - y_min) / grid_cols
    cols = np.floor((points[:, 1] - y_min) / y_cell_size).astype(np.int32)
    rows_from_bottom = np.floor((points[:, 0] - x_min) / x_cell_size).astype(np.int32)
    rows = grid_rows - 1 - rows_from_bottom
    valid = (rows >= 0) & (rows < grid_rows) & (cols >= 0) & (cols < grid_cols)
    linear = (
        rows[valid].astype(np.int64) * int(grid_cols)
        + cols[valid].astype(np.int64)
    )
    return np.bincount(
        linear,
        minlength=int(grid_rows) * int(grid_cols),
    ).reshape(grid_rows, grid_cols).astype(np.float32)


def make_reliability_maps(
    clean_points,
    clean_point_ids,
    faulty_points,
    faulty_point_ids,
    faulty_source_ids,
    movement_tolerance_m,
    x_min,
    x_max,
    y_min,
    y_max,
    grid_rows,
    grid_cols,
):
    """Build reliability targets from exact point provenance rather than occupancy matching."""

    clean_points = np.asarray(clean_points)
    faulty_points = np.asarray(faulty_points)
    for name, points in (
        ("clean_points", clean_points),
        ("faulty_points", faulty_points),
    ):
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"{name} must have shape [N,>=3], got {points.shape}")
        if not np.isfinite(points[:, :3]).all():
            raise ValueError(f"{name} contains non-finite XYZ coordinates")
    if not np.isfinite(movement_tolerance_m) or movement_tolerance_m < 0.0:
        raise ValueError("movement_tolerance_m must be finite and non-negative")
    clean_point_ids = np.asarray(clean_point_ids, dtype=np.int64)
    faulty_point_ids = np.asarray(faulty_point_ids, dtype=np.int64)
    faulty_source_ids = np.asarray(faulty_source_ids, dtype=np.int64)
    if clean_point_ids.shape != (len(clean_points),):
        raise ValueError("clean_point_ids must contain one ID per clean point.")
    if faulty_point_ids.shape != (len(faulty_points),):
        raise ValueError("faulty_point_ids must contain one ID per faulty point.")
    if faulty_source_ids.shape != (len(faulty_points),):
        raise ValueError("faulty_source_ids must contain one source ID per faulty point.")
    if len(np.unique(clean_point_ids)) != len(clean_point_ids):
        raise ValueError("clean_point_ids must be unique within a frame.")
    if len(np.unique(faulty_point_ids)) != len(faulty_point_ids):
        raise ValueError("faulty_point_ids must be unique within a frame.")

    clean_counts = point_counts_grid(clean_points[:, :4], x_min, x_max, y_min, y_max, grid_rows, grid_cols)
    faulty_counts = point_counts_grid(faulty_points[:, :4], x_min, x_max, y_min, y_max, grid_rows, grid_cols)

    clean_index_by_id = {int(point_id): index for index, point_id in enumerate(clean_point_ids)}
    original_mask = faulty_source_ids >= 0
    original_source_ids = faulty_source_ids[original_mask]
    if len(np.unique(original_source_ids)) != len(original_source_ids):
        raise ValueError("An injector produced duplicate rows for the same clean source ID.")
    unknown_ids = sorted(set(map(int, original_source_ids)) - set(clean_index_by_id))
    if unknown_ids:
        raise ValueError(f"Faulty points reference unknown clean source IDs: {unknown_ids[:5]}")

    original_faulty_indices = np.flatnonzero(original_mask)
    original_clean_indices = np.array(
        [clean_index_by_id[int(source_id)] for source_id in original_source_ids],
        dtype=np.int64,
    )
    displacement = np.zeros(len(original_faulty_indices), dtype=np.float32)
    if len(original_faulty_indices):
        coordinate_delta = (
            faulty_points[original_faulty_indices, :3]
            - clean_points[original_clean_indices, :3]
        )
        displacement = np.linalg.norm(coordinate_delta, axis=1)
    moved_original = displacement > movement_tolerance_m

    present_clean = np.zeros(len(clean_points), dtype=bool)
    present_clean[original_clean_indices] = True
    missing_clean = ~present_clean

    correct_clean_indices = original_clean_indices[~moved_original]
    moved_clean_indices = original_clean_indices[moved_original]
    moved_faulty_indices = original_faulty_indices[moved_original]
    added_faulty_indices = np.flatnonzero(~original_mask)

    clean_point_status = np.full(len(clean_points), POINT_STATUS_MISSING, dtype=np.int8)
    clean_point_status[correct_clean_indices] = POINT_STATUS_CORRECT
    clean_point_status[moved_clean_indices] = POINT_STATUS_MOVED
    faulty_point_status = np.full(len(faulty_points), POINT_STATUS_ADDED, dtype=np.int8)
    faulty_point_status[original_faulty_indices[~moved_original]] = POINT_STATUS_CORRECT
    faulty_point_status[moved_faulty_indices] = POINT_STATUS_MOVED

    correct_faulty_indices = original_faulty_indices[~moved_original]
    correct_points = faulty_points[correct_faulty_indices]
    missing_points = clean_points[missing_clean]
    moved_points = faulty_points[moved_faulty_indices]
    added_points = faulty_points[added_faulty_indices]

    clean_point_counts = point_counts_grid(
        correct_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    missing_faulty = point_counts_grid(
        missing_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    moved_faulty = point_counts_grid(
        moved_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    added_faulty = point_counts_grid(
        added_points, x_min, x_max, y_min, y_max, grid_rows, grid_cols
    )
    faulty_point_counts = missing_faulty + moved_faulty + added_faulty

    denominator = clean_point_counts + faulty_point_counts
    reliability = np.ones_like(clean_counts, dtype=np.float32)
    occupied = denominator > 0
    reliability[occupied] = clean_point_counts[occupied] / denominator[occupied]
    fault_heatmap = 1.0 - np.clip(reliability, 0.0, 1.0)

    return {
        "clean_counts": clean_counts,
        "faulty_counts": faulty_counts,
        "clean_point_counts": clean_point_counts,
        "faulty_point_counts": faulty_point_counts,
        "missing_faulty_counts": missing_faulty,
        "moved_faulty_counts": moved_faulty,
        "added_faulty_counts": added_faulty,
        "correct_counts": clean_point_counts,
        "missing_counts": missing_faulty,
        "wrong_counts": added_faulty,
        "fault_heatmap": fault_heatmap.astype(np.float32),
        "reliability_map": reliability.astype(np.float32),
        "correct_point_ids": clean_point_ids[correct_clean_indices],
        "missing_point_ids": clean_point_ids[missing_clean],
        "moved_point_ids": faulty_point_ids[moved_faulty_indices],
        "moved_source_ids": faulty_source_ids[moved_faulty_indices],
        "moved_displacement_m": displacement[moved_original],
        "added_point_ids": faulty_point_ids[added_faulty_indices],
        "clean_point_status": clean_point_status,
        "faulty_point_status": faulty_point_status,
    }


def canonical_maps_for_storage(maps):
    """Drop only aliases whose canonical arrays are saved under newer names."""

    return {
        key: value
        for key, value in maps.items()
        if key not in LEGACY_DUPLICATE_MAP_KEYS
    }
