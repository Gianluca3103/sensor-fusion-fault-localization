"""Detector-independent reconstruction metrics inside annotated VoD objects."""

from __future__ import annotations

import math

import numpy as np

from Fault_Localization_Model.bev_utils import HEIGHT_RANGE_M
from models.two_stage_reconstruction_head.object_detection.annotations import (
    RotatedBEVBox,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


def rasterize_rotated_box(
    box: RotatedBEVBox,
    geometry: BEVGridGeometry,
) -> np.ndarray:
    """Rasterize a metric rotated box at BEV cell centres."""

    geometry.validate()
    rows = np.arange(geometry.height, dtype=np.float32)
    columns = np.arange(geometry.width, dtype=np.float32)
    x = geometry.x_min + (
        geometry.height - rows - 0.5
    ) * geometry.pillar_size_x
    y = geometry.y_min + (columns + 0.5) * geometry.pillar_size_y
    delta_x = x[:, None] - float(box.x)
    delta_y = y[None, :] - float(box.y)
    cosine = math.cos(float(box.yaw))
    sine = math.sin(float(box.yaw))
    longitudinal = cosine * delta_x + sine * delta_y
    lateral = -sine * delta_x + cosine * delta_y
    return (
        (np.abs(longitudinal) <= float(box.length) / 2.0)
        & (np.abs(lateral) <= float(box.width) / 2.0)
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _disk_offsets(tolerance_m: float, resolution_m: float) -> tuple[tuple[int, int], ...]:
    radius = int(math.floor(tolerance_m / resolution_m + 1.0e-6))
    return tuple(
        (row, column)
        for row in range(-radius, radius + 1)
        for column in range(-radius, radius + 1)
        if math.hypot(row * resolution_m, column * resolution_m)
        <= tolerance_m + 1.0e-6
    )


def _dilate(values: np.ndarray, tolerance_m: float, resolution_m: float) -> np.ndarray:
    output = np.zeros_like(values, dtype=bool)
    if not values.size:
        return output
    height, width = values.shape
    for row_offset, column_offset in _disk_offsets(tolerance_m, resolution_m):
        source_top = max(0, -row_offset)
        source_bottom = min(height, height - row_offset)
        source_left = max(0, -column_offset)
        source_right = min(width, width - column_offset)
        target_top = source_top + row_offset
        target_bottom = source_bottom + row_offset
        target_left = source_left + column_offset
        target_right = source_right + column_offset
        output[target_top:target_bottom, target_left:target_right] |= values[
            source_top:source_bottom, source_left:source_right
        ]
    return output


def _crop(values: tuple[np.ndarray, ...], mask: np.ndarray, padding: int) -> tuple[np.ndarray, ...]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return tuple(value[:0, :0] for value in values)
    top = max(0, int(rows.min()) - padding)
    bottom = min(mask.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(mask.shape[1], int(columns.max()) + padding + 1)
    return tuple(value[top:bottom, left:right] for value in values)


def _nearest_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Nearest XY distances in cell coordinates, with a dependency-light fallback."""

    if not len(first) or not len(second):
        return np.empty((0,), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(second).query(first, k=1)
        return np.asarray(distances, dtype=np.float32)
    except ImportError:
        result = np.empty((len(first),), dtype=np.float32)
        for start in range(0, len(first), 1024):
            chunk = first[start : start + 1024]
            squared = np.sum((chunk[:, None] - second[None, :]) ** 2, axis=2)
            result[start : start + len(chunk)] = np.sqrt(squared.min(axis=1))
        return result


def evaluate_bev_condition(
    clean_bev: np.ndarray,
    condition_bev: np.ndarray,
    scope_mask: np.ndarray,
    *,
    resolution_m: float,
    tolerances_m: tuple[float, ...] = (0.2, 0.5),
    occupancy_threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Compare one BEV condition with clean geometry inside an oracle scope."""

    clean = np.asarray(clean_bev, dtype=np.float32)
    condition = np.asarray(condition_bev, dtype=np.float32)
    scope = np.asarray(scope_mask, dtype=bool)
    if clean.shape != condition.shape or clean.ndim != 3 or clean.shape[0] != 3:
        raise ValueError("clean_bev and condition_bev must both have shape [3,H,W]")
    if scope.shape != clean.shape[1:]:
        raise ValueError("scope_mask must align with the BEV")
    if resolution_m <= 0.0:
        raise ValueError("resolution_m must be positive")

    maximum_tolerance = max(tolerances_m, default=0.0)
    padding = int(math.ceil(maximum_tolerance / resolution_m))
    clean_occupancy = (clean[0] >= occupancy_threshold) & scope
    condition_occupancy = (condition[0] >= occupancy_threshold) & scope
    clean_occupancy, condition_occupancy, scope = _crop(
        (clean_occupancy, condition_occupancy, scope), scope, padding
    )

    true_positive = int((clean_occupancy & condition_occupancy).sum())
    false_positive = int((~clean_occupancy & condition_occupancy & scope).sum())
    false_negative = int((clean_occupancy & ~condition_occupancy).sum())
    true_negative = int((~clean_occupancy & ~condition_occupancy & scope).sum())
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    result: dict[str, float | int | None] = {
        "scope_cells": int(scope.sum()),
        "clean_occupied_cells": int(clean_occupancy.sum()),
        "predicted_occupied_cells": int(condition_occupancy.sum()),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "exact_precision": precision,
        "exact_recall": recall,
        "exact_f1": f1,
        "exact_iou": _safe_ratio(
            true_positive, true_positive + false_positive + false_negative
        ),
        "hallucination_fraction": _safe_ratio(
            false_positive, true_positive + false_positive
        ),
        "occupied_count_ratio": _safe_ratio(
            int(condition_occupancy.sum()), int(clean_occupancy.sum())
        ),
    }
    for tolerance_m in tolerances_m:
        target_neighborhood = _dilate(clean_occupancy, tolerance_m, resolution_m)
        prediction_neighborhood = _dilate(
            condition_occupancy, tolerance_m, resolution_m
        )
        matched_predictions = int(
            (condition_occupancy & target_neighborhood).sum()
        )
        matched_targets = int((clean_occupancy & prediction_neighborhood).sum())
        tolerant_precision = _safe_ratio(
            matched_predictions, int(condition_occupancy.sum())
        )
        tolerant_recall = _safe_ratio(
            matched_targets, int(clean_occupancy.sum())
        )
        tolerant_f1 = _safe_ratio(
            2.0 * tolerant_precision * tolerant_recall,
            tolerant_precision + tolerant_recall,
        )
        name = str(tolerance_m).replace(".", "_") + "m"
        result.update(
            {
                f"tolerant_{name}_matched_predictions": matched_predictions,
                f"tolerant_{name}_matched_targets": matched_targets,
                f"tolerant_{name}_precision": tolerant_precision,
                f"tolerant_{name}_recall": tolerant_recall,
                f"tolerant_{name}_f1": tolerant_f1,
                f"tolerant_{name}_iou": _safe_ratio(
                    tolerant_f1, 2.0 - tolerant_f1
                ),
            }
        )

    clean_points = np.argwhere(clean_occupancy).astype(np.float32)
    condition_points = np.argwhere(condition_occupancy).astype(np.float32)
    if len(clean_points) and len(condition_points):
        clean_to_condition = _nearest_distances(clean_points, condition_points)
        condition_to_clean = _nearest_distances(condition_points, clean_points)
        result["symmetric_chamfer_m"] = float(
            0.5
            * resolution_m
            * (clean_to_condition.mean() + condition_to_clean.mean())
        )
        result["clean_to_condition_p95_m"] = float(
            resolution_m * np.percentile(clean_to_condition, 95)
        )
    else:
        result["symmetric_chamfer_m"] = None
        result["clean_to_condition_p95_m"] = None

    full_scope = np.asarray(scope_mask, dtype=bool)
    clean_support = full_scope & (clean[0] >= occupancy_threshold)
    matched_support = clean_support & (condition[0] >= occupancy_threshold)
    height_scale = float(HEIGHT_RANGE_M[1] - HEIGHT_RANGE_M[0])
    result["clean_support_cells"] = int(clean_support.sum())
    result["matched_support_cells"] = int(matched_support.sum())
    result["density_abs_error_sum"] = float(
        np.abs(condition[1] - clean[1])[clean_support].sum()
    )
    result["height_abs_error_m_sum"] = float(
        height_scale * np.abs(condition[2] - clean[2])[clean_support].sum()
    )
    result["matched_height_abs_error_m_sum"] = float(
        height_scale * np.abs(condition[2] - clean[2])[matched_support].sum()
    )
    result["density_mae_clean_support"] = (
        _safe_ratio(result["density_abs_error_sum"], result["clean_support_cells"])
        if result["clean_support_cells"]
        else None
    )
    result["height_mae_m_clean_support"] = (
        _safe_ratio(result["height_abs_error_m_sum"], result["clean_support_cells"])
        if result["clean_support_cells"]
        else None
    )
    result["height_mae_m_matched_support"] = (
        _safe_ratio(
            result["matched_height_abs_error_m_sum"],
            result["matched_support_cells"],
        )
        if result["matched_support_cells"]
        else None
    )
    return result
