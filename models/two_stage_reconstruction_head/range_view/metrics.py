"""XYZ point-set and original-provenance metrics for range-view repairs."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .data import RangeSample
from .merge import MergeResult


def _match_fraction(source: np.ndarray, target: np.ndarray, tolerance_m: float) -> float:
    if not len(source):
        return 1.0 if not len(target) else 0.0
    if not len(target):
        return 0.0
    distance, _ = cKDTree(target[:, :3]).query(source[:, :3], k=1, workers=1)
    return float(np.mean(distance <= tolerance_m))


def _point_set_scores(predicted: np.ndarray, clean: np.ndarray, tolerance_m: float, prefix: str) -> dict[str, float]:
    precision = _match_fraction(predicted, clean, tolerance_m)
    recall = _match_fraction(clean, predicted, tolerance_m)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    if len(predicted) and len(clean):
        forward = cKDTree(clean[:, :3]).query(predicted[:, :3], k=1, workers=1)[0].mean()
        reverse = cKDTree(predicted[:, :3]).query(clean[:, :3], k=1, workers=1)[0].mean()
        chamfer = float((forward + reverse) / 2)
    else:
        chamfer = float("nan")
    return {
        f"{prefix}_precision_at_{tolerance_m:g}m": precision,
        f"{prefix}_recall_at_{tolerance_m:g}m": recall,
        f"{prefix}_f1_at_{tolerance_m:g}m": f1,
        f"{prefix}_iou_at_{tolerance_m:g}m": f1 / max(2 - f1, 1e-12),
        f"{prefix}_chamfer_m": chamfer,
    }


def evaluate_xyz(sample: RangeSample, merged: MergeResult, *, tolerance_m: float = 0.2) -> dict[str, float]:
    healthy = sample.targets.healthy_original
    corrupted = sample.targets.corrupted_original
    deleted = np.zeros(len(sample.faulty_points), dtype=bool)
    deleted[merged.deleted_original_indices] = True
    healthy_total = int(healthy.sum())
    corrupt_total = int(corrupted.sum())
    source = sample.faulty_source_ids
    stable_sources = source[(source >= 0) & healthy]
    missing_clean_mask = np.ones(len(sample.clean_points), dtype=bool)
    missing_clean_mask[stable_sources] = False
    missing_clean = sample.clean_points[missing_clean_mask]
    generated = merged.generated_points
    addition_precision = _match_fraction(generated, sample.clean_points, tolerance_m)
    addition_recall = _match_fraction(missing_clean, generated, tolerance_m)
    metrics = {
        # No-original samples have no meaningful preservation/deletion rate.
        # Mark them undefined so macro averages do not treat them as failures.
        "healthy_original_preservation_rate": (
            float((healthy & ~deleted).sum() / healthy_total)
            if healthy_total else float("nan")
        ),
        "false_original_delete_rate": (
            float((healthy & deleted).sum() / healthy_total)
            if healthy_total else float("nan")
        ),
        "corrupted_point_rejection_rate": (
            float((corrupted & deleted).sum() / corrupt_total)
            if corrupt_total else float("nan")
        ),
        "addition_precision": addition_precision,
        "addition_recall": addition_recall,
        "generated_hallucination_rate": 1 - addition_precision if len(generated) else 0.0,
        "original_count": float(len(sample.faulty_points)),
        "generated_count": float(len(generated)),
        "deleted_original_count": float(len(merged.deleted_original_indices)),
        "same_ray_original_and_generated": float(merged.same_ray_original_and_generated),
    }
    metrics.update(_point_set_scores(sample.faulty_points, sample.clean_points, tolerance_m, "faulty"))
    metrics.update(_point_set_scores(merged.points, sample.clean_points, tolerance_m, "reconstructed"))
    metrics["net_f1_improvement"] = (
        metrics[f"reconstructed_f1_at_{tolerance_m:g}m"] - metrics[f"faulty_f1_at_{tolerance_m:g}m"]
    )
    return metrics
