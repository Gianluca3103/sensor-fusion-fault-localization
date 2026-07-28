from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows


EPS = 1e-8


def _as_bchw(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Return a heat-map tensor as [B, 1, H, W]."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W], got {tuple(tensor.shape)}")
    return tensor


def probabilities_from_output(output: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """Convert model output to probabilities without applying sigmoid twice."""
    output = _as_bchw(output, "output")
    if not torch.isfinite(output).all():
        raise ValueError("output contains non-finite values")
    if from_logits:
        return torch.sigmoid(output)
    if torch.any((output < 0.0) | (output > 1.0)):
        raise ValueError("Probability output values must lie in [0,1]")
    return output


def prepare_probability_target(
    output: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    metric_grid_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate shapes and return detached probability and target tensors."""
    prob = probabilities_from_output(output, from_logits=from_logits)
    target = _as_bchw(target, "target").float()
    if not torch.isfinite(target).all():
        raise ValueError("target contains non-finite values")
    if torch.any((target < 0.0) | (target > 1.0)):
        raise ValueError("target values must lie in [0,1]")
    if prob.shape[-2:] != target.shape[-2:]:
        prob = F.interpolate(prob, size=target.shape[-2:], mode="bilinear", align_corners=False)
    if metric_grid_size is not None:
        if int(metric_grid_size) != metric_grid_size or metric_grid_size <= 0:
            raise ValueError("metric_grid_size must be a positive integer or None")
        size = (metric_grid_size, metric_grid_size)
        prob = F.adaptive_avg_pool2d(prob, output_size=size)
        target = F.adaptive_avg_pool2d(target, output_size=size)
    return prob.detach(), target.detach().clamp(0.0, 1.0)


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, pred_mask: np.ndarray, target_mask: np.ndarray) -> None:
        pred = pred_mask.astype(bool)
        target = target_mask.astype(bool)
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target masks must match, got {pred.shape} and {target.shape}"
            )
        self.tp += int(np.logical_and(pred, target).sum())
        self.fp += int(np.logical_and(pred, ~target).sum())
        self.tn += int(np.logical_and(~pred, ~target).sum())
        self.fn += int(np.logical_and(~pred, target).sum())

    def metrics(self) -> Dict[str, float]:
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        iou = tp / max(tp + fp + fn, 1)
        f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
        return {
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "balanced_accuracy": float(0.5 * (recall + specificity)),
            "tp": float(tp),
            "fp": float(fp),
            "tn": float(tn),
            "fn": float(fn),
        }


@dataclass
class LocalizationToleranceCounts:
    """Accumulate one-to-one fault localization matches within a metric radius."""

    tolerance_m: float = 0.20
    pred_total: int = 0
    target_total: int = 0
    pred_matched: int = 0
    target_matched: int = 0

    def update(self, pred_mask: np.ndarray, target_mask: np.ndarray, x_cell_size_m: float, y_cell_size_m: float) -> None:
        pred = pred_mask.astype(bool)
        target = target_mask.astype(bool)
        if pred.shape != target.shape:
            raise ValueError(
                f"Prediction and target masks must match, got {pred.shape} and {target.shape}"
            )
        pred_count = int(pred.sum())
        target_count = int(target.sum())
        self.pred_total += pred_count
        self.target_total += target_count

        if pred_count == 0 or target_count == 0:
            return

        pred_matched, target_matched = one_to_one_match_masks(
            pred,
            target,
            x_cell_size_m=x_cell_size_m,
            y_cell_size_m=y_cell_size_m,
            tolerance_m=self.tolerance_m,
        )
        matched_count = int(pred_matched.sum())
        if matched_count != int(target_matched.sum()):
            raise RuntimeError("One-to-one localization matching produced unequal match counts")
        self.pred_matched += matched_count
        self.target_matched += matched_count

    def metrics(self) -> Dict[str, float]:
        true_positive = self.pred_matched
        false_positive = self.pred_total - true_positive
        false_negative = self.target_total - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = (2 * true_positive) / max(
            2 * true_positive + false_positive + false_negative,
            1,
        )
        iou = true_positive / max(
            true_positive + false_positive + false_negative,
            1,
        )
        return {
            "localization_iou": float(iou),
            "localization_precision": float(precision),
            "localization_recall": float(recall),
            "localization_f1": float(f1),
            "localization_pred_total": float(self.pred_total),
            "localization_target_total": float(self.target_total),
            "localization_pred_matched": float(self.pred_matched),
            "localization_target_matched": float(self.target_matched),
            "localization_tolerance_m": float(self.tolerance_m),
        }


def one_to_one_match_masks(
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
    x_cell_size_m: float,
    y_cell_size_m: float,
    tolerance_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return maximum-cardinality one-to-one matches within a metric radius."""
    prediction = np.asarray(prediction_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target masks must match, got "
            f"{prediction.shape} and {target.shape}"
        )
    if (
        not np.isfinite([x_cell_size_m, y_cell_size_m, tolerance_m]).all()
        or x_cell_size_m <= 0.0
        or y_cell_size_m <= 0.0
        or tolerance_m < 0.0
    ):
        raise ValueError("Cell sizes must be positive and tolerance must be non-negative")

    prediction_matched = np.zeros_like(prediction, dtype=bool)
    target_matched = np.zeros_like(target, dtype=bool)
    prediction_indices = np.argwhere(prediction)
    target_indices = np.argwhere(target)
    if len(prediction_indices) == 0 or len(target_indices) == 0:
        return prediction_matched, target_matched

    scale = np.asarray([x_cell_size_m, y_cell_size_m], dtype=np.float64)
    prediction_coordinates = prediction_indices.astype(np.float64) * scale
    target_coordinates = target_indices.astype(np.float64) * scale
    target_tree = cKDTree(target_coordinates)
    neighborhoods = target_tree.query_ball_point(
        prediction_coordinates,
        r=float(tolerance_m) + 1e-12,
    )
    edge_counts = np.fromiter(
        (len(neighbors) for neighbors in neighborhoods),
        dtype=np.int64,
        count=len(neighborhoods),
    )
    if not np.any(edge_counts):
        return prediction_matched, target_matched

    row_indices = np.repeat(np.arange(len(neighborhoods)), edge_counts)
    column_indices = np.concatenate(
        [
            np.asarray(neighbors, dtype=np.int64)
            for neighbors in neighborhoods
            if neighbors
        ]
    )
    graph = csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.uint8),
            (row_indices, column_indices),
        ),
        shape=(len(prediction_indices), len(target_indices)),
    )
    matched_target_by_prediction = maximum_bipartite_matching(
        graph,
        perm_type="column",
    )
    matched_prediction_rows = np.flatnonzero(matched_target_by_prediction >= 0)
    matched_target_rows = matched_target_by_prediction[matched_prediction_rows]
    prediction_matched[tuple(prediction_indices[matched_prediction_rows].T)] = True
    target_matched[tuple(target_indices[matched_target_rows].T)] = True
    return prediction_matched, target_matched


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a simple 4-connected binary boundary."""
    if not mask.any():
        return mask.astype(bool)
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & ~eroded


def chamfer_distance_m(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    x_cell_size_m: float,
    y_cell_size_m: float,
    boundary_only: bool = False,
) -> Tuple[Optional[float], bool]:
    """Return symmetric Chamfer distance in meters and whether one mask was empty.

    If both masks are empty, distance is 0. If exactly one mask is empty, the
    distance is returned as None and the mismatch flag is True.
    """
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target masks must match, got {pred.shape} and {target.shape}"
        )
    if (
        not np.isfinite([x_cell_size_m, y_cell_size_m]).all()
        or x_cell_size_m <= 0.0
        or y_cell_size_m <= 0.0
    ):
        raise ValueError("Chamfer cell sizes must be finite and positive")
    if boundary_only:
        pred = _boundary(pred)
        target = _boundary(target)

    pred_empty = not pred.any()
    target_empty = not target.any()
    if pred_empty and target_empty:
        return 0.0, False
    if pred_empty != target_empty:
        return None, True

    distance_to_target = distance_transform_edt(
        ~target, sampling=(x_cell_size_m, y_cell_size_m)
    )
    distance_to_prediction = distance_transform_edt(
        ~pred, sampling=(x_cell_size_m, y_cell_size_m)
    )
    return (
        0.5
        * (
            float(distance_to_target[pred].mean())
            + float(distance_to_prediction[target].mean())
        ),
        False,
    )


def infer_cell_sizes(metadata: Dict, default_x: float, default_y: float) -> Tuple[float, float]:
    """Infer metric cell sizes from metadata, falling back to configured defaults."""
    x_cell = metadata.get("x_cell_size_m", default_x)
    y_cell = metadata.get("y_cell_size_m", default_y)
    x_cell, y_cell = float(x_cell), float(y_cell)
    if (
        not np.isfinite([x_cell, y_cell]).all()
        or x_cell <= 0.0
        or y_cell <= 0.0
    ):
        raise ValueError(f"Metric cell sizes must be positive, got {x_cell}, {y_cell}")
    return x_cell, y_cell


@dataclass
class HeatmapMetricAccumulator:
    """Accumulate validation metrics over a full epoch."""

    threshold: float = 0.5
    target_threshold: Optional[float] = None
    metric_grid_size: Optional[int] = 100
    x_cell_size_m: float = 0.64
    y_cell_size_m: float = 0.64
    boundary_chamfer: bool = False
    compute_chamfer: bool = True
    localization_tolerance_m: float = 0.20
    confusion: ConfusionCounts = field(default_factory=ConfusionCounts)
    faulty_confusion: ConfusionCounts = field(default_factory=ConfusionCounts)
    localization: LocalizationToleranceCounts = field(default_factory=LocalizationToleranceCounts)
    brier_sum: float = 0.0
    mae_sum: float = 0.0
    cell_count: int = 0
    sample_count: int = 0
    faulty_sample_count: int = 0
    chamfer_sum: float = 0.0
    chamfer_count: int = 0
    empty_mismatch_count: int = 0
    error_sum: Optional[np.ndarray] = None
    groups: Dict[str, "HeatmapMetricAccumulator"] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold must be strictly between 0 and 1, got {self.threshold}")
        if self.target_threshold is not None and not 0.0 <= self.target_threshold < 1.0:
            raise ValueError(
                "target_threshold must lie in [0,1), "
                f"got {self.target_threshold}"
            )
        if self.metric_grid_size is not None and self.metric_grid_size <= 0:
            raise ValueError("metric_grid_size must be positive or None")
        if (
            not np.isfinite([self.x_cell_size_m, self.y_cell_size_m]).all()
            or self.x_cell_size_m <= 0.0
            or self.y_cell_size_m <= 0.0
        ):
            raise ValueError("Metric cell sizes must be finite and positive")
        if (
            not np.isfinite(self.localization_tolerance_m)
            or self.localization_tolerance_m < 0.0
        ):
            raise ValueError("localization_tolerance_m must be finite and non-negative")
        self.localization.tolerance_m = self.localization_tolerance_m

    def _new_group_accumulator(self) -> "HeatmapMetricAccumulator":
        return HeatmapMetricAccumulator(
            threshold=self.threshold,
            target_threshold=self.target_threshold,
            metric_grid_size=None,
            x_cell_size_m=self.x_cell_size_m,
            y_cell_size_m=self.y_cell_size_m,
            boundary_chamfer=self.boundary_chamfer,
            compute_chamfer=self.compute_chamfer,
            localization_tolerance_m=self.localization_tolerance_m,
        )

    def update(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        metadata: Optional[Iterable[Dict]] = None,
        from_logits: bool = True,
        update_groups: bool = True,
    ) -> None:
        with torch.no_grad():
            prob_t, target_t = prepare_probability_target(
                output,
                target,
                from_logits=from_logits,
                metric_grid_size=self.metric_grid_size,
            )
        prob = prob_t.squeeze(1).cpu().numpy()
        target_np = target_t.squeeze(1).cpu().numpy()
        metadata_list = (
            [{} for _ in range(prob.shape[0])]
            if metadata is None
            else list(metadata)
        )
        if len(metadata_list) != prob.shape[0]:
            raise ValueError(
                f"Expected {prob.shape[0]} metadata records, got {len(metadata_list)}"
            )

        for index in range(prob.shape[0]):
            pred_values = prob[index]
            target_values = target_np[index]
            pred_mask = pred_values >= self.threshold
            if self.target_threshold is None:
                target_mask = target_values >= self.threshold
            else:
                target_mask = target_values > self.target_threshold
            self.confusion.update(pred_mask, target_mask)

            has_fault = bool(target_mask.any())
            if has_fault:
                self.faulty_sample_count += 1
                self.faulty_confusion.update(pred_mask, target_mask)
            self.sample_count += 1

            err = np.abs(pred_values - target_values)
            sq_err = (pred_values - target_values) ** 2
            self.mae_sum += float(err.sum())
            self.brier_sum += float(sq_err.sum())
            self.cell_count += int(err.size)
            self.error_sum = err.astype(np.float64) if self.error_sum is None else self.error_sum + err

            meta = metadata_list[index]
            if not isinstance(meta, dict):
                raise TypeError(
                    f"Metadata record {index} must be a dictionary, got {type(meta).__name__}"
                )
            x_cell, y_cell = infer_cell_sizes(meta, self.x_cell_size_m, self.y_cell_size_m)
            self.localization.tolerance_m = self.localization_tolerance_m
            self.localization.update(pred_mask, target_mask, x_cell, y_cell)
            if self.compute_chamfer:
                chamfer, mismatch = chamfer_distance_m(
                    pred_mask,
                    target_mask,
                    x_cell,
                    y_cell,
                    boundary_only=self.boundary_chamfer,
                )
                if mismatch:
                    self.empty_mismatch_count += 1
                elif chamfer is not None:
                    self.chamfer_sum += chamfer
                    self.chamfer_count += 1

            if update_groups:
                for group_key in group_keys_from_metadata(meta):
                    if group_key not in self.groups:
                        self.groups[group_key] = self._new_group_accumulator()
                    single_output = torch.from_numpy(pred_values[None, None]).float()
                    single_target = torch.from_numpy(target_values[None, None]).float()
                    self.groups[group_key].update(
                        single_output,
                        single_target,
                        metadata=[meta],
                        from_logits=False,
                        update_groups=False,
                    )

    def compute(self, prefix: str = "") -> Dict[str, float]:
        metrics = self.confusion.metrics()
        faulty_metrics = self.faulty_confusion.metrics() if self.faulty_sample_count else {}
        localization_metrics = self.localization.metrics()
        output = {
            f"{prefix}iou": metrics["iou"],
            f"{prefix}f1": metrics["f1"],
            f"{prefix}precision": metrics["precision"],
            f"{prefix}recall": metrics["recall"],
            f"{prefix}specificity": metrics["specificity"],
            f"{prefix}balanced_accuracy": metrics["balanced_accuracy"],
            f"{prefix}brier_score": self.brier_sum / max(self.cell_count, 1),
            f"{prefix}pixel_mae": self.mae_sum / max(self.cell_count, 1),
            f"{prefix}sample_count": float(self.sample_count),
            f"{prefix}faulty_sample_count": float(self.faulty_sample_count),
        }
        if self.compute_chamfer:
            output.update(
                {
                    f"{prefix}chamfer_distance_m": self.chamfer_sum
                    / max(self.chamfer_count, 1),
                    f"{prefix}empty_mask_mismatch_rate": self.empty_mismatch_count
                    / max(self.sample_count, 1),
                    f"{prefix}chamfer_valid_count": float(self.chamfer_count),
                }
            )
        output.update({f"{prefix}{key}": value for key, value in localization_metrics.items()})
        if faulty_metrics:
            output.update(
                {
                    f"{prefix}faulty_only_iou": faulty_metrics["iou"],
                    f"{prefix}faulty_only_f1": faulty_metrics["f1"],
                    f"{prefix}faulty_only_precision": faulty_metrics["precision"],
                    f"{prefix}faulty_only_recall": faulty_metrics["recall"],
                    f"{prefix}faulty_only_balanced_accuracy": faulty_metrics["balanced_accuracy"],
                }
            )
        return output

    def mean_error_map(self) -> Optional[np.ndarray]:
        if self.error_sum is None or self.sample_count == 0:
            return None
        return self.error_sum / float(self.sample_count)


def group_keys_from_metadata(metadata: Dict) -> List[str]:
    """Create safe group keys for available metadata fields."""
    keys = []
    fault = metadata.get("fault")
    severity = metadata.get("severity")
    if fault is not None:
        keys.append(f"fault={fault}")
    if severity is not None:
        keys.append(f"severity={severity}")
    if fault is not None and severity is not None:
        keys.append(f"fault={fault}|severity={severity}")
    clean_label = "clean" if str(fault).lower() in {"clean", "none", "no_fault"} else "corrupted"
    keys.append(f"condition={clean_label}")
    return keys


def save_threshold_sweep(rows: List[Dict[str, float]], output_dir: Path) -> None:
    """Save threshold sweep CSV and JSON summary."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "threshold_sweep.csv",
        rows,
        fieldnames=list(rows[0]),
    )
    best_f1 = max(rows, key=lambda row: row["f1"])
    best_iou = max(rows, key=lambda row: row["iou"])
    atomic_write_json(
        output_dir / "threshold_sweep_summary.json",
        {"best_by_f1": best_f1, "best_by_iou": best_iou},
    )


def save_group_metrics(groups: Dict[str, HeatmapMetricAccumulator], output_dir: Path) -> None:
    """Save grouped metrics to CSV when metadata groups are available."""
    if not groups:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, accumulator in sorted(groups.items()):
        row = {"group": key}
        row.update(accumulator.compute())
        rows.append(row)
    write_csv_rows(
        output_dir / "group_metrics.csv",
        rows,
        fieldnames=list(rows[0]),
    )


def save_spatial_error_map(
    mean_error: np.ndarray,
    output_dir: Path,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Save raw and visual dataset-level mean absolute error map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "mean_abs_error_map.npy", mean_error.astype(np.float32))

    plt.figure(figsize=(7, 6))
    extent = None
    if x_range is not None and y_range is not None:
        extent = [y_range[0], y_range[1], x_range[1], x_range[0]]
    image = plt.imshow(mean_error, cmap="magma", vmin=0.0, vmax=max(float(mean_error.max()), EPS), extent=extent)
    plt.colorbar(image, label="mean absolute error")
    if extent is not None:
        plt.xlabel("y lateral position (m)")
        plt.ylabel("x forward position (m)")
    else:
        plt.xlabel("BEV column")
        plt.ylabel("BEV row")
    plt.title("Validation Mean Absolute Error Map")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_abs_error_map.png", dpi=180)
    plt.close()
