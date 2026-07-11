from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


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
    if from_logits:
        return torch.sigmoid(output)
    return output.clamp(0.0, 1.0)


def prepare_probability_target(
    output: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    metric_grid_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate shapes and return detached probability and target tensors."""
    prob = probabilities_from_output(output, from_logits=from_logits)
    target = _as_bchw(target, "target").float()
    if prob.shape[-2:] != target.shape[-2:]:
        prob = F.interpolate(prob, size=target.shape[-2:], mode="bilinear", align_corners=False)
    if metric_grid_size is not None:
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
    if boundary_only:
        pred = _boundary(pred)
        target = _boundary(target)

    pred_empty = not pred.any()
    target_empty = not target.any()
    if pred_empty and target_empty:
        return 0.0, False
    if pred_empty != target_empty:
        return None, True

    pred_points = np.argwhere(pred).astype(np.float64)
    target_points = np.argwhere(target).astype(np.float64)
    pred_points[:, 0] *= x_cell_size_m
    pred_points[:, 1] *= y_cell_size_m
    target_points[:, 0] *= x_cell_size_m
    target_points[:, 1] *= y_cell_size_m

    def mean_min_distance(a: np.ndarray, b: np.ndarray, chunk_size: int = 4096) -> float:
        distances = []
        for start in range(0, len(a), chunk_size):
            chunk = a[start : start + chunk_size]
            diff = chunk[:, None, :] - b[None, :, :]
            distances.append(np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1))
        return float(np.concatenate(distances).mean())

    return 0.5 * (mean_min_distance(pred_points, target_points) + mean_min_distance(target_points, pred_points)), False


def infer_cell_sizes(metadata: Dict, default_x: float, default_y: float) -> Tuple[float, float]:
    """Infer metric cell sizes from metadata, falling back to configured defaults."""
    x_cell = metadata.get("x_cell_size_m", default_x)
    y_cell = metadata.get("y_cell_size_m", default_y)
    return float(x_cell), float(y_cell)


@dataclass
class HeatmapMetricAccumulator:
    """Accumulate validation metrics over a full epoch."""

    threshold: float = 0.5
    metric_grid_size: Optional[int] = 100
    x_cell_size_m: float = 0.64
    y_cell_size_m: float = 0.64
    boundary_chamfer: bool = False
    confusion: ConfusionCounts = field(default_factory=ConfusionCounts)
    faulty_confusion: ConfusionCounts = field(default_factory=ConfusionCounts)
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

    def _new_group_accumulator(self) -> "HeatmapMetricAccumulator":
        return HeatmapMetricAccumulator(
            threshold=self.threshold,
            metric_grid_size=None,
            x_cell_size_m=self.x_cell_size_m,
            y_cell_size_m=self.y_cell_size_m,
            boundary_chamfer=self.boundary_chamfer,
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
        metadata_list = list(metadata or [{} for _ in range(prob.shape[0])])

        for index in range(prob.shape[0]):
            pred_values = prob[index]
            target_values = target_np[index]
            pred_mask = pred_values >= self.threshold
            target_mask = target_values >= self.threshold
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

            meta = metadata_list[index] if index < len(metadata_list) else {}
            x_cell, y_cell = infer_cell_sizes(meta, self.x_cell_size_m, self.y_cell_size_m)
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
        output = {
            f"{prefix}iou": metrics["iou"],
            f"{prefix}f1": metrics["f1"],
            f"{prefix}precision": metrics["precision"],
            f"{prefix}recall": metrics["recall"],
            f"{prefix}specificity": metrics["specificity"],
            f"{prefix}balanced_accuracy": metrics["balanced_accuracy"],
            f"{prefix}brier_score": self.brier_sum / max(self.cell_count, 1),
            f"{prefix}pixel_mae": self.mae_sum / max(self.cell_count, 1),
            f"{prefix}chamfer_distance_m": self.chamfer_sum / max(self.chamfer_count, 1),
            f"{prefix}empty_mask_mismatch_rate": self.empty_mismatch_count / max(self.sample_count, 1),
            f"{prefix}sample_count": float(self.sample_count),
            f"{prefix}faulty_sample_count": float(self.faulty_sample_count),
            f"{prefix}chamfer_valid_count": float(self.chamfer_count),
        }
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


def threshold_sweep(
    outputs: List[torch.Tensor],
    targets: List[torch.Tensor],
    thresholds: Iterable[float],
    metric_grid_size: int,
) -> List[Dict[str, float]]:
    """Compute thresholded validation metrics for a saved validation epoch."""
    rows = []
    output_tensor = torch.cat(outputs, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    for threshold in thresholds:
        acc = HeatmapMetricAccumulator(threshold=threshold, metric_grid_size=metric_grid_size)
        acc.update(output_tensor, target_tensor, from_logits=False, update_groups=False)
        row = {"threshold": float(threshold)}
        row.update(acc.compute())
        rows.append(row)
    return rows


def save_threshold_sweep(rows: List[Dict[str, float]], output_dir: Path) -> None:
    """Save threshold sweep CSV and JSON summary."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best_f1 = max(rows, key=lambda row: row["f1"])
    best_iou = max(rows, key=lambda row: row["iou"])
    with (output_dir / "threshold_sweep_summary.json").open("w", encoding="utf-8") as file:
        json.dump({"best_by_f1": best_f1, "best_by_iou": best_iou}, file, indent=2)


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
    with (output_dir / "group_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
