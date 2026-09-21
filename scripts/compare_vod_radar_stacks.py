"""Compare 3-, 5-, and 20-frame VoD radar stacks against clean LiDAR BEV.

The comparison uses the same radar-to-LiDAR calibration and BEV geometry as
the reconstruction pipeline.  Metrics are accumulated globally over occupied
cells, rather than averaging per-frame percentages.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np

from Fault_Localization_Model.bev_utils import metric_to_grid
from Fault_Localization_Model.vod_dataset import (
    align_radar_to_lidar,
    discover_vod_frames,
    load_vod_lidar,
    load_vod_radar,
    load_vod_radar_to_lidar,
    load_vod_split_ids,
)


DEFAULT_STACK_SIZES = (3, 5, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "train_val", "full"),
        default="val",
    )
    parser.add_argument(
        "--stack-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_STACK_SIZES,
    )
    parser.add_argument(
        "--variant-suffix",
        default="",
        help="Optional suffix such as temporal_filtered.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Randomly evaluate this many split frames; omit for the full split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=50,
        help="Number of evaluated frames rendered as comparison PNGs.",
    )
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=64.0)
    parser.add_argument("--y-min", type=float, default=-32.0)
    parser.add_argument("--y-max", type=float, default=32.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    return parser.parse_args()


def _occupancy(
    xyz: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> np.ndarray:
    _xyz, rows, columns, _valid, height, width = metric_to_grid(
        xyz[:, :3], x_range, y_range, resolution
    )
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[rows, columns] = True
    return occupancy


def _aligned_radar(frame) -> np.ndarray:
    radar = load_vod_radar(frame.radar_path)
    lidar_from_radar = load_vod_radar_to_lidar(
        frame.lidar_calibration_path,
        frame.radar_calibration_path,
    )
    return align_radar_to_lidar(radar, lidar_from_radar)


def _metric_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    tolerance_m: float,
    resolution: float,
) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target occupancy shapes must match")
    if tolerance_m < 0.0 or resolution <= 0.0:
        raise ValueError("Tolerance must be non-negative and resolution positive")

    prediction_count = int(prediction.sum())
    target_count = int(target.sum())
    if tolerance_m == 0.0:
        matched = int((prediction & target).sum())
        matched_predictions = matched_targets = matched
    else:
        target_neighborhood = _metric_dilation(target, tolerance_m, resolution)
        prediction_neighborhood = _metric_dilation(
            prediction, tolerance_m, resolution
        )
        matched_predictions = int((prediction & target_neighborhood).sum())
        matched_targets = int((target & prediction_neighborhood).sum())
    return {
        "matched_predictions": matched_predictions,
        "matched_targets": matched_targets,
        "prediction_count": prediction_count,
        "target_count": target_count,
    }


def _metric_dilation(
    occupancy: np.ndarray,
    tolerance_m: float,
    resolution: float,
) -> np.ndarray:
    """Dilate a BEV mask with a metric disk using only NumPy."""

    radius = int(np.ceil(tolerance_m / resolution))
    output = np.zeros_like(occupancy, dtype=bool)
    height, width = occupancy.shape
    for row_offset in range(-radius, radius + 1):
        for column_offset in range(-radius, radius + 1):
            distance = resolution * np.hypot(row_offset, column_offset)
            if distance > tolerance_m + 1.0e-9:
                continue
            source_row_start = max(0, -row_offset)
            source_row_stop = min(height, height - row_offset)
            source_column_start = max(0, -column_offset)
            source_column_stop = min(width, width - column_offset)
            destination_row_start = source_row_start + row_offset
            destination_row_stop = source_row_stop + row_offset
            destination_column_start = source_column_start + column_offset
            destination_column_stop = source_column_stop + column_offset
            output[
                destination_row_start:destination_row_stop,
                destination_column_start:destination_column_stop,
            ] |= occupancy[
                source_row_start:source_row_stop,
                source_column_start:source_column_stop,
            ]
    return output


def _metrics(counts: dict[str, float]) -> dict[str, float]:
    prediction_count = counts["prediction_count"]
    target_count = counts["target_count"]
    precision = (
        counts["matched_predictions"] / prediction_count
        if prediction_count
        else float(target_count == 0)
    )
    recall = (
        counts["matched_targets"] / target_count
        if target_count
        else float(prediction_count == 0)
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # This is the symmetric tolerant-IoU convention used by the
        # reconstruction evaluators. At zero tolerance it is standard IoU.
        "iou": f1 / (2.0 - f1) if f1 else 0.0,
    }


def _add_counts(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] += value


def _save_visualization(
    radar_by_size: dict[int, np.ndarray],
    destination: Path,
    frame_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = list(radar_by_size)
    figure, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(5 * len(sizes), 6),
        facecolor="black",
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, size in zip(axes, sizes):
        radar = radar_by_size[size]
        axis.imshow(
            radar,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_title(
            f"{size}-frame radar stack\n{int(radar.sum()):,} occupied cells",
            color="white",
        )
    for axis in axes:
        axis.set_facecolor("black")
        axis.axis("off")
    figure.suptitle(
        f"VoD {frame_id} | ego-motion-aligned radar occupancy",
        color="white",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150, facecolor="black")
    plt.close(figure)


def _print_table(rows: list[dict[str, float]]) -> None:
    print("\nGLOBAL RADAR-TO-LIDAR BEV OCCUPANCY AGREEMENT")
    print(
        f"{'Stack':>5} {'Cells/frame':>12} {'Density':>9} "
        f"{'Exact IoU':>10} {'Exact F1':>9} "
        f"{'IoU@0.2m':>10} {'F1@0.2m':>9} "
        f"{'IoU@0.5m':>10} {'F1@0.5m':>9}"
    )
    print("-" * 103)
    for row in rows:
        print(
            f"{int(row['stack_size']):5d} "
            f"{row['mean_radar_occupied_cells']:12.1f} "
            f"{row['radar_to_lidar_density_ratio']:8.2f}x "
            f"{row['exact_iou']:9.2%} {row['exact_f1']:9.2%} "
            f"{row['iou_0p2m']:10.2%} {row['f1_0p2m']:9.2%} "
            f"{row['iou_0p5m']:10.2%} {row['f1_0p5m']:9.2%}"
        )


def main() -> None:
    args = parse_args()
    stack_sizes = list(dict.fromkeys(args.stack_sizes))
    if not stack_sizes or any(size < 1 for size in stack_sizes):
        raise ValueError("--stack-sizes must contain positive integers")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when supplied")
    if args.visualize_count < 0:
        raise ValueError("--visualize-count must be non-negative")
    if args.resolution <= 0.0:
        raise ValueError("--resolution must be positive")
    if args.x_max <= args.x_min or args.y_max <= args.y_min:
        raise ValueError("BEV maxima must exceed minima")

    frame_ids = load_vod_split_ids(args.vod_root, args.split)
    rng = random.Random(args.seed)
    if args.limit is not None and args.limit < len(frame_ids):
        frame_ids = rng.sample(frame_ids, args.limit)
    variants = {
        size: f"radar_{size}frames"
        + (f"_{args.variant_suffix.strip('_')}" if args.variant_suffix else "")
        for size in stack_sizes
    }
    frames_by_size = {
        size: discover_vod_frames(
            args.vod_root,
            args.split,
            radar_variant=variant,
            frame_ids=frame_ids,
        )
        for size, variant in variants.items()
    }
    frames_by_size = {
        size: {frame.frame_id: frame for frame in frames}
        for size, frames in frames_by_size.items()
    }

    tolerance_labels = ((0.0, "exact"), (0.2, "0p2m"), (0.5, "0p5m"))
    totals = {
        size: {
            label: {
                "matched_predictions": 0.0,
                "matched_targets": 0.0,
                "prediction_count": 0.0,
                "target_count": 0.0,
            }
            for _tolerance, label in tolerance_labels
        }
        for size in stack_sizes
    }
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    visualization_ids = set(rng.sample(frame_ids, min(args.visualize_count, len(frame_ids))))

    for index, frame_id in enumerate(frame_ids, 1):
        reference_frame = frames_by_size[stack_sizes[0]][frame_id]
        lidar = _occupancy(
            load_vod_lidar(reference_frame.lidar_path),
            x_range=x_range,
            y_range=y_range,
            resolution=args.resolution,
        )
        radar_by_size = {}
        for size in stack_sizes:
            radar = _occupancy(
                _aligned_radar(frames_by_size[size][frame_id]),
                x_range=x_range,
                y_range=y_range,
                resolution=args.resolution,
            )
            radar_by_size[size] = radar
            for tolerance, label in tolerance_labels:
                _add_counts(
                    totals[size][label],
                    _metric_counts(
                        radar,
                        lidar,
                        tolerance_m=tolerance,
                        resolution=args.resolution,
                    ),
                )
        if frame_id in visualization_ids:
            _save_visualization(
                radar_by_size,
                args.output_root / "visualizations" / f"{frame_id}.png",
                frame_id,
            )
        if index % 100 == 0 or index == len(frame_ids):
            print(f"Evaluated {index}/{len(frame_ids)} frames", flush=True)

    rows = []
    for size in stack_sizes:
        row: dict[str, float] = {
            "stack_size": size,
            "frames": len(frame_ids),
            "mean_radar_occupied_cells": (
                totals[size]["exact"]["prediction_count"] / len(frame_ids)
            ),
            "mean_lidar_occupied_cells": (
                totals[size]["exact"]["target_count"] / len(frame_ids)
            ),
        }
        row["radar_to_lidar_density_ratio"] = (
            totals[size]["exact"]["prediction_count"]
            / totals[size]["exact"]["target_count"]
            if totals[size]["exact"]["target_count"]
            else 0.0
        )
        for _tolerance, label in tolerance_labels:
            metrics = _metrics(totals[size][label])
            for metric, value in metrics.items():
                key = f"{label}_{metric}" if label == "exact" else f"{metric}_{label}"
                row[key] = value
        rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "frames": len(frame_ids),
        "seed": args.seed,
        "bev": {
            "x_range": list(x_range),
            "y_range": list(y_range),
            "resolution_m": args.resolution,
        },
        "variants": variants,
        "metric_definition": (
            "Global occupied-cell precision/recall with bidirectional metric "
            "tolerance; tolerant IoU is derived from tolerant F1."
        ),
        "results": rows,
    }
    (args.output_root / "radar_stack_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_root / "radar_stack_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _print_table(rows)
    print(f"\nSaved results to {args.output_root}")


if __name__ == "__main__":
    main()
