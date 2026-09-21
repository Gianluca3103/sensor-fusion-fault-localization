"""Analyze occupied 3D voxels before choosing a point-capacity limit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.io_utils import atomic_write_json
from voxelization import (
    HardVoxelizer,
    load_voxelization_config,
)
from voxelization.inputs import (
    discover_sample_paths,
    load_aligned_point_inputs,
)


PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 100.0)


def histogram_percentile(histogram: Counter[int], percentile: float) -> float:
    total = sum(histogram.values())
    if total == 0:
        return 0.0
    rank = max(1, int(np.ceil(percentile / 100.0 * total)))
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= rank:
            return float(value)
    raise RuntimeError("Histogram percentile traversal failed")


def summarize(
    histogram: Counter[int],
    occupied_per_frame: list[int],
    configured_limit: int | None,
) -> dict:
    total_voxels = sum(histogram.values())
    total_points = sum(count * frequency for count, frequency in histogram.items())
    percentiles = {
        f"p{str(value).replace('.', '_')}": histogram_percentile(histogram, value)
        for value in PERCENTILES
    }
    # p99 was too aggressive for dense LiDAR voxels in the repository audit:
    # a small voxel tail can contain a large fraction of all returns.  Use
    # p99.9 as the reported finite-cap starting point and always expose the
    # actual point-loss calculation beside it.
    recommended = max(1, int(np.ceil(percentiles["p99_9"])))

    def truncation_at(limit: int) -> dict:
        truncated_voxels = sum(
            frequency for count, frequency in histogram.items() if count > limit
        )
        dropped_points = sum(
            (count - limit) * frequency
            for count, frequency in histogram.items()
            if count > limit
        )
        return {
            "limit": limit,
            "truncated_voxels": truncated_voxels,
            "voxel_fraction": truncated_voxels / total_voxels if total_voxels else 0.0,
            "dropped_points": dropped_points,
            "point_fraction": dropped_points / total_points if total_points else 0.0,
        }

    candidates = sorted(
        {
            16,
            32,
            64,
            recommended,
            max(1, int(np.ceil(percentiles["p99_5"]))),
            max(1, int(np.ceil(percentiles["p99_9"]))),
        }
    )
    result = {
        "occupied_voxels": total_voxels,
        "points_in_occupied_voxels": total_points,
        "points_per_occupied_voxel": {
            "mean": total_points / total_voxels if total_voxels else 0.0,
            **percentiles,
        },
        "occupied_voxels_per_frame": {
            "min": int(np.min(occupied_per_frame)) if occupied_per_frame else 0,
            "mean": float(np.mean(occupied_per_frame)) if occupied_per_frame else 0.0,
            "median": float(np.median(occupied_per_frame)) if occupied_per_frame else 0.0,
            "p95": float(np.percentile(occupied_per_frame, 95)) if occupied_per_frame else 0.0,
            "max": int(np.max(occupied_per_frame)) if occupied_per_frame else 0,
        },
        "recommended_max_points_per_voxel": recommended,
        "recommendation_rule": "ceil(global occupied-voxel p99.9)",
        "candidate_limit_truncation": [truncation_at(limit) for limit in candidates],
        "actual_truncation": (
            truncation_at(configured_limit)
            if configured_limit is not None
            else {
                "limit": None,
                "truncated_voxels": 0,
                "voxel_fraction": 0.0,
                "dropped_points": 0,
                "point_fraction": 0.0,
            }
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/voxelization_3d.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--limit-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar-source", choices=("clean", "faulty"), default="clean")
    parser.add_argument("--include-duplicate-frames", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_voxelization_config(args.config)
    paths = discover_sample_paths(
        args.data_root,
        args.split,
        limit=args.limit_samples,
        seed=args.seed,
        unique_frames=not args.include_duplicate_frames,
    )
    if not paths:
        raise FileNotFoundError(f"No samples found in {args.data_root / args.split}")
    histograms = {"lidar": Counter(), "radar": Counter()}
    occupied = {"lidar": [], "radar": []}
    dropped = {
        "lidar": {"nonfinite": 0, "out_of_range": 0},
        "radar": {"nonfinite": 0, "out_of_range": 0},
    }
    unlimited = HardVoxelizer(config.grid, max_points_per_voxel=None)
    for index, path in enumerate(paths, start=1):
        inputs = load_aligned_point_inputs(
            path, args.radar_root, lidar_source=args.lidar_source
        )
        for modality, points, names in (
            ("lidar", inputs.lidar_points, inputs.lidar_feature_names),
            ("radar", inputs.radar_points, inputs.radar_feature_names),
        ):
            result = unlimited.voxelize(points, names)
            histograms[modality].update(
                int(value) for value in result.original_num_points
            )
            occupied[modality].append(result.occupied_voxel_count)
            dropped[modality]["nonfinite"] += result.nonfinite_point_count
            dropped[modality]["out_of_range"] += result.out_of_range_point_count
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"Analyzed {index}/{len(paths)} frames", flush=True)

    report = {
        "samples": len(paths),
        "split": args.split,
        "lidar_source": args.lidar_source,
        "unique_physical_frames": not args.include_duplicate_frames,
        "grid": config.to_dict()["grid"],
        "dimensions_xyz": list(config.grid.dimensions_xyz),
        "sparse_coordinate_order": "zyx",
        "lidar": {
            **summarize(
                histograms["lidar"], occupied["lidar"], config.lidar.max_points_per_voxel
            ),
            "excluded_points": dropped["lidar"],
        },
        "radar": {
            **summarize(
                histograms["radar"], occupied["radar"], config.radar.max_points_per_voxel
            ),
            "excluded_points": dropped["radar"],
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
