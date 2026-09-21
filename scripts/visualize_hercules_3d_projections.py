"""Save HeRCULES LiDAR/radar XY, XZ, and YZ projection comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Fault_Localization_Model.hercules_dataset import (
    discover_hercules_frames,
    load_frame_radar,
    load_hercules_lidar,
)
from Fault_Localization_Model.io_utils import atomic_write_json
from voxelization import (
    HardVoxelizer,
    VoxelTemporalConsistencyConfig,
    filter_temporally_consistent_radar_voxels,
    load_voxelization_config,
)


LIDAR_FIELDS = ("x", "y", "z", "reflectivity")
RADAR_FIELDS = (
    "x", "y", "z", "rcs", "radial_velocity",
    "compensated_radial_velocity", "time_index",
)
COLORS = {"lidar": "#00d5ff", "radar": "#ff00d4"}
PROJECTIONS = (
    (0, 1, "XY", "x forward [m]", "y lateral [m]"),
    (0, 2, "XZ", "x forward [m]", "z vertical [m]"),
    (1, 2, "YZ", "y lateral [m]", "z vertical [m]"),
)


def sample_rows(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.random.default_rng(seed).choice(len(values), maximum, replace=False)
    return values[np.sort(indices)]


def axis_limits(grid, projection: str):
    if projection == "XY":
        return grid.x_range, grid.y_range
    if projection == "XZ":
        return grid.x_range, grid.z_range
    return grid.y_range, grid.z_range


def save_projection_grid(
    path: Path,
    values: dict[str, np.ndarray],
    grid,
    *,
    title: str,
    maximum: int,
    seed: int,
) -> None:
    displayed = {
        modality: sample_rows(points, maximum, seed + index)
        for index, (modality, points) in enumerate(values.items())
    }
    figure, axes = plt.subplots(3, 3, figsize=(16, 14), facecolor="black")
    rows = (
        ("LiDAR", ("lidar",)),
        ("Stacked radar", ("radar",)),
        ("LiDAR + stacked radar", ("lidar", "radar")),
    )
    for row_index, (row_title, modalities) in enumerate(rows):
        for column, (horizontal, vertical, name, x_label, y_label) in enumerate(PROJECTIONS):
            axis = axes[row_index, column]
            axis.set_facecolor("black")
            for modality in modalities:
                points = displayed[modality]
                if len(points):
                    axis.scatter(
                        points[:, horizontal], points[:, vertical],
                        s=0.6 if modality == "lidar" else 4.0,
                        c=COLORS[modality],
                        alpha=0.75,
                        linewidths=0,
                        label=modality,
                    )
            x_limits, y_limits = axis_limits(grid, name)
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
            axis.set_xlabel(x_label, color="white")
            axis.set_ylabel(y_label, color="white")
            axis.set_title(f"{row_title}: {name}", color="white")
            axis.tick_params(colors="white")
            for spine in axis.spines.values():
                spine.set_color("#777777")
            axis.grid(color="#555555", alpha=0.22, linewidth=0.5)
            if len(modalities) > 1:
                legend = axis.legend(loc="upper right", facecolor="black", framealpha=0.7)
                for text in legend.get_texts():
                    text.set_color("white")
    figure.suptitle(title, color="white", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hercules-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--radar-frames", type=int, default=0,
        help="Numerical scan cap; 0 uses every scan passing the quality gates",
    )
    parser.add_argument("--max-history-s", type=float, default=1.0)
    parser.add_argument("--max-translation-m", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--max-radar-age-ms", type=float, default=100.0)
    parser.add_argument("--max-pose-gap-ms", type=float, default=200.0)
    parser.add_argument("--temporal-radius-m", type=float, default=0.75)
    parser.add_argument("--doppler-sign", choices=("auto", "1", "-1"), default="auto")
    parser.add_argument(
        "--voxel-temporal-filter", action="store_true",
        help="Require local XYZ voxel support from multiple distinct radar scans",
    )
    parser.add_argument("--voxel-min-scans", type=int, default=3)
    parser.add_argument("--voxel-min-scan-fraction", type=float, default=0.15)
    parser.add_argument("--voxel-neighbor-radius-cells", type=int, default=1)
    parser.add_argument(
        "--voxel-preserve-current-scan", action="store_true",
        help="Keep all newest-scan points even without temporal support",
    )
    parser.add_argument("--max-points-per-sensor", type=int, default=75000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.radar_frames < 0:
        parser.error("--radar-frames must be zero or positive")

    frames = discover_hercules_frames(
        args.hercules_root,
        args.split,
        radar_variant="hercules_v2_projection",
        split_manifest=args.split_manifest,
    )
    if not 0 <= args.frame_index < len(frames):
        raise IndexError(f"frame-index must be in [0, {len(frames) - 1}]")
    frame = frames[args.frame_index]
    lidar = load_hercules_lidar(frame.lidar_path)
    config = load_voxelization_config(args.config)
    radar_config = {
        "hercules_radar_frames": args.radar_frames,
        "hercules_temporal_radius": args.temporal_radius_m,
        "hercules_max_radar_age_ms": args.max_radar_age_ms,
        "hercules_max_pose_gap_ms": args.max_pose_gap_ms,
        "hercules_stack": {
            "max_frames": args.radar_frames or None,
            "max_age_s": args.max_history_s,
            "max_translation_m": args.max_translation_m,
            "max_rotation_deg": args.max_rotation_deg,
        },
        "hercules_tracking": {"doppler_sign": args.doppler_sign},
        "_hercules_cache_source_preprocessing": True,
    }
    _, radar, _ = load_frame_radar(frame, radar_config)
    alignment = radar_config["_hercules_alignment"]
    voxel_temporal_stats = None
    if args.voxel_temporal_filter:
        radar, voxel_temporal_stats = filter_temporally_consistent_radar_voxels(
            radar,
            config.grid,
            VoxelTemporalConsistencyConfig(
                min_scans=args.voxel_min_scans,
                min_scan_fraction=args.voxel_min_scan_fraction,
                neighbor_radius_cells=args.voxel_neighbor_radius_cells,
                preserve_current_scan=args.voxel_preserve_current_scan,
            ),
        )

    lidar_voxelizer = HardVoxelizer(
        config.grid, max_points_per_voxel=config.lidar.max_points_per_voxel
    )
    radar_voxelizer = HardVoxelizer(
        config.grid, max_points_per_voxel=config.radar.max_points_per_voxel
    )
    lidar_voxels = lidar_voxelizer.voxelize(lidar, LIDAR_FIELDS)
    radar_voxels = radar_voxelizer.voxelize(radar, RADAR_FIELDS)
    raw = {"lidar": lidar[:, :3], "radar": radar[:, :3]}
    centers = {
        "lidar": lidar_voxelizer.voxel_centers(lidar_voxels.voxel_coords),
        "radar": radar_voxelizer.voxel_centers(radar_voxels.voxel_coords),
    }

    scene = frame.lidar_path.parent.parent.parent.name
    sample_name = f"{scene}_{frame.lidar_path.stem}"
    output = args.output_root / sample_name
    output.mkdir(parents=True, exist_ok=True)
    rows = alignment["alignment_rows"]
    oldest_age = max((float(row["age_s"]) for row in rows), default=0.0)
    common_title = (
        f"HeRCULES {scene} | LiDAR {frame.lidar_path.stem} | "
        f"{len(rows)} aligned radar scans | oldest {oldest_age:.3f}s"
    )
    save_projection_grid(
        output / "raw_xyz_projections.png",
        raw,
        config.grid,
        title=f"{common_title} | raw points",
        maximum=args.max_points_per_sensor,
        seed=args.seed,
    )
    save_projection_grid(
        output / "voxel_center_xyz_projections.png",
        centers,
        config.grid,
        title=f"{common_title} | occupied voxel centers",
        maximum=args.max_points_per_sensor,
        seed=args.seed,
    )
    summary = {
        "scene": scene,
        "lidar_path": str(frame.lidar_path),
        "split": args.split,
        "frame_index": args.frame_index,
        "lidar_points": len(lidar),
        "filtered_radar_points": len(radar),
        "lidar_occupied_voxels": lidar_voxels.occupied_voxel_count,
        "radar_occupied_voxels": radar_voxels.occupied_voxel_count,
        "accepted_radar_scans": len(rows),
        "oldest_radar_age_s": oldest_age,
        "radar_frame_cap": args.radar_frames or None,
        "stack_gates": radar_config["hercules_stack"],
        "temporal_filter_counts": alignment["filter_counts"],
        "confirmed_tracks": alignment["confirmed_tracks"],
        "motion_compensated_points": alignment["motion_compensated_points"],
        "voxel_temporal_filter": voxel_temporal_stats,
        "alignment_rows": rows,
    }
    atomic_write_json(output / "projection_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "alignment_rows"}, indent=2))
    print(f"Saved raw projections: {output / 'raw_xyz_projections.png'}")
    print(f"Saved voxel projections: {output / 'voxel_center_xyz_projections.png'}")
    print(f"Saved alignment summary: {output / 'projection_summary.json'}")


if __name__ == "__main__":
    main()
