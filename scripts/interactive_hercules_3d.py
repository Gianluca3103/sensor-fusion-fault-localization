"""Interactively inspect raw HeRCULES LiDAR and a V2-aligned radar stack.

Radar accumulation is causal. ``--radar-frames 0`` (the default) imposes no
numerical frame cap: every preceding radar scan that passes the history,
translation, rotation, synchronization, and pose-coverage gates is aligned to
the selected LiDAR frame. Dynamic tracks are motion compensated and the normal
temporal consistency filter is applied before visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from Fault_Localization_Model.hercules_dataset import (
    discover_hercules_frames,
    load_frame_radar,
    load_hercules_lidar,
)
from scripts.interactive_3d_voxels import InteractiveVoxelViewer
from voxelization import HardVoxelizer, load_voxelization_config


HERCULES_LIDAR_FIELDS = ("x", "y", "z", "reflectivity")
HERCULES_RADAR_FIELDS = (
    "x",
    "y",
    "z",
    "rcs",
    "radial_velocity",
    "compensated_radial_velocity",
    "time_index",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hercules-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--frame-index", type=int, default=0,
        help="Index within the selected split after deterministic scene/timestamp sorting",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--radar-frames", type=int, default=0,
        help="Maximum causal radar scans; 0 means no numerical cap",
    )
    parser.add_argument(
        "--max-history-s", type=float, default=1.0,
        help="Oldest permitted radar scan; increase cautiously for a denser stack",
    )
    parser.add_argument("--max-translation-m", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--max-radar-age-ms", type=float, default=100.0)
    parser.add_argument("--max-pose-gap-ms", type=float, default=200.0)
    parser.add_argument("--temporal-radius-m", type=float, default=0.75)
    parser.add_argument("--doppler-sign", choices=("auto", "1", "-1"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-raw-points", type=int, default=20000)
    parser.add_argument("--max-centers", type=int, default=5000)
    parser.add_argument("--max-cubes", type=int, default=100)
    parser.add_argument(
        "--save-snapshot", type=Path,
        help="Save the initial view before opening the interactive window",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Validate/save without opening a GUI; requires --save-snapshot",
    )
    args = parser.parse_args()
    if args.radar_frames < 0:
        parser.error("--radar-frames must be zero or positive")
    if args.no_show and args.save_snapshot is None:
        parser.error("--no-show requires --save-snapshot")

    frames = discover_hercules_frames(
        args.hercules_root,
        args.split,
        radar_variant="hercules_v2_interactive",
        split_manifest=args.split_manifest,
    )
    if not 0 <= args.frame_index < len(frames):
        raise IndexError(f"frame-index must be in [0, {len(frames) - 1}]")
    frame = frames[args.frame_index]
    lidar = load_hercules_lidar(frame.lidar_path)
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
    rows = alignment["alignment_rows"]

    voxel_config = load_voxelization_config(args.config)
    lidar_voxelizer = HardVoxelizer(
        voxel_config.grid,
        max_points_per_voxel=voxel_config.lidar.max_points_per_voxel,
    )
    radar_voxelizer = HardVoxelizer(
        voxel_config.grid,
        max_points_per_voxel=voxel_config.radar.max_points_per_voxel,
    )
    lidar_voxels = lidar_voxelizer.voxelize(lidar, HERCULES_LIDAR_FIELDS)
    radar_voxels = radar_voxelizer.voxelize(radar, HERCULES_RADAR_FIELDS)
    raw = {"lidar": lidar[:, :3], "radar": radar[:, :3]}
    centers = {
        "lidar": lidar_voxelizer.voxel_centers(lidar_voxels.voxel_coords),
        "radar": radar_voxelizer.voxel_centers(radar_voxels.voxel_coords),
    }
    sample_name = f"{frame.lidar_path.parent.parent.parent.name}_{frame.lidar_path.stem}"
    viewer = InteractiveVoxelViewer(
        raw,
        centers,
        voxel_config.grid,
        args.output_root,
        sample_name=sample_name,
        max_raw_points=args.max_raw_points,
        max_centers=args.max_centers,
        max_cubes=args.max_cubes,
        seed=args.seed,
    )

    oldest_age = max((float(row["age_s"]) for row in rows), default=0.0)
    print(f"HeRCULES frame: {frame.lidar_path}")
    print(f"Split frame index: {args.frame_index}/{len(frames) - 1}")
    print(f"LiDAR: {len(lidar):,} points, {len(centers['lidar']):,} occupied voxels")
    print(
        f"Radar: {len(radar):,} filtered points, "
        f"{len(centers['radar']):,} occupied voxels"
    )
    print(
        f"Accepted radar scans: {len(rows)} "
        f"(oldest={oldest_age:.3f}s, numerical cap="
        f"{'none' if args.radar_frames == 0 else args.radar_frames})"
    )
    print(
        "Gates: "
        f"history={args.max_history_s:g}s, "
        f"translation={args.max_translation_m:g}m, "
        f"rotation={args.max_rotation_deg:g}deg"
    )
    print(f"Temporal filter counts: {alignment['filter_counts']}")
    print(f"Confirmed dynamic tracks: {alignment['confirmed_tracks']}")
    print(f"Motion-compensated radar points: {alignment['motion_compensated_points']}")

    if args.save_snapshot is not None:
        args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.figure.savefig(args.save_snapshot, dpi=180, bbox_inches="tight")
        print(f"Saved initial snapshot: {args.save_snapshot}")
    if args.no_show:
        plt.close(viewer.figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
