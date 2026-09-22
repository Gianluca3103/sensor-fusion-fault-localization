"""Inject a fault into raw View-of-Delft LiDAR and visualize the 3D result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.config.defaults import (
    DEFAULT_FOG_ROOT,
    DEFAULT_INJECTOR_ROOT,
)
from Fault_Localization_Model.data_injection_utils import (
    SUPPORTED_CORRUPTIONS,
    validate_fault_spec,
)
from Fault_Localization_Model.fault_injector import (
    inject_fault,
    load_fault_injector,
    remove_added_returns,
)
from Fault_Localization_Model.io_utils import atomic_savez, atomic_write_json
from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_lidar,
    load_vod_split_ids,
    resolve_vod_public_root,
)
from scripts.visualize_hercules_raw_lidar_fault import (
    _change_sets,
    _jsonable,
    _volume_mask,
    save_3d_comparison,
    save_projection_comparison,
)
from voxelization import load_voxelization_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "val", "test", "train_val"), default="val"
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--frame-index", type=int, default=0)
    selection.add_argument("--frame-id")
    parser.add_argument("--fault", choices=sorted(SUPPORTED_CORRUPTIONS), default="fog_sim")
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--movement-tolerance-m", type=float, default=0.05)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-display-points", type=int, default=75000)
    parser.add_argument("--artifact-compression-level", type=int, default=1)
    args = parser.parse_args()
    validate_fault_spec(args.fault, args.severity)
    if args.frame_index is not None and args.frame_index < 0:
        parser.error("--frame-index must be non-negative")
    if args.max_display_points < 1:
        parser.error("--max-display-points must be positive")
    if args.movement_tolerance_m < 0:
        parser.error("--movement-tolerance-m must be non-negative")

    public_root = resolve_vod_public_root(args.vod_root)
    frame_ids = load_vod_split_ids(public_root, args.split)
    if args.frame_id is not None:
        if args.frame_id not in frame_ids:
            raise ValueError(
                f"Frame {args.frame_id!r} is not part of the official {args.split} split"
            )
        frame_id = args.frame_id
        frame_index = frame_ids.index(frame_id)
    else:
        if args.frame_index >= len(frame_ids):
            raise IndexError(f"frame-index must be in [0, {len(frame_ids) - 1}]")
        frame_index = args.frame_index
        frame_id = frame_ids[frame_index]
    lidar_path = public_root / "lidar" / "training" / "velodyne" / f"{frame_id}.bin"
    if not lidar_path.is_file():
        raise FileNotFoundError(f"VoD LiDAR frame is missing: {lidar_path}")
    clean = load_vod_lidar(lidar_path).astype(np.float32, copy=False)
    if not len(clean):
        raise ValueError(f"VoD LiDAR frame is empty: {lidar_path}")

    injector = load_fault_injector(DEFAULT_INJECTOR_ROOT)
    clean_ids = np.arange(len(clean), dtype=np.int64)
    injected, injection_metadata = inject_fault(
        args.fault,
        clean.copy(),
        clean_ids,
        args.severity,
        DEFAULT_INJECTOR_ROOT,
        DEFAULT_FOG_ROOT,
        lidar_corruptions=injector,
        rng_seed=args.seed,
    )
    injected_point_count = len(injected.points)
    faulty, added_particles_removed = remove_added_returns(injected)
    changes = _change_sets(clean, faulty, args.movement_tolerance_m)
    config = load_voxelization_config(args.config)
    grid = config.grid

    sample_name = f"{frame_id}_{args.fault}_s{args.severity}"
    output = args.output_root / sample_name
    output.mkdir(parents=True, exist_ok=True)
    title = (
        f"View-of-Delft {frame_id} | {args.split} | "
        f"{args.fault} severity {args.severity} | seed {args.seed}"
    )
    save_3d_comparison(
        output / "clean_vs_faulty_raw_3d.png",
        clean,
        faulty.points,
        changes,
        grid,
        args.max_display_points,
        args.seed,
        title,
    )
    save_projection_comparison(
        output / "clean_vs_faulty_xyz_projections.png",
        clean,
        faulty.points,
        changes,
        grid,
        args.max_display_points,
        args.seed,
        title,
    )

    metadata = {
        "dataset": "View-of-Delft",
        "split": args.split,
        "frame_index": frame_index,
        "frame_id": frame_id,
        "lidar_path": str(lidar_path),
        "fault": args.fault,
        "severity": args.severity,
        "seed": args.seed,
        "fault_applied_before_voxelization": True,
        "added_particle_filter": "exact_provenance_source_id_negative",
        "visualization_grid": {
            "x_range": grid.x_range,
            "y_range": grid.y_range,
            "z_range": grid.z_range,
        },
        "counts": {
            "clean_raw": len(clean),
            "faulty_raw": len(faulty.points),
            "injected_faulty_before_particle_filter": injected_point_count,
            "added_particles_removed": added_particles_removed,
            "clean_visible": int(_volume_mask(clean, grid).sum()),
            "faulty_visible": int(_volume_mask(faulty.points, grid).sum()),
            "source_returns_retained": len(changes["retained"]),
            "source_returns_moved": len(changes["moved"]),
            "clean_returns_missing_after_fault": len(changes["removed"]),
            "synthetic_returns_remaining": len(changes["synthetic"]),
        },
        "movement_tolerance_m": args.movement_tolerance_m,
        "injection_metadata": injection_metadata,
    }
    metadata_json = json.dumps(_jsonable(metadata), sort_keys=True)
    atomic_savez(
        output / "raw_lidar_fault.npz",
        compression_level=args.artifact_compression_level,
        clean_points=clean,
        clean_point_ids=clean_ids,
        faulty_points=faulty.points,
        faulty_point_ids=faulty.point_ids,
        faulty_source_ids=faulty.source_ids,
        faulty_injector_labels=faulty.injector_labels,
        metadata_json=np.asarray(metadata_json),
    )
    atomic_write_json(output / "summary.json", _jsonable(metadata))
    print(json.dumps(_jsonable(metadata), indent=2, sort_keys=True))
    print(f"Saved raw fault artifact: {output / 'raw_lidar_fault.npz'}")
    print(f"Saved full 3D comparison: {output / 'clean_vs_faulty_raw_3d.png'}")
    print(
        "Saved XY/XZ/YZ comparison: "
        f"{output / 'clean_vs_faulty_xyz_projections.png'}"
    )


if __name__ == "__main__":
    main()
