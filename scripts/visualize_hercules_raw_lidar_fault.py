"""Inject a fault into raw HeRCULES LiDAR and visualize the 3D result.

The fault is applied before voxelization.  The output NPZ retains the complete
raw clean/faulty point clouds and exact provenance, while the figures use the
configured model volume only to keep their axes directly comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from Fault_Localization_Model.hercules_dataset import (
    discover_hercules_frames,
    load_hercules_lidar,
)
from Fault_Localization_Model.io_utils import atomic_savez, atomic_write_json
from voxelization import load_voxelization_config


PROJECTIONS = (
    (0, 1, "XY", "x forward [m]", "y lateral [m]"),
    (0, 2, "XZ", "x forward [m]", "z vertical [m]"),
    (1, 2, "YZ", "y lateral [m]", "z vertical [m]"),
)
COLORS = {
    "clean": "#00d5ff",
    "faulty": "#42f56f",
    "retained": "#42f56f",
    "moved": "#ffd23f",
    "removed": "#ff5a36",
    "synthetic": "#ff00d4",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _volume_mask(points: np.ndarray, grid) -> np.ndarray:
    if not len(points):
        return np.zeros(0, dtype=bool)
    return (
        (points[:, 0] >= grid.x_range[0])
        & (points[:, 0] < grid.x_range[1])
        & (points[:, 1] >= grid.y_range[0])
        & (points[:, 1] < grid.y_range[1])
        & (points[:, 2] >= grid.z_range[0])
        & (points[:, 2] < grid.z_range[1])
    )


def _sample(points: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.sort(rng.choice(len(points), maximum, replace=False))
    return points[indices]


def _change_sets(clean: np.ndarray, faulty, movement_tolerance_m: float) -> dict[str, np.ndarray]:
    source_ids = np.asarray(faulty.source_ids, dtype=np.int64)
    derived = source_ids >= 0
    present_ids = source_ids[derived]
    removed_mask = np.ones(len(clean), dtype=bool)
    removed_mask[present_ids] = False

    displacement = np.zeros(len(faulty.points), dtype=np.float32)
    if np.any(derived):
        displacement[derived] = np.linalg.norm(
            faulty.points[derived, :3] - clean[source_ids[derived], :3], axis=1
        )
    moved = derived & (displacement > movement_tolerance_m)
    retained = derived & ~moved
    synthetic = ~derived
    return {
        "retained": faulty.points[retained],
        "moved": faulty.points[moved],
        "removed": clean[removed_mask],
        "synthetic": faulty.points[synthetic],
        "displacement_m": displacement,
        "removed_mask": removed_mask,
        "moved_mask": moved,
        "retained_mask": retained,
        "synthetic_mask": synthetic,
    }


def _style_axis(axis, title: str, x_label: str, y_label: str) -> None:
    axis.set_facecolor("black")
    axis.set_title(title, color="white")
    axis.set_xlabel(x_label, color="white")
    axis.set_ylabel(y_label, color="white")
    axis.tick_params(colors="white")
    axis.grid(color="#555555", alpha=0.22, linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color("#777777")


def _projection_limits(grid, name: str):
    if name == "XY":
        return grid.x_range, grid.y_range
    if name == "XZ":
        return grid.x_range, grid.z_range
    return grid.y_range, grid.z_range


def save_projection_comparison(
    destination: Path,
    clean: np.ndarray,
    faulty_points: np.ndarray,
    changes: dict[str, np.ndarray],
    grid,
    maximum: int,
    seed: int,
    title: str,
) -> None:
    rng = np.random.default_rng(seed)
    clean = _sample(clean[_volume_mask(clean, grid)], maximum, rng)
    faulty_points = _sample(
        faulty_points[_volume_mask(faulty_points, grid)], maximum, rng
    )
    visible_changes = {
        name: _sample(points[_volume_mask(points, grid)], maximum, rng)
        for name, points in changes.items()
        if isinstance(points, np.ndarray) and points.ndim == 2
    }

    figure, axes = plt.subplots(3, 3, figsize=(16, 14), facecolor="black")
    for column, (horizontal, vertical, name, x_label, y_label) in enumerate(PROJECTIONS):
        limits = _projection_limits(grid, name)
        for row, row_title in enumerate(("Clean raw LiDAR", "Faulty raw LiDAR", "Fault changes")):
            axis = axes[row, column]
            _style_axis(axis, f"{row_title}: {name}", x_label, y_label)
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
        axes[0, column].scatter(
            clean[:, horizontal], clean[:, vertical], s=0.7,
            c=COLORS["clean"], alpha=0.8, linewidths=0,
        )
        axes[1, column].scatter(
            faulty_points[:, horizontal], faulty_points[:, vertical], s=0.7,
            c=COLORS["faulty"], alpha=0.8, linewidths=0,
        )
        for category in ("retained", "moved", "removed", "synthetic"):
            points = visible_changes[category]
            if len(points):
                axes[2, column].scatter(
                    points[:, horizontal], points[:, vertical],
                    s=0.8 if category == "retained" else 2.0,
                    c=COLORS[category], alpha=0.8, linewidths=0, label=category,
                )
        if column == 2:
            legend = axes[2, column].legend(
                loc="upper right", facecolor="black", framealpha=0.75
            )
            if legend:
                for text in legend.get_texts():
                    text.set_color("white")
    figure.suptitle(title, color="white", fontsize=15)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _scatter_3d(axis, points: np.ndarray, color: str, label: str, size: float) -> None:
    if len(points):
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            s=size, c=color, alpha=0.72, linewidths=0, label=label,
        )


def save_3d_comparison(
    destination: Path,
    clean: np.ndarray,
    faulty_points: np.ndarray,
    changes: dict[str, np.ndarray],
    grid,
    maximum: int,
    seed: int,
    title: str,
) -> None:
    rng = np.random.default_rng(seed)
    clean = _sample(clean[_volume_mask(clean, grid)], maximum, rng)
    faulty_points = _sample(
        faulty_points[_volume_mask(faulty_points, grid)], maximum, rng
    )
    visible_changes = {
        name: _sample(points[_volume_mask(points, grid)], maximum, rng)
        for name, points in changes.items()
        if isinstance(points, np.ndarray) and points.ndim == 2
    }
    figure = plt.figure(figsize=(19, 6.5), facecolor="black")
    axes = [figure.add_subplot(1, 3, index, projection="3d") for index in (1, 2, 3)]
    _scatter_3d(axes[0], clean, COLORS["clean"], "clean", 0.7)
    _scatter_3d(axes[1], faulty_points, COLORS["faulty"], "faulty", 0.7)
    for category in ("retained", "moved", "removed", "synthetic"):
        _scatter_3d(
            axes[2], visible_changes[category], COLORS[category], category,
            0.7 if category == "retained" else 2.0,
        )
    for axis, axis_title in zip(axes, ("Clean raw LiDAR", "Faulty raw LiDAR", "Fault changes")):
        axis.set_facecolor("black")
        axis.set_title(axis_title, color="white")
        axis.set_xlabel("x forward [m]", color="white")
        axis.set_ylabel("y lateral [m]", color="white")
        axis.set_zlabel("z vertical [m]", color="white")
        axis.set_xlim(*grid.x_range)
        axis.set_ylim(*grid.y_range)
        axis.set_zlim(*grid.z_range)
        axis.tick_params(colors="white")
        axis.view_init(elev=23, azim=-62)
        axis.grid(True, alpha=0.2)
    legend = axes[2].legend(loc="upper right", facecolor="black", framealpha=0.75)
    if legend:
        for text in legend.get_texts():
            text.set_color("white")
    figure.suptitle(title, color="white", fontsize=15)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hercules-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--fault", choices=sorted(SUPPORTED_CORRUPTIONS), default="fog_sim")
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--movement-tolerance-m", type=float, default=0.05,
        help="Minimum source-point displacement classified as movement",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-display-points", type=int, default=75000)
    parser.add_argument(
        "--artifact-compression-level", type=int, default=1,
        help="NPZ ZIP level; point values remain lossless",
    )
    args = parser.parse_args()
    validate_fault_spec(args.fault, args.severity)
    if args.frame_index < 0:
        parser.error("--frame-index must be non-negative")
    if args.max_display_points < 1:
        parser.error("--max-display-points must be positive")
    if args.movement_tolerance_m < 0:
        parser.error("--movement-tolerance-m must be non-negative")

    frames = discover_hercules_frames(
        args.hercules_root,
        args.split,
        radar_variant="raw_lidar_fault_visualization",
        split_manifest=args.split_manifest,
    )
    if args.frame_index >= len(frames):
        raise IndexError(f"frame-index must be in [0, {len(frames) - 1}]")
    frame = frames[args.frame_index]
    clean = load_hercules_lidar(frame.lidar_path).astype(np.float32, copy=False)
    if not len(clean):
        raise ValueError(f"HeRCULES LiDAR frame is empty: {frame.lidar_path}")

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

    scene = frame.lidar_path.parent.parent.parent.name
    sample_name = f"{scene}_{frame.lidar_path.stem}_{args.fault}_s{args.severity}"
    output = args.output_root / sample_name
    output.mkdir(parents=True, exist_ok=True)
    title = (
        f"HeRCULES {scene} | {frame.lidar_path.stem} | "
        f"{args.fault} severity {args.severity} | seed {args.seed}"
    )
    save_3d_comparison(
        output / "clean_vs_faulty_raw_3d.png",
        clean, faulty.points, changes, grid,
        args.max_display_points, args.seed, title,
    )
    save_projection_comparison(
        output / "clean_vs_faulty_xyz_projections.png",
        clean, faulty.points, changes, grid,
        args.max_display_points, args.seed, title,
    )

    metadata = {
        "dataset": "HeRCULES",
        "scene": scene,
        "split": args.split,
        "frame_index": args.frame_index,
        "lidar_path": str(frame.lidar_path),
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
