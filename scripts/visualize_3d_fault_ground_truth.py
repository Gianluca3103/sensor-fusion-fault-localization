"""Build and visualize exact 3D voxel fault targets from a raw fault artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez, atomic_write_json
from voxelization import (
    HardVoxelizer,
    build_voxel_fault_targets,
    load_voxelization_config,
)


COLORS = {
    "preserve": "#42f56f",
    "repair": "#ff9f1c",
    "remove": "#ff00d4",
    "both": "#ffffff",
}
PROJECTIONS = (
    (2, 1, "XY", "x forward [m]", "y lateral [m]"),
    (2, 0, "XZ", "x forward [m]", "z vertical [m]"),
    (1, 0, "YZ", "y lateral [m]", "z vertical [m]"),
)


def _coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(np.int32, copy=False)


def _sample(points: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.sort(rng.choice(len(points), maximum, replace=False))]


def _centers(voxelizer: HardVoxelizer, mask: np.ndarray) -> np.ndarray:
    coordinates = _coords(mask)
    if not len(coordinates):
        return np.empty((0, 3), dtype=np.float32)
    return voxelizer.voxel_centers(coordinates)


def _style(axis, title: str, x_label: str, y_label: str) -> None:
    axis.set_facecolor("black")
    axis.set_title(title, color="white")
    axis.set_xlabel(x_label, color="white")
    axis.set_ylabel(y_label, color="white")
    axis.tick_params(colors="white")
    axis.grid(color="#555555", alpha=0.22, linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color("#777777")


def _limits(grid, name: str):
    if name == "XY":
        return grid.x_range, grid.y_range
    if name == "XZ":
        return grid.x_range, grid.z_range
    return grid.y_range, grid.z_range


def save_projection_figure(path: Path, target_points: dict[str, np.ndarray], grid) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="black")
    for axis, (horizontal, vertical, name, x_label, y_label) in zip(axes, PROJECTIONS):
        _style(axis, f"3D fault ground truth: {name}", x_label, y_label)
        x_limits, y_limits = _limits(grid, name)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        for category in ("preserve", "repair", "remove", "both"):
            points = target_points[category]
            if len(points):
                axis.scatter(
                    points[:, horizontal], points[:, vertical],
                    s=1.0 if category == "preserve" else 4.0,
                    c=COLORS[category], alpha=0.82, linewidths=0, label=category,
                )
        if name == "YZ":
            legend = axis.legend(loc="upper right", facecolor="black", framealpha=0.75)
            for text in legend.get_texts():
                text.set_color("white")
    figure.suptitle(
        "Green: preserve | Orange: reconstruct | Magenta: remove | White: both",
        color="white",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_3d_figure(path: Path, target_points: dict[str, np.ndarray], grid) -> None:
    figure = plt.figure(figsize=(12, 8), facecolor="black")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("black")
    for category in ("preserve", "repair", "remove", "both"):
        points = target_points[category]
        if len(points):
            axis.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=1.0 if category == "preserve" else 5.0,
                c=COLORS[category], alpha=0.78, linewidths=0, label=category,
            )
    axis.set_xlim(*grid.x_range)
    axis.set_ylim(*grid.y_range)
    axis.set_zlim(*grid.z_range)
    axis.set_xlabel("x forward [m]", color="white")
    axis.set_ylabel("y lateral [m]", color="white")
    axis.set_zlabel("z vertical [m]", color="white")
    axis.tick_params(colors="white")
    axis.set_title("Exact provenance-derived 3D fault targets", color="white")
    axis.view_init(elev=23, azim=-62)
    legend = axis.legend(loc="upper right", facecolor="black", framealpha=0.75)
    if legend:
        for text in legend.get_texts():
            text.set_color("white")
    figure.tight_layout()
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_slice_montage(path: Path, targets, grid, slices: int) -> None:
    occupied_z = np.flatnonzero(np.any(targets.change_mask, axis=(1, 2)))
    if not len(occupied_z):
        occupied_z = np.arange(grid.dimensions_zyx[0])
    selected = np.unique(
        np.round(np.linspace(occupied_z[0], occupied_z[-1], min(slices, len(occupied_z))))
        .astype(int)
    )
    columns = min(4, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.3 * rows), facecolor="black")
    axes = np.atleast_1d(axes).reshape(-1)
    cmap = ListedColormap(["#000000", COLORS["preserve"], COLORS["repair"], COLORS["remove"], COLORS["both"]])
    for axis, z_index in zip(axes, selected):
        labels = np.zeros(targets.repair_mask.shape[1:], dtype=np.uint8)
        labels[targets.preserve_mask[z_index]] = 1
        labels[targets.repair_mask[z_index]] = 2
        labels[targets.remove_mask[z_index]] = 3
        labels[targets.repair_mask[z_index] & targets.remove_mask[z_index]] = 4
        z_center = grid.z_range[0] + (z_index + 0.5) * grid.voxel_size[2]
        axis.imshow(
            labels,
            origin="lower",
            extent=(*grid.x_range, *grid.y_range),
            cmap=cmap,
            vmin=0,
            vmax=4,
            interpolation="nearest",
            aspect="equal",
        )
        _style(axis, f"z={z_center:.2f} m (index {z_index})", "x forward [m]", "y lateral [m]")
    for axis in axes[len(selected):]:
        axis.axis("off")
    figure.suptitle(
        "Per-height 3D target slices — green preserve, orange repair, magenta remove, white both",
        color="white",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--movement-tolerance-m", type=float, default=0.05)
    parser.add_argument("--feature-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-display-voxels", type=int, default=150000)
    parser.add_argument("--height-slices", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.movement_tolerance_m < 0 or args.feature_tolerance < 0:
        parser.error("tolerances must be non-negative")
    if args.max_display_voxels < 1 or args.height_slices < 1:
        parser.error("display limits must be positive")
    if not args.artifact.is_file():
        raise FileNotFoundError(f"Raw fault artifact not found: {args.artifact}")

    with np.load(args.artifact, allow_pickle=False) as archive:
        clean = archive["clean_points"]
        faulty = archive["faulty_points"]
        source_ids = archive["faulty_source_ids"]
        source_metadata = json.loads(str(archive["metadata_json"].item()))
    config = load_voxelization_config(args.config)
    targets = build_voxel_fault_targets(
        clean,
        faulty,
        source_ids,
        config.grid,
        movement_tolerance_m=args.movement_tolerance_m,
        feature_tolerance=args.feature_tolerance,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    voxelizer = HardVoxelizer(config.grid)
    repair_only = targets.repair_mask & ~targets.remove_mask
    remove_only = targets.remove_mask & ~targets.repair_mask
    both = targets.repair_mask & targets.remove_mask
    rng = np.random.default_rng(args.seed)
    target_points = {
        "preserve": _sample(_centers(voxelizer, targets.preserve_mask), args.max_display_voxels, rng),
        "repair": _sample(_centers(voxelizer, repair_only), args.max_display_voxels, rng),
        "remove": _sample(_centers(voxelizer, remove_only), args.max_display_voxels, rng),
        "both": _sample(_centers(voxelizer, both), args.max_display_voxels, rng),
    }

    save_projection_figure(
        args.output_root / "fault_ground_truth_xyz_projections.png",
        target_points,
        config.grid,
    )
    save_3d_figure(
        args.output_root / "fault_ground_truth_3d.png", target_points, config.grid
    )
    save_slice_montage(
        args.output_root / "fault_ground_truth_height_slices.png",
        targets,
        config.grid,
        args.height_slices,
    )

    atomic_savez(
        args.output_root / "fault_ground_truth_3d.npz",
        compression_level=1,
        clean_count=targets.clean_count,
        faulty_count=targets.faulty_count,
        stable_count=targets.stable_count,
        damaged_clean_count=targets.damaged_clean_count,
        unreliable_faulty_count=targets.unreliable_faulty_count,
        clean_occupancy=targets.clean_occupancy,
        faulty_occupancy=targets.faulty_occupancy,
        preserve_mask=targets.preserve_mask,
        repair_mask=targets.repair_mask,
        remove_mask=targets.remove_mask,
        change_mask=targets.change_mask,
        repair_fraction=targets.repair_fraction.astype(np.float16),
        removal_fraction=targets.removal_fraction.astype(np.float16),
    )
    summary = {
        "source_artifact": str(args.artifact),
        "source_metadata": source_metadata,
        "coordinate_order": "zyx",
        "grid_shape_zyx": config.grid.dimensions_zyx,
        "movement_tolerance_m": args.movement_tolerance_m,
        "feature_tolerance": args.feature_tolerance,
        "points": {
            "clean_in_grid": targets.clean_points_in_grid,
            "faulty_in_grid": targets.faulty_points_in_grid,
            "stable": targets.stable_points,
            "damaged_clean": targets.damaged_clean_points,
            "unreliable_faulty": targets.unreliable_faulty_points,
        },
        "voxels": {
            "clean_occupied": int(targets.clean_occupancy.sum()),
            "faulty_occupied": int(targets.faulty_occupancy.sum()),
            "preserve": int(targets.preserve_mask.sum()),
            "repair": int(targets.repair_mask.sum()),
            "remove": int(targets.remove_mask.sum()),
            "repair_and_remove": int(both.sum()),
            "changed_union": int(targets.change_mask.sum()),
        },
    }
    atomic_write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved 3D targets: {args.output_root / 'fault_ground_truth_3d.npz'}")
    print(f"Saved visualizations: {args.output_root}")


if __name__ == "__main__":
    main()
