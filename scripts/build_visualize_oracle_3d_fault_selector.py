"""Build and visualize an oracle 3D selector from verified fault targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    OracleFaultSelector3DConfig,
    load_voxelization_config,
    select_oracle_fault_regions_3d,
)


COLORS = {
    "halo": "#2979ff",
    "repair": "#ff9f1c",
    "remove": "#ff00d4",
    "both": "#ffffff",
}
PROJECTIONS = (
    (2, 1, "XY", "x forward [m]", "y lateral [m]"),
    (2, 0, "XZ", "x forward [m]", "z vertical [m]"),
    (1, 0, "YZ", "y lateral [m]", "z vertical [m]"),
)


def _centers(voxelizer: HardVoxelizer, mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask).astype(np.int32, copy=False)
    if not len(coordinates):
        return np.empty((0, 3), dtype=np.float32)
    return voxelizer.voxel_centers(coordinates)


def _sample(points: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.sort(rng.choice(len(points), maximum, replace=False))]


def _limits(grid, name: str):
    if name == "XY":
        return grid.x_range, grid.y_range
    if name == "XZ":
        return grid.x_range, grid.z_range
    return grid.y_range, grid.z_range


def _style(axis, title: str, x_label: str, y_label: str) -> None:
    axis.set_facecolor("black")
    axis.set_title(title, color="white")
    axis.set_xlabel(x_label, color="white")
    axis.set_ylabel(y_label, color="white")
    axis.tick_params(colors="white")
    axis.grid(color="#555555", alpha=0.22, linewidth=0.5)
    for spine in axis.spines.values():
        spine.set_color("#777777")


def save_projections(path: Path, points: dict[str, np.ndarray], grid) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="black")
    for axis, (horizontal, vertical, name, x_label, y_label) in zip(axes, PROJECTIONS):
        _style(axis, f"Oracle selector: {name}", x_label, y_label)
        x_limits, y_limits = _limits(grid, name)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        for category in ("halo", "repair", "remove", "both"):
            values = points[category]
            if len(values):
                axis.scatter(
                    values[:, horizontal],
                    values[:, vertical],
                    s=0.5 if category == "halo" else 4.0,
                    c=COLORS[category],
                    alpha=0.22 if category == "halo" else 0.86,
                    linewidths=0,
                    label=category,
                )
        if name == "YZ":
            legend = axis.legend(loc="upper right", facecolor="black", framealpha=0.75)
            for text in legend.get_texts():
                text.set_color("white")
    figure.suptitle(
        "Blue: context-only halo | Orange: repair | Magenta: remove | White: both",
        color="white",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_3d(path: Path, points: dict[str, np.ndarray], grid) -> None:
    figure = plt.figure(figsize=(12, 8), facecolor="black")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("black")
    for category in ("halo", "repair", "remove", "both"):
        values = points[category]
        if len(values):
            axis.scatter(
                values[:, 0], values[:, 1], values[:, 2],
                s=0.5 if category == "halo" else 5.0,
                c=COLORS[category],
                alpha=0.16 if category == "halo" else 0.85,
                linewidths=0,
                label=category,
            )
    axis.set_xlim(*grid.x_range)
    axis.set_ylim(*grid.y_range)
    axis.set_zlim(*grid.z_range)
    axis.set_xlabel("x forward [m]", color="white")
    axis.set_ylabel("y lateral [m]", color="white")
    axis.set_zlabel("z vertical [m]", color="white")
    axis.tick_params(colors="white")
    axis.set_title("Oracle 3D fault selector", color="white")
    axis.view_init(elev=23, azim=-62)
    legend = axis.legend(loc="upper right", facecolor="black", framealpha=0.75)
    if legend:
        for text in legend.get_texts():
            text.set_color("white")
    figure.tight_layout()
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_height_slices(path: Path, selection, grid, requested_slices: int) -> None:
    active_z = np.flatnonzero(np.any(selection.context_mask, axis=(1, 2)))
    if not len(active_z):
        active_z = np.arange(grid.dimensions_zyx[0])
    selected = np.unique(
        np.round(
            np.linspace(active_z[0], active_z[-1], min(requested_slices, len(active_z)))
        ).astype(int)
    )
    columns = min(4, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4.3 * rows), facecolor="black")
    axes = np.atleast_1d(axes).reshape(-1)
    cmap = ListedColormap(
        ["#000000", COLORS["halo"], COLORS["repair"], COLORS["remove"], COLORS["both"]]
    )
    for axis, z_index in zip(axes, selected):
        labels = np.zeros(selection.operation_mask.shape[1:], dtype=np.uint8)
        labels[selection.context_halo[z_index]] = 1
        labels[selection.repair_core[z_index]] = 2
        labels[selection.remove_core[z_index]] = 3
        labels[selection.repair_core[z_index] & selection.remove_core[z_index]] = 4
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
        "Oracle selector slices — blue context only, orange repair, magenta remove, white both",
        color="white",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--halo-m", type=float, default=1.0)
    parser.add_argument("--grouping-radius-m", type=float, default=0.4)
    parser.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=26)
    parser.add_argument("--min-component-voxels", type=int, default=1)
    parser.add_argument(
        "--min-crop-shape-zyx", type=int, nargs=3, default=(4, 10, 10),
        metavar=("Z", "Y", "X"),
    )
    parser.add_argument("--height-slices", type=int, default=12)
    parser.add_argument("--max-display-voxels", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.ground_truth.is_file():
        raise FileNotFoundError(f"3D ground truth not found: {args.ground_truth}")
    if args.height_slices < 1 or args.max_display_voxels < 1:
        parser.error("visualization limits must be positive")

    with np.load(args.ground_truth, allow_pickle=False) as archive:
        repair_mask = archive["repair_mask"]
        remove_mask = archive["remove_mask"]
    voxelization = load_voxelization_config(args.config)
    selector_config = OracleFaultSelector3DConfig(
        halo_m=args.halo_m,
        grouping_radius_m=args.grouping_radius_m,
        connectivity=args.connectivity,
        min_component_voxels=args.min_component_voxels,
        min_crop_shape_zyx=tuple(args.min_crop_shape_zyx),
    )
    selection = select_oracle_fault_regions_3d(
        repair_mask, remove_mask, voxelization.grid, selector_config
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    voxelizer = HardVoxelizer(voxelization.grid)
    repair_only = selection.repair_core & ~selection.remove_core
    remove_only = selection.remove_core & ~selection.repair_core
    both = selection.repair_core & selection.remove_core
    rng = np.random.default_rng(args.seed)
    points = {
        "halo": _sample(_centers(voxelizer, selection.context_halo), args.max_display_voxels, rng),
        "repair": _sample(_centers(voxelizer, repair_only), args.max_display_voxels, rng),
        "remove": _sample(_centers(voxelizer, remove_only), args.max_display_voxels, rng),
        "both": _sample(_centers(voxelizer, both), args.max_display_voxels, rng),
    }
    save_projections(args.output_root / "oracle_selector_xyz_projections.png", points, voxelization.grid)
    save_3d(args.output_root / "oracle_selector_3d.png", points, voxelization.grid)
    save_height_slices(
        args.output_root / "oracle_selector_height_slices.png",
        selection,
        voxelization.grid,
        args.height_slices,
    )

    component_rows = [asdict(component) for component in selection.components]
    component_arrays = {
        "component_id": np.asarray([row["component_id"] for row in component_rows], dtype=np.int32),
        "operation_voxels": np.asarray([row["operation_voxels"] for row in component_rows], dtype=np.int32),
        "repair_voxels": np.asarray([row["repair_voxels"] for row in component_rows], dtype=np.int32),
        "remove_voxels": np.asarray([row["remove_voxels"] for row in component_rows], dtype=np.int32),
        "core_min_zyx": np.asarray([row["core_min_zyx"] for row in component_rows], dtype=np.int32).reshape(-1, 3),
        "core_max_exclusive_zyx": np.asarray(
            [row["core_max_exclusive_zyx"] for row in component_rows], dtype=np.int32
        ).reshape(-1, 3),
        "crop_min_zyx": np.asarray([row["crop_min_zyx"] for row in component_rows], dtype=np.int32).reshape(-1, 3),
        "crop_max_exclusive_zyx": np.asarray(
            [row["crop_max_exclusive_zyx"] for row in component_rows], dtype=np.int32
        ).reshape(-1, 3),
    }
    atomic_savez(
        args.output_root / "oracle_selector_3d.npz",
        compression_level=1,
        repair_core=selection.repair_core,
        remove_core=selection.remove_core,
        operation_mask=selection.operation_mask,
        context_mask=selection.context_mask,
        context_halo=selection.context_halo,
        grouping_mask=selection.grouping_mask,
        component_labels=selection.component_labels,
        **component_arrays,
    )
    summary = {
        "ground_truth": str(args.ground_truth),
        "selector_type": "oracle_provenance_3d",
        "learned_parameters": 0,
        "coordinate_order": "zyx",
        "config": asdict(selector_config),
        "repair_core_voxels": int(selection.repair_core.sum()),
        "remove_core_voxels": int(selection.remove_core.sum()),
        "operation_voxels": int(selection.operation_mask.sum()),
        "context_halo_voxels": int(selection.context_halo.sum()),
        "context_voxels": int(selection.context_mask.sum()),
        "components": len(selection.components),
        "component_details": component_rows,
    }
    atomic_write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved oracle selector: {args.output_root / 'oracle_selector_3d.npz'}")
    print(f"Saved visualizations: {args.output_root}")


if __name__ == "__main__":
    main()
