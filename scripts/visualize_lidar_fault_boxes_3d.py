"""Visualize faulty LiDAR, removed clean points, and 3D selector boxes.

No radar data is loaded. The base cloud is the particle-filtered faulty LiDAR;
points present in clean LiDAR but absent after fault injection are overlaid in
a contrasting color. Selector crop or tight-core boxes are drawn as wireframes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from Fault_Localization_Model.io_utils import atomic_write_json
from voxelization import load_voxelization_config


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _save_figure(figure, destination: Path, *, dpi: int = 210) -> None:
    """Atomically save a figure, retrying transient Windows file locks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.writing{destination.suffix}"
    )
    for attempt in range(3):
        try:
            figure.savefig(temporary, dpi=dpi, facecolor=figure.get_facecolor())
            temporary.replace(destination)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))


def _sample(points: np.ndarray, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= maximum:
        return points
    return points[np.sort(rng.choice(len(points), maximum, replace=False))]


def _in_grid(points: np.ndarray, grid) -> np.ndarray:
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


def _removed_clean_points(
    clean_points: np.ndarray,
    faulty_source_ids: np.ndarray,
) -> np.ndarray:
    source_ids = np.asarray(faulty_source_ids, dtype=np.int64)
    valid_sources = source_ids[(source_ids >= 0) & (source_ids < len(clean_points))]
    present = np.zeros(len(clean_points), dtype=bool)
    present[valid_sources] = True
    return clean_points[~present]


def _box_vertices(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    x0, y0, z0 = min_xyz
    x1, y1, z1 = max_xyz
    return np.asarray(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def _voxel_boxes_to_metric(
    minimum_zyx: np.ndarray,
    maximum_zyx: np.ndarray,
    grid,
) -> list[tuple[np.ndarray, np.ndarray]]:
    mins = np.asarray(grid.mins_xyz, dtype=np.float64)
    sizes = np.asarray(grid.voxel_size, dtype=np.float64)
    boxes = []
    for lower_zyx, upper_zyx in zip(minimum_zyx, maximum_zyx):
        lower_xyz = mins + np.asarray(lower_zyx[::-1], dtype=np.float64) * sizes
        upper_xyz = mins + np.asarray(upper_zyx[::-1], dtype=np.float64) * sizes
        boxes.append((lower_xyz.astype(np.float32), upper_xyz.astype(np.float32)))
    return boxes


def _merge_boxes_to_limit(
    minimum_zyx: np.ndarray,
    maximum_zyx: np.ndarray,
    maximum_boxes: int,
    voxel_size_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily merge boxes with the smallest added physical volume.

    Every source box remains fully enclosed. No component is discarded.
    Coordinates remain integer ``zyx`` voxel boundaries.
    """

    minimum = np.asarray(minimum_zyx, dtype=np.int32)
    maximum = np.asarray(maximum_zyx, dtype=np.int32)
    if minimum.shape != maximum.shape or minimum.ndim != 2 or minimum.shape[1] != 3:
        raise ValueError("box minima/maxima must both have shape [N, 3]")
    if maximum_boxes < 1:
        raise ValueError("maximum_boxes must be positive")
    if len(minimum) <= maximum_boxes:
        return minimum.copy(), maximum.copy()
    spacing_zyx = np.asarray(voxel_size_xyz[::-1], dtype=np.float64)
    boxes = [(lower.copy(), upper.copy()) for lower, upper in zip(minimum, maximum)]

    def volume(lower: np.ndarray, upper: np.ndarray) -> float:
        return float(np.prod((upper - lower) * spacing_zyx))

    def merge_overlaps() -> bool:
        """Merge one positive-volume intersection and report whether it changed."""
        for first in range(len(boxes) - 1):
            for second in range(first + 1, len(boxes)):
                intersection_lower = np.maximum(boxes[first][0], boxes[second][0])
                intersection_upper = np.minimum(boxes[first][1], boxes[second][1])
                if np.all(intersection_lower < intersection_upper):
                    boxes[first] = (
                        np.minimum(boxes[first][0], boxes[second][0]),
                        np.maximum(boxes[first][1], boxes[second][1]),
                    )
                    boxes.pop(second)
                    return True
        return False

    while merge_overlaps():
        pass

    while len(boxes) > maximum_boxes:
        best = None
        for first in range(len(boxes) - 1):
            for second in range(first + 1, len(boxes)):
                lower = np.minimum(boxes[first][0], boxes[second][0])
                upper = np.maximum(boxes[first][1], boxes[second][1])
                added_volume = (
                    volume(lower, upper)
                    - volume(*boxes[first])
                    - volume(*boxes[second])
                )
                candidate = (added_volume, first, second, lower, upper)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        _, first, second, lower, upper = best
        boxes[first] = (lower, upper)
        boxes.pop(second)
        while merge_overlaps():
            pass
    boxes.sort(key=lambda box: tuple(int(value) for value in box[0]))
    return (
        np.stack([box[0] for box in boxes]).astype(np.int32),
        np.stack([box[1] for box in boxes]).astype(np.int32),
    )


def _draw_box(axis, minimum: np.ndarray, maximum: np.ndarray, color: str) -> None:
    vertices = _box_vertices(minimum, maximum)
    for start, end in BOX_EDGES:
        axis.plot(
            vertices[[start, end], 0],
            vertices[[start, end], 1],
            vertices[[start, end], 2],
            color=color,
            linewidth=1.25,
            alpha=0.92,
        )


def _draw_projected_box(
    axis,
    minimum: np.ndarray,
    maximum: np.ndarray,
    horizontal: int,
    vertical: int,
    color: str,
) -> None:
    x0, y0 = minimum[[horizontal, vertical]]
    x1, y1 = maximum[[horizontal, vertical]]
    axis.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        color=color,
        linewidth=1.4,
        alpha=0.95,
    )


def _save_xyz_projections(
    destination: Path,
    faulty: np.ndarray,
    removed: np.ndarray,
    boxes: list[tuple[np.ndarray, np.ndarray]],
    grid,
    title: str,
) -> None:
    projections = (
        (0, 1, "X-Y (bird's-eye)", "x forward [m]", "y lateral [m]", grid.x_range, grid.y_range),
        (1, 2, "Y-Z (rear view)", "y lateral [m]", "z vertical [m]", grid.y_range, grid.z_range),
        (0, 2, "X-Z (side view)", "x forward [m]", "z vertical [m]", grid.x_range, grid.z_range),
    )
    figure, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="black")
    for axis, (horizontal, vertical, name, x_label, y_label, x_limits, y_limits) in zip(
        axes, projections
    ):
        axis.set_facecolor("black")
        if len(faulty):
            axis.scatter(
                faulty[:, horizontal], faulty[:, vertical],
                s=0.55, c="#00d5ff", alpha=0.58, linewidths=0,
            )
        if len(removed):
            axis.scatter(
                removed[:, horizontal], removed[:, vertical],
                s=2.5, c="#ff3b30", alpha=0.95, linewidths=0,
            )
        for minimum, maximum in boxes:
            _draw_projected_box(
                axis, minimum, maximum, horizontal, vertical, "#ffe600"
            )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel(x_label, color="white")
        axis.set_ylabel(y_label, color="white")
        axis.set_title(name, color="white")
        axis.tick_params(colors="white")
        axis.grid(color="#555555", linewidth=0.35, alpha=0.4)
        # Preserve metric aspect in BEV. Stretch the vertical cross-sections
        # to use their panels; their axes remain labelled in metres.
        axis.set_aspect("equal" if (horizontal, vertical) == (0, 1) else "auto",
                        adjustable="box")
    legend = axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#00d5ff",
                   markeredgecolor="none", label="faulty LiDAR retained"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#ff3b30",
                   markeredgecolor="none", label="LiDAR points removed by fault"),
            Line2D([0], [0], color="#ffe600", linewidth=2,
                   label="projected 3D reconstruction box"),
        ],
        loc="upper right",
        facecolor="black",
        framealpha=0.78,
    )
    for text in legend.get_texts():
        text.set_color("white")
    figure.suptitle(title, color="white")
    figure.tight_layout()
    _save_figure(figure, destination)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-artifact", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--box-type", choices=("crop", "core"), default="crop")
    parser.add_argument("--max-lidar-points", type=int, default=100000)
    parser.add_argument("--max-removed-points", type=int, default=100000)
    parser.add_argument(
        "--max-boxes", type=int, choices=(1, 2), default=2,
        help="Merge every fault component into at most this many enclosing boxes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-62.0)
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive Matplotlib window after saving")
    args = parser.parse_args()
    if args.max_lidar_points < 1 or args.max_removed_points < 1:
        parser.error("point limits must be positive")
    if not args.raw_artifact.is_file():
        raise FileNotFoundError(f"Raw fault artifact not found: {args.raw_artifact}")
    if not args.selector.is_file():
        raise FileNotFoundError(f"Selector artifact not found: {args.selector}")

    with np.load(args.raw_artifact, allow_pickle=False) as raw:
        clean = raw["clean_points"]
        faulty = raw["faulty_points"]
        source_ids = raw["faulty_source_ids"]
        metadata = json.loads(str(raw["metadata_json"].item()))
    with np.load(args.selector, allow_pickle=False) as selector:
        prefix = "crop" if args.box_type == "crop" else "core"
        box_min = selector[f"{prefix}_min_zyx"]
        box_max = selector[f"{prefix}_max_exclusive_zyx"]

    config = load_voxelization_config(args.config)
    grid = config.grid
    removed = _removed_clean_points(clean, source_ids)
    faulty = faulty[_in_grid(faulty, grid)]
    removed = removed[_in_grid(removed, grid)]
    rng = np.random.default_rng(args.seed)
    displayed_faulty = _sample(faulty, args.max_lidar_points, rng)
    displayed_removed = _sample(removed, args.max_removed_points, rng)

    component_box_count = len(box_min)
    box_min, box_max = _merge_boxes_to_limit(
        box_min,
        box_max,
        args.max_boxes,
        grid.voxel_size,
    )
    boxes = _voxel_boxes_to_metric(box_min, box_max, grid)

    figure = plt.figure(figsize=(14, 10), facecolor="black")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("black")
    if len(displayed_faulty):
        axis.scatter(
            displayed_faulty[:, 0], displayed_faulty[:, 1], displayed_faulty[:, 2],
            s=0.55, c="#00d5ff", alpha=0.58, linewidths=0,
        )
    if len(displayed_removed):
        axis.scatter(
            displayed_removed[:, 0], displayed_removed[:, 1], displayed_removed[:, 2],
            s=2.5, c="#ff3b30", alpha=0.95, linewidths=0,
        )
    for minimum, maximum in boxes:
        _draw_box(axis, minimum, maximum, "#ffe600")

    axis.set_xlim(*grid.x_range)
    axis.set_ylim(*grid.y_range)
    axis.set_zlim(*grid.z_range)
    axis.set_xlabel("x forward [m]", color="white")
    axis.set_ylabel("y lateral [m]", color="white")
    axis.set_zlabel("z vertical [m]", color="white")
    axis.tick_params(colors="white")
    axis.view_init(elev=args.elevation, azim=args.azimuth)
    fault = metadata.get("fault", "unknown")
    severity = metadata.get("severity", "unknown")
    frame = metadata.get("frame_id", metadata.get("lidar_path", "frame"))
    axis.set_title(
        f"{frame} | {fault} severity {severity} | {args.box_type} 3D boxes",
        color="white",
    )
    legend = axis.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#00d5ff",
                   markeredgecolor="none", label="faulty LiDAR retained"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#ff3b30",
                   markeredgecolor="none", label="LiDAR points removed by fault"),
            Line2D([0], [0], color="#ffe600", linewidth=2,
                   label=f"3D {args.box_type} bounding box"),
        ],
        loc="upper right",
        facecolor="black",
        framealpha=0.78,
    )
    for text in legend.get_texts():
        text.set_color("white")
    figure.tight_layout()
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / f"lidar_removed_points_{args.box_type}_boxes_3d.png"
    _save_figure(figure, destination)
    projection_destination = (
        args.output_root
        / f"lidar_removed_points_{args.box_type}_boxes_xyz_projections.png"
    )
    _save_xyz_projections(
        projection_destination,
        displayed_faulty,
        displayed_removed,
        boxes,
        grid,
        f"{frame} | {fault} severity {severity} | {args.box_type} box projections",
    )
    summary = {
        "raw_artifact": str(args.raw_artifact),
        "selector": str(args.selector),
        "radar_loaded": False,
        "box_type": args.box_type,
        "component_boxes_before_merge": int(component_box_count),
        "boxes_displayed": int(len(boxes)),
        "merged_box_min_zyx": box_min.tolist(),
        "merged_box_max_exclusive_zyx": box_max.tolist(),
        "faulty_lidar_points_in_grid": int(len(faulty)),
        "removed_clean_lidar_points_in_grid": int(len(removed)),
        "faulty_lidar_points_displayed": int(len(displayed_faulty)),
        "removed_clean_lidar_points_displayed": int(len(displayed_removed)),
        "projection_image": str(projection_destination),
    }
    atomic_write_json(args.output_root / "lidar_fault_boxes_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved 3D LiDAR/box view: {destination}")
    print(f"Saved XY/YZ/XZ projections: {projection_destination}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
