"""Save raw-point, occupied-voxel, cube, overlay, and projection diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from voxelization import (
    HardVoxelizer,
    load_voxelization_config,
)
from voxelization.inputs import (
    discover_sample_paths,
    load_aligned_point_inputs,
)


COLORS = {"lidar": "#00d5ff", "radar": "#ff00d4"}


def sample_rows(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    chosen = np.random.default_rng(seed).choice(len(values), maximum, replace=False)
    return values[np.sort(chosen)]


def set_axes(ax, grid, title: str) -> None:
    ax.set_xlim(*grid.x_range)
    ax.set_ylim(*grid.y_range)
    ax.set_zlim(*grid.z_range)
    ax.set_xlabel("x forward [m]")
    ax.set_ylabel("y lateral [m]")
    ax.set_zlabel("z vertical [m]")
    ax.set_title(title)
    ax.view_init(elev=24, azim=-62)


def scatter_3d(ax, xyz: np.ndarray, color: str, label: str, size: float = 0.5):
    if len(xyz):
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=color, label=label)
        ax.legend(loc="upper right")


def cube_faces(center: np.ndarray, size: tuple[float, float, float]):
    half = np.asarray(size, dtype=np.float32) / 2.0
    low, high = center - half, center + half
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = np.asarray(
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
         [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]
    )
    return [[vertices[i] for i in face] for face in (
        (0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(3,0,4,7)
    )]


def add_cubes(ax, centers: np.ndarray, size, color: str) -> None:
    faces = [face for center in centers for face in cube_faces(center, size)]
    if faces:
        collection = Poly3DCollection(
            faces, facecolors=color, edgecolors=color, linewidths=0.15, alpha=0.10
        )
        ax.add_collection3d(collection)


def save_overview(output: Path, grid, raw, centers, max_raw, max_centers, seed):
    figure = plt.figure(figsize=(18, 11))
    panels = (
        (raw["lidar"], None, "Raw clean LiDAR", "lidar"),
        (raw["radar"], None, "Raw aligned radar", "radar"),
        (centers["lidar"], None, "Occupied LiDAR voxel centers", "lidar"),
        (centers["radar"], None, "Occupied radar voxel centers", "radar"),
    )
    for panel, (values, _, title, modality) in enumerate(panels, start=1):
        ax = figure.add_subplot(2, 3, panel, projection="3d")
        maximum = max_raw if panel <= 2 else max_centers
        scatter_3d(ax, sample_rows(values, maximum, seed + panel), COLORS[modality], modality)
        set_axes(ax, grid, title)
    for panel, source, title in (
        (5, raw, "Combined raw points"),
        (6, centers, "Combined occupied voxel centers"),
    ):
        ax = figure.add_subplot(2, 3, panel, projection="3d")
        maximum = max_raw if panel == 5 else max_centers
        for offset, modality in enumerate(("lidar", "radar")):
            scatter_3d(
                ax, sample_rows(source[modality], maximum, seed + panel + offset),
                COLORS[modality], modality,
            )
        set_axes(ax, grid, title)
    figure.suptitle("3D voxelization diagnostics (visualization sampling only)")
    figure.tight_layout()
    figure.savefig(output / "raw_centers_and_overlays.png", dpi=170)
    plt.close(figure)


def save_cubes(output: Path, grid, centers, maximum, seed):
    figure = plt.figure(figsize=(13, 6))
    for panel, modality in enumerate(("lidar", "radar"), start=1):
        ax = figure.add_subplot(1, 2, panel, projection="3d")
        displayed = sample_rows(centers[modality], maximum, seed + panel)
        add_cubes(ax, displayed, grid.voxel_size, COLORS[modality])
        set_axes(
            ax, grid,
            f"{modality.title()} occupied voxel cubes ({len(displayed)}/{len(centers[modality])})",
        )
    figure.tight_layout()
    figure.savefig(output / "occupied_voxel_cubes.png", dpi=170)
    plt.close(figure)


def save_projections(output: Path, grid, raw, centers, maximum, seed):
    figure, axes = plt.subplots(4, 3, figsize=(15, 16))
    rows = (
        ("LiDAR raw", raw["lidar"], "lidar"),
        ("LiDAR voxel centers", centers["lidar"], "lidar"),
        ("Radar raw", raw["radar"], "radar"),
        ("Radar voxel centers", centers["radar"], "radar"),
    )
    projections = (
        (0, 1, "XY", grid.x_range, grid.y_range),
        (0, 2, "XZ", grid.x_range, grid.z_range),
        (1, 2, "YZ", grid.y_range, grid.z_range),
    )
    for row_index, (row_name, values, modality) in enumerate(rows):
        values = sample_rows(values, maximum, seed + row_index)
        for column, (horizontal, vertical, name, x_limits, y_limits) in enumerate(projections):
            ax = axes[row_index, column]
            if len(values):
                ax.scatter(values[:, horizontal], values[:, vertical], s=0.5, c=COLORS[modality])
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.set_title(f"{row_name}: {name}")
            ax.set_aspect("equal" if name == "XY" else "auto")
            ax.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output / "orthographic_projections.png", dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/voxelization_3d.json"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar-source", choices=("clean", "faulty"), default="clean")
    parser.add_argument("--max-raw-points", type=int, default=50000)
    parser.add_argument("--max-centers", type=int, default=20000)
    parser.add_argument("--max-cubes", type=int, default=2000)
    args = parser.parse_args()

    config = load_voxelization_config(args.config)
    paths = discover_sample_paths(args.data_root, args.split, unique_frames=True)
    if not paths:
        raise FileNotFoundError(f"No samples found in {args.data_root / args.split}")
    if not 0 <= args.sample_index < len(paths):
        raise IndexError(f"sample-index must be in [0, {len(paths) - 1}]")
    inputs = load_aligned_point_inputs(
        paths[args.sample_index], args.radar_root, lidar_source=args.lidar_source
    )
    voxelizers = {
        "lidar": HardVoxelizer(config.grid, max_points_per_voxel=config.lidar.max_points_per_voxel),
        "radar": HardVoxelizer(config.grid, max_points_per_voxel=config.radar.max_points_per_voxel),
    }
    raw = {"lidar": inputs.lidar_points[:, :3], "radar": inputs.radar_points[:, :3]}
    results = {
        "lidar": voxelizers["lidar"].voxelize(inputs.lidar_points, inputs.lidar_feature_names),
        "radar": voxelizers["radar"].voxelize(inputs.radar_points, inputs.radar_feature_names),
    }
    centers = {
        modality: voxelizers[modality].voxel_centers(result.voxel_coords)
        for modality, result in results.items()
    }
    output = args.output_root / paths[args.sample_index].stem
    output.mkdir(parents=True, exist_ok=True)
    save_overview(output, config.grid, raw, centers, args.max_raw_points, args.max_centers, args.seed)
    save_cubes(output, config.grid, centers, args.max_cubes, args.seed)
    save_projections(output, config.grid, raw, centers, args.max_raw_points, args.seed)
    print(f"Sample: {paths[args.sample_index]}")
    print(f"LiDAR points/voxels: {len(raw['lidar'])}/{len(centers['lidar'])}")
    print(f"Radar points/voxels: {len(raw['radar'])}/{len(centers['radar'])}")
    print(f"Saved diagnostics: {output}")


if __name__ == "__main__":
    main()
