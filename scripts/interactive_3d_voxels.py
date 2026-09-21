"""Interactively rotate and inspect aligned LiDAR/radar 3D point clouds.

Controls
--------
Mouse drag: rotate (Matplotlib's native 3D controls)
Mouse wheel: zoom
Arrow keys: rotate in fixed increments
1/2/3: LiDAR only / radar only / both
R: raw points
V: occupied voxel centers
X: raw points and voxel centers
C: toggle occupied voxel cube wireframes
0: reset camera
S: save the current view
Q or Escape: close
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from voxelization import HardVoxelizer, load_voxelization_config
from voxelization.inputs import discover_sample_paths, load_aligned_point_inputs


COLORS = {"lidar": "#00bfe8", "radar": "#ff00d4"}
DEFAULT_ELEVATION = 24.0
DEFAULT_AZIMUTH = -62.0


def _sample_rows(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Deterministically limit rendering cost without changing voxelization."""
    if len(values) <= maximum:
        return values
    indices = np.random.default_rng(seed).choice(
        len(values), size=maximum, replace=False
    )
    return values[np.sort(indices)]


def _cube_faces(center: np.ndarray, size: tuple[float, float, float]):
    half = np.asarray(size, dtype=np.float32) / 2.0
    low, high = center - half, center + half
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = np.asarray(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    return [
        [vertices[index] for index in face]
        for face in (
            (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
            (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7),
        )
    ]


class InteractiveVoxelViewer:
    def __init__(
        self,
        raw: dict[str, np.ndarray],
        centers: dict[str, np.ndarray],
        grid,
        output_root: Path,
        *,
        sample_name: str,
        max_raw_points: int,
        max_centers: int,
        max_cubes: int,
        seed: int,
    ) -> None:
        self.raw = {
            modality: _sample_rows(values, max_raw_points, seed + index)
            for index, (modality, values) in enumerate(raw.items())
        }
        self.centers = {
            modality: _sample_rows(values, max_centers, seed + 10 + index)
            for index, (modality, values) in enumerate(centers.items())
        }
        self.cube_centers = {
            modality: _sample_rows(values, max_cubes, seed + 20 + index)
            for index, (modality, values) in enumerate(centers.items())
        }
        self.grid = grid
        self.output_root = output_root
        self.sample_name = sample_name
        self.sensor_mode = "both"
        self.geometry_mode = "raw"
        self.show_cubes = False
        self.elevation = DEFAULT_ELEVATION
        self.azimuth = DEFAULT_AZIMUTH
        self.saved_views = 0
        self.figure = plt.figure(figsize=(13, 9))
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw()

    def _visible_modalities(self) -> tuple[str, ...]:
        if self.sensor_mode == "both":
            return "lidar", "radar"
        return (self.sensor_mode,)

    def draw(self) -> None:
        # Preserve camera changes made by mouse dragging before a mode toggle.
        if hasattr(self.axis, "elev"):
            self.elevation = float(self.axis.elev)
            self.azimuth = float(self.axis.azim)
        self.axis.clear()
        modalities = self._visible_modalities()
        if self.geometry_mode in {"raw", "overlay"}:
            for modality in modalities:
                points = self.raw[modality]
                if len(points):
                    self.axis.scatter(
                        points[:, 0], points[:, 1], points[:, 2],
                        s=0.7 if modality == "lidar" else 4.0,
                        c=COLORS[modality], alpha=0.70,
                        label=f"{modality} raw ({len(points):,} shown)",
                    )
        if self.geometry_mode in {"voxels", "overlay"}:
            for modality in modalities:
                points = self.centers[modality]
                if len(points):
                    self.axis.scatter(
                        points[:, 0], points[:, 1], points[:, 2],
                        s=2.0 if modality == "lidar" else 9.0,
                        c=COLORS[modality], marker="s", alpha=0.85,
                        label=f"{modality} voxel centers ({len(points):,} shown)",
                    )
        if self.show_cubes:
            for modality in modalities:
                faces = [
                    face
                    for center in self.cube_centers[modality]
                    for face in _cube_faces(center, self.grid.voxel_size)
                ]
                if faces:
                    self.axis.add_collection3d(
                        Poly3DCollection(
                            faces,
                            facecolors="none",
                            edgecolors=COLORS[modality],
                            linewidths=0.20,
                            alpha=0.20,
                        )
                    )
        self.axis.set_xlim(*self.grid.x_range)
        self.axis.set_ylim(*self.grid.y_range)
        self.axis.set_zlim(*self.grid.z_range)
        self.axis.set_xlabel("x forward [m]")
        self.axis.set_ylabel("y lateral [m]")
        self.axis.set_zlabel("z vertical [m]")
        self.axis.view_init(elev=self.elevation, azim=self.azimuth)
        self.axis.set_title(
            f"{self.sample_name} | sensors={self.sensor_mode} | "
            f"geometry={self.geometry_mode} | cubes={'on' if self.show_cubes else 'off'}\n"
            "Drag=rotate, wheel=zoom, arrows=rotate, 1/2/3=sensors, "
            "R/V/X=geometry, C=cubes, S=save, 0=reset, Q=quit"
        )
        handles, _ = self.axis.get_legend_handles_labels()
        if handles:
            self.axis.legend(loc="upper right")
        self.figure.canvas.draw_idle()

    def _save(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.saved_views += 1
        path = self.output_root / (
            f"{self.sample_name}_{self.geometry_mode}_{self.sensor_mode}_"
            f"elev{self.axis.elev:.1f}_azim{self.axis.azim:.1f}_"
            f"{self.saved_views:02d}.png"
        )
        self.figure.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved current view: {path}", flush=True)
        return path

    def _on_key(self, event) -> None:
        key = str(event.key).lower()
        self.elevation = float(self.axis.elev)
        self.azimuth = float(self.axis.azim)
        if key == "1":
            self.sensor_mode = "lidar"
        elif key == "2":
            self.sensor_mode = "radar"
        elif key == "3":
            self.sensor_mode = "both"
        elif key == "r":
            self.geometry_mode = "raw"
        elif key == "v":
            self.geometry_mode = "voxels"
        elif key == "x":
            self.geometry_mode = "overlay"
        elif key == "c":
            self.show_cubes = not self.show_cubes
        elif key == "left":
            self.azimuth -= 5.0
        elif key == "right":
            self.azimuth += 5.0
        elif key == "up":
            self.elevation = min(90.0, self.elevation + 5.0)
        elif key == "down":
            self.elevation = max(-90.0, self.elevation - 5.0)
        elif key == "0":
            self.elevation, self.azimuth = DEFAULT_ELEVATION, DEFAULT_AZIMUTH
        elif key == "s":
            self._save()
            return
        elif key in {"q", "escape"}:
            plt.close(self.figure)
            return
        else:
            return
        self.axis.view_init(elev=self.elevation, azim=self.azimuth)
        self.draw()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/voxelization_3d.json")
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar-source", choices=("clean", "faulty"), default="clean")
    parser.add_argument("--max-raw-points", type=int, default=100000)
    parser.add_argument("--max-centers", type=int, default=30000)
    parser.add_argument("--max-cubes", type=int, default=1500)
    parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="Save the initial view to this path before opening the viewer",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save/validate without opening a GUI (requires --save-snapshot)",
    )
    args = parser.parse_args()
    if args.no_show and args.save_snapshot is None:
        parser.error("--no-show requires --save-snapshot")

    config = load_voxelization_config(args.config)
    paths = discover_sample_paths(args.data_root, args.split, unique_frames=True)
    if not paths:
        raise FileNotFoundError(f"No samples found in {args.data_root / args.split}")
    if not 0 <= args.sample_index < len(paths):
        raise IndexError(f"sample-index must be in [0, {len(paths) - 1}]")
    sample_path = paths[args.sample_index]
    inputs = load_aligned_point_inputs(
        sample_path, args.radar_root, lidar_source=args.lidar_source
    )
    voxelizers = {
        "lidar": HardVoxelizer(
            config.grid, max_points_per_voxel=config.lidar.max_points_per_voxel
        ),
        "radar": HardVoxelizer(
            config.grid, max_points_per_voxel=config.radar.max_points_per_voxel
        ),
    }
    raw = {
        "lidar": inputs.lidar_points[:, :3],
        "radar": inputs.radar_points[:, :3],
    }
    lidar_result = voxelizers["lidar"].voxelize(
        inputs.lidar_points, inputs.lidar_feature_names
    )
    radar_result = voxelizers["radar"].voxelize(
        inputs.radar_points, inputs.radar_feature_names
    )
    centers = {
        "lidar": voxelizers["lidar"].voxel_centers(lidar_result.voxel_coords),
        "radar": voxelizers["radar"].voxel_centers(radar_result.voxel_coords),
    }
    viewer = InteractiveVoxelViewer(
        raw,
        centers,
        config.grid,
        args.output_root,
        sample_name=sample_path.stem,
        max_raw_points=args.max_raw_points,
        max_centers=args.max_centers,
        max_cubes=args.max_cubes,
        seed=args.seed,
    )
    print(f"Loaded: {sample_path}")
    print(f"LiDAR: {len(raw['lidar']):,} points, {len(centers['lidar']):,} voxels")
    print(f"Radar: {len(raw['radar']):,} points, {len(centers['radar']):,} voxels")
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
