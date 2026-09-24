"""Export full-scan inputs and visualize clean/faulty LiDAR in angular range space.

Without --geometry, the image uses continuous elevation bins for a diagnostic
preview, not a claimed LiDAR beam assignment. With a calibrated geometry JSON,
it displays the exact model range-image cells. The NPZ keeps every point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.two_stage_reconstruction_head.voxelization.inputs import (
    load_aligned_point_inputs, load_clean_lidar_from_metadata,
)
from models.two_stage_reconstruction_head.range_view.geometry import (
    RangeGeometry, angular_indices, project_lidar,
)
from models.two_stage_reconstruction_head.range_view.radar import project_aligned_radar
from Fault_Localization_Model.vod_dataset.vod_io import load_vod_radar


def _display_indices(count: int, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.sort(rng.choice(count, maximum, replace=False))


def _angles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = points[:, :3]
    radius = np.linalg.norm(xyz, axis=1)
    azimuth = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    elevation = np.degrees(np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1)))
    return azimuth, elevation, radius


def _preview_range(points: np.ndarray, azimuth_edges: np.ndarray,
                   elevation_edges: np.ndarray) -> np.ndarray:
    azimuth, elevation, radius = _angles(points)
    width, height = len(azimuth_edges) - 1, len(elevation_edges) - 1
    col = np.searchsorted(azimuth_edges, azimuth, side="right") - 1
    row = np.searchsorted(elevation_edges, elevation, side="right") - 1
    valid = (np.isfinite(radius) & (radius > 0) & (row >= 0) & (row < height)
             & (col >= 0) & (col < width))
    nearest = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(nearest, row[valid] * width + col[valid], radius[valid])
    image = nearest.reshape(height, width)
    image[~np.isfinite(image)] = np.nan
    return image


def _plot_range(figure, axes, clean: np.ndarray, faulty: np.ndarray,
                radar: np.ndarray, *, include_rear: bool,
                geometry: RangeGeometry | None, azimuth_bins: int,
                elevation_bins: int, max_display_radar: int, seed: int,
                separate_radar: bool = False) -> str:
    if geometry is None:
        # Only a visual angular raster. The model never trains on these
        # arbitrary elevation bins; calibrated beam rows require --geometry.
        _, clean_elevation, _ = _angles(clean)
        _, faulty_elevation, _ = _angles(faulty)
        observed = np.r_[clean_elevation, faulty_elevation]
        if separate_radar:
            _, radar_elevation, _ = _angles(radar)
            observed = np.r_[observed, radar_elevation]
        observed = observed[np.isfinite(observed)]
        if not len(observed):
            raise ValueError("No finite LiDAR elevations to display")
        low, high = float(observed.min()), float(observed.max())
        if low == high:
            low -= 0.5
            high += 0.5
        azimuth_edges = np.linspace(-180 if include_rear else -90,
                                    180 if include_rear else 90, azimuth_bins + 1)
        elevation_edges = np.linspace(low, high, elevation_bins + 1)
        images = [_preview_range(clean, azimuth_edges, elevation_edges),
                  _preview_range(faulty, azimuth_edges, elevation_edges)]
        if separate_radar:
            images.append(_preview_range(radar, azimuth_edges, elevation_edges))
        extent = (azimuth_edges[0], azimuth_edges[-1], low, high)
        radar_x, radar_y, _ = _angles(radar)
        xlabel, ylabel = "azimuth (degrees)", "elevation (degrees)"
        mode = "continuous-angle preview; NOT calibrated beam rows"
    else:
        images = [np.where(projection.valid, projection.range_m, np.nan)
                  for projection in (project_lidar(clean, geometry),
                                     project_lidar(faulty, geometry))]
        if separate_radar:
            radar_channels = project_aligned_radar(radar, geometry)
            images.append(np.where(radar_channels[0] > 0,
                                   radar_channels[2] * geometry.max_range_m, np.nan))
        extent = (-0.5, geometry.azimuth_bins - 0.5,
                  -0.5, len(geometry.beam_elevations_rad) - 0.5)
        row, col, _, valid = angular_indices(radar, geometry, require_beam_match=False)
        radar_x, radar_y = col[valid], row[valid]
        xlabel, ylabel = "azimuth bin", "calibrated LiDAR beam row"
        mode = "calibrated model range-image cells"
    finite_ranges = np.concatenate(tuple(image[np.isfinite(image)] for image in images))
    shared_max = float(np.percentile(finite_ranges, 99)) if len(finite_ranges) else 1.0
    shared_max = max(shared_max, 1.0)
    titles = (("Clean LiDAR range", "Faulty LiDAR range", "Stacked radar range")
              if separate_radar else ("Clean LiDAR range", "Faulty LiDAR range + radar"))
    for axis, image, title in zip(axes, images, titles):
        plotted = axis.imshow(image, origin="lower", interpolation="nearest",
                              aspect="auto", extent=extent, cmap="viridis",
                              vmin=0, vmax=shared_max)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        figure.colorbar(plotted, ax=axis, label="range (m)", fraction=0.046)
    if not separate_radar and len(radar_x):
        selected = _display_indices(len(radar_x), max_display_radar,
                                    np.random.default_rng(seed))
        axes[1].scatter(radar_x[selected], radar_y[selected], s=7,
                        facecolors="none", edgecolors="#f05a28", linewidths=0.8,
                        label="radar angular evidence")
        axes[1].legend(loc="upper right")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True,
                        help="Full-scan range-view fault artifact")
    parser.add_argument("--radar-root", type=Path, required=True,
                        help="Existing aligned VoD or HeRCULES radar point cache")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-display-lidar", type=int, default=40000)
    parser.add_argument("--max-display-radar", type=int, default=10000)
    parser.add_argument("--view", choices=("range", "range-separate", "xyz"), default="range")
    parser.add_argument("--require-radar-frames", type=int,
                        help="Verify the radar source contains this many distinct scan time indices")
    parser.add_argument("--geometry", type=Path,
                        help="Measured sensor geometry JSON for exact model beam rows")
    parser.add_argument("--azimuth-bins", type=int, default=1024,
                        help="Diagnostic preview only when --geometry is absent")
    parser.add_argument("--elevation-bins", type=int, default=256,
                        help="Diagnostic preview only when --geometry is absent")
    parser.add_argument("--include-rear", action="store_true",
                        help="Show all azimuths; default matches the forward-only model input (x >= 0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_display_lidar < 1 or args.max_display_radar < 1:
        parser.error("display limits must be positive")
    if args.azimuth_bins < 2 or args.elevation_bins < 2:
        parser.error("angular preview bins must be at least 2")
    aligned = load_aligned_point_inputs(args.sample, args.radar_root, lidar_source="faulty")
    if not aligned.metadata.get("range_view_full_scan", False):
        parser.error("sample is a legacy cropped artifact; regenerate it with create_range_view_fault_samples")
    with np.load(aligned.radar_path, allow_pickle=False) as radar_archive:
        radar_metadata = json.loads(str(radar_archive["metadata_json"].item()))
    raw_radar_path = Path(str(radar_metadata.get("radar_source", "")))
    radar_time_indices = None
    if raw_radar_path.is_file():
        raw_radar = load_vod_radar(raw_radar_path)
        radar_time_indices = np.unique(raw_radar[:, 6])
    elif radar_metadata.get("dataset") == "HeRCULES":
        alignment_rows = radar_metadata.get("hercules_alignment", {}).get("alignment_rows", [])
        radar_time_indices = np.asarray([row["timestamp_ns"] for row in alignment_rows],
                                        dtype=np.int64)
    if args.require_radar_frames is not None:
        if radar_time_indices is None or len(radar_time_indices) != args.require_radar_frames:
            raise ValueError(
                f"Expected {args.require_radar_frames} radar scan time indices, "
                f"found {None if radar_time_indices is None else len(radar_time_indices)} "
                f"in {raw_radar_path}"
            )
    lidar, radar = aligned.lidar_points, aligned.radar_points
    clean = load_clean_lidar_from_metadata(aligned.metadata)
    if not args.include_rear:
        lidar = lidar[lidar[:, 0] >= 0]
        radar = radar[radar[:, 0] >= 0]
        clean = clean[clean[:, 0] >= 0]
    rng = np.random.default_rng(args.seed)
    shown_clean = clean[_display_indices(len(clean), args.max_display_lidar, rng)]
    shown_lidar = lidar[_display_indices(len(lidar), args.max_display_lidar, rng)]
    shown_radar = radar[_display_indices(len(radar), args.max_display_radar, rng)]
    args.output_root.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_root / f"{args.sample.stem}_lidar_radar_inputs.npz"
    np.savez_compressed(
        archive_path,
        clean_lidar_points=clean,
        faulty_lidar_points=lidar,
        radar_points_lidar_frame=radar,
        combined_xyz=np.concatenate((lidar[:, :3], radar[:, :3]), axis=0),
        combined_is_radar=np.r_[np.zeros(len(lidar), dtype=bool), np.ones(len(radar), dtype=bool)],
    )
    if args.view in {"range", "range-separate"}:
        separate = args.view == "range-separate"
        figure, axes = plt.subplots(1, 3 if separate else 2,
                                   figsize=(21 if separate else 18, 7), constrained_layout=True)
        mode = _plot_range(figure, axes, clean, lidar, radar,
                           include_rear=args.include_rear,
                           geometry=RangeGeometry.from_json(args.geometry) if args.geometry else None,
                           azimuth_bins=args.azimuth_bins,
                           elevation_bins=args.elevation_bins,
                           max_display_radar=args.max_display_radar, seed=args.seed,
                           separate_radar=separate)
    else:
        figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
        for row, (vertical, name) in enumerate(((1, "XY"), (2, "XZ"))):
            clean_axis, overlay_axis = axes[row]
            if len(shown_clean):
                clean_axis.scatter(shown_clean[:, 0], shown_clean[:, vertical],
                                   s=0.25, c="#279b54", label="clean LiDAR")
            if len(shown_lidar):
                overlay_axis.scatter(shown_lidar[:, 0], shown_lidar[:, vertical],
                                     s=0.25, c="#1685cc", label="faulty LiDAR")
            if len(shown_radar):
                overlay_axis.scatter(shown_radar[:, 0], shown_radar[:, vertical],
                                     s=3, c="#ea7600", label="radar")
            clean_axis.set_title(f"Clean LiDAR ({name})")
            overlay_axis.set_title(f"Faulty LiDAR + radar ({name})")
            for axis in (clean_axis, overlay_axis):
                axis.set_xlabel("x forward (m)")
                axis.set_ylabel(f"{'y lateral' if vertical == 1 else 'z vertical'} (m)")
                axis.grid(alpha=0.2)
                axis.legend(loc="upper right")
            clean_axis.sharex(overlay_axis)
            clean_axis.sharey(overlay_axis)
            if name == "XY":
                clean_axis.set_aspect("equal", adjustable="box")
                overlay_axis.set_aspect("equal", adjustable="box")
        mode = "XYZ orthographic projections"
    figure.suptitle(f"{args.sample.name} | {mode} | clean {len(clean):,} | "
                    f"faulty {len(lidar):,} | radar {len(radar):,} | "
                    f"radar scans {None if radar_time_indices is None else len(radar_time_indices)}")
    figure_path = args.output_root / f"{args.sample.stem}_lidar_radar_{args.view}.png"
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    print(json.dumps({
        "sample": str(args.sample), "radar_cache": str(aligned.radar_path),
        "forward_only": not args.include_rear,
        "clean_lidar_points": len(clean), "faulty_lidar_points": len(lidar),
        "radar_points": len(radar),
        "radar_variant": radar_metadata.get("radar_variant"),
        "radar_scan_time_indices": None if radar_time_indices is None else radar_time_indices.tolist(),
        "visualization_mode": mode,
        "point_cloud": str(archive_path), "figure": str(figure_path),
    }, indent=2))


if __name__ == "__main__":
    main()
