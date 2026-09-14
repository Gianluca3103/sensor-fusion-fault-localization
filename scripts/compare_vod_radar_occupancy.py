"""Render unfiltered and temporally filtered VoD radar occupancy side by side."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Fault_Localization_Model.bev_utils import metric_to_grid
from Fault_Localization_Model.vod_dataset import (
    align_radar_to_lidar,
    discover_vod_frames,
    load_vod_radar,
    load_vod_radar_to_lidar,
    load_vod_split_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "train_val", "full"),
        default="train",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=64.0)
    parser.add_argument("--y-min", type=float, default=-32.0)
    parser.add_argument("--y-max", type=float, default=32.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--unfiltered-variant", default="radar_20frames")
    parser.add_argument(
        "--filtered-variant",
        default="radar_20frames_temporal_filtered",
    )
    return parser.parse_args()


def _occupancy(
    aligned_points: np.ndarray,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> np.ndarray:
    _xyz, rows, cols, _valid, height, width = metric_to_grid(
        aligned_points[:, :3],
        x_range,
        y_range,
        resolution,
    )
    occupancy = np.zeros((height, width), dtype=np.float32)
    occupancy[rows, cols] = 1.0
    return occupancy


def _aligned_radar(frame) -> np.ndarray:
    radar = load_vod_radar(frame.radar_path)
    lidar_from_radar = load_vod_radar_to_lidar(
        frame.lidar_calibration_path,
        frame.radar_calibration_path,
    )
    return align_radar_to_lidar(radar, lidar_from_radar)


def _save_comparison(
    unfiltered: np.ndarray,
    filtered: np.ndarray,
    destination: Path,
    frame_id: str,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7),
        facecolor="black",
        constrained_layout=True,
    )
    panels = (
        (unfiltered, "20-frame radar occupancy — unfiltered"),
        (filtered, "20-frame radar occupancy — temporal filter"),
    )
    for axis, (occupancy, title) in zip(axes, panels):
        axis.imshow(
            occupancy,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(
            f"{title}\n{int(occupancy.sum()):,} occupied cells",
            color="white",
        )
        axis.set_facecolor("black")
        axis.axis("off")
    figure.suptitle(f"VoD frame {frame_id}", color="white", fontsize=15)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, facecolor="black")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.x_max <= args.x_min or args.y_max <= args.y_min:
        raise ValueError("BEV maxima must exceed minima")
    if args.resolution <= 0.0:
        raise ValueError("--resolution must be positive")

    split_ids = load_vod_split_ids(args.vod_root, args.split)
    requested_ids = split_ids[
        args.start_index : args.start_index + args.limit
    ]
    if not requested_ids:
        raise ValueError("The requested frame range is empty")

    unfiltered_frames = discover_vod_frames(
        args.vod_root,
        args.split,
        radar_variant=args.unfiltered_variant,
        frame_ids=requested_ids,
    )
    filtered_frames = discover_vod_frames(
        args.vod_root,
        args.split,
        radar_variant=args.filtered_variant,
        frame_ids=requested_ids,
    )
    filtered_by_id = {frame.frame_id: frame for frame in filtered_frames}
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    for index, unfiltered_frame in enumerate(unfiltered_frames, 1):
        frame_id = unfiltered_frame.frame_id
        filtered_frame = filtered_by_id[frame_id]
        unfiltered = _occupancy(
            _aligned_radar(unfiltered_frame),
            x_range=x_range,
            y_range=y_range,
            resolution=args.resolution,
        )
        filtered = _occupancy(
            _aligned_radar(filtered_frame),
            x_range=x_range,
            y_range=y_range,
            resolution=args.resolution,
        )
        _save_comparison(
            unfiltered,
            filtered,
            args.output_root / args.split / f"{frame_id}.png",
            frame_id,
        )
        print(f"Rendered {index}/{len(unfiltered_frames)}: {frame_id}", flush=True)

    print(f"Saved comparisons under {args.output_root / args.split}")


if __name__ == "__main__":
    main()
