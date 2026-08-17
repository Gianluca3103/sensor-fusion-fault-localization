"""Export rich accumulated VoD radar and LiDAR BEV channels for analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez_compressed, atomic_write_json
from Fault_Localization_Model.vod_dataset import (
    BEVGeometry,
    LIDAR_UNAVAILABLE_CHANNELS,
    RADAR_CHANNEL_NOTES,
    SUPPORTED_RADAR_VARIANTS,
    align_radar_to_lidar,
    discover_vod_frames,
    lidar_analysis_channels,
    load_vod_lidar,
    load_vod_radar,
    load_vod_radar_to_lidar,
    radar_analysis_channels,
)


SIGNED_CHANNEL_TOKENS = (
    "velocity",
    "projection_vx",
    "projection_vy",
    "azimuth",
    "elevation",
    "height_",
    "time_index_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--radar-variant",
        choices=SUPPORTED_RADAR_VARIANTS,
        default="radar_5frames",
    )
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
    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _display_limits(
    values: np.ndarray,
    support: np.ndarray,
    *,
    signed: bool,
) -> tuple[float, float]:
    finite = values[support & np.isfinite(values)]
    if not finite.size:
        return (-1.0, 1.0) if signed else (0.0, 1.0)
    if signed:
        magnitude = float(np.quantile(np.abs(finite), 0.99))
        magnitude = max(magnitude, 1.0e-6)
        return -magnitude, magnitude
    lower, upper = np.quantile(finite, (0.01, 0.99))
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def _plot_channels(
    channels: dict[str, np.ndarray],
    destination: Path,
    title: str,
) -> None:
    names = list(channels)
    columns = 5
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 4.2 * rows),
        facecolor="#171717",
        squeeze=False,
    )
    support = channels["occupancy"] > 0
    for axis, name in zip(axes.flat, names):
        values = channels[name]
        signed = any(token in name for token in SIGNED_CHANNEL_TOKENS) and not any(
            token in name for token in ("spread", "variance", "std")
        )
        vmin, vmax = _display_limits(values, support, signed=signed)
        shown = np.ma.masked_where(~support, values)
        image = axis.imshow(
            shown,
            cmap="coolwarm" if signed else "viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(name.replace("_", " "), color="white", fontsize=10)
        axis.set_facecolor("black")
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    for axis in axes.flat[len(names) :]:
        axis.axis("off")
    figure.suptitle(title, color="white", fontsize=16)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight", facecolor="#171717")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    geometry = BEVGeometry(
        x_range=(args.x_min, args.x_max),
        y_range=(args.y_min, args.y_max),
        resolution=args.resolution,
    )
    frames = discover_vod_frames(
        args.vod_root,
        args.split,
        radar_variant=args.radar_variant,
    )
    selected = frames[args.start_index : args.start_index + args.limit]
    if not selected:
        raise ValueError("The requested frame range is empty")

    output_split = args.output_root / args.split
    for index, frame in enumerate(selected, 1):
        lidar = load_vod_lidar(frame.lidar_path)
        raw_radar = load_vod_radar(frame.radar_path)
        lidar_from_radar = load_vod_radar_to_lidar(
            frame.lidar_calibration_path,
            frame.radar_calibration_path,
        )
        aligned_radar = align_radar_to_lidar(raw_radar, lidar_from_radar)
        radar_channels = radar_analysis_channels(raw_radar, aligned_radar, geometry)
        lidar_channels = lidar_analysis_channels(lidar, geometry)

        metadata = {
            "dataset": "View-of-Delft",
            "split": args.split,
            "frame_id": frame.frame_id,
            "radar_variant": args.radar_variant,
            "radar_source_fields": [
                "x",
                "y",
                "z",
                "rcs",
                "raw_radial_velocity",
                "compensated_radial_velocity",
                "time_index",
            ],
            "lidar_source_fields": ["x", "y", "z", "reflectivity"],
            "radar_channel_names": list(radar_channels),
            "lidar_channel_names": list(lidar_channels),
            "radar_channel_notes": RADAR_CHANNEL_NOTES,
            "lidar_unavailable_channels": LIDAR_UNAVAILABLE_CHANNELS,
            "x_range": geometry.x_range,
            "y_range": geometry.y_range,
            "resolution": geometry.resolution,
            "radar_points_total": len(raw_radar),
            "lidar_points_total": len(lidar),
        }
        if args.save_npz:
            arrays = {
                **{f"radar__{name}": value for name, value in radar_channels.items()},
                **{f"lidar__{name}": value for name, value in lidar_channels.items()},
                "metadata_json": np.asarray(json.dumps(metadata)),
            }
            atomic_savez_compressed(
                output_split / "npz" / f"{int(frame.frame_id):05d}.npz",
                **arrays,
            )
        if args.save_figures:
            _plot_channels(
                radar_channels,
                output_split / "radar" / f"{int(frame.frame_id):05d}.png",
                f"VoD frame {frame.frame_id} | {args.radar_variant} analysis",
            )
            _plot_channels(
                lidar_channels,
                output_split / "lidar" / f"{int(frame.frame_id):05d}.png",
                f"VoD frame {frame.frame_id} | LiDAR analysis",
            )
        print(f"[{index:03d}/{len(selected):03d}] frame {frame.frame_id}", flush=True)

    atomic_write_json(
        args.output_root / "channel_manifest.json",
        {
            "dataset": "View-of-Delft",
            "radar_variant": args.radar_variant,
            "split": args.split,
            "frames": [frame.frame_id for frame in selected],
            "radar_channel_notes": RADAR_CHANNEL_NOTES,
            "lidar_unavailable_channels": LIDAR_UNAVAILABLE_CHANNELS,
            "geometry": {
                "x_range": geometry.x_range,
                "y_range": geometry.y_range,
                "resolution": geometry.resolution,
            },
        },
    )
    print(f"Saved analysis under {args.output_root}")


if __name__ == "__main__":
    main()
