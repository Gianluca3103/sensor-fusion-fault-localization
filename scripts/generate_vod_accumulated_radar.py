"""Generate ego-motion aligned 10- and 20-scan VoD radar releases."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from Fault_Localization_Model.vod_dataset import (
    RadarTemporalFilterConfig,
    accumulate_vod_radar_scans,
    load_vod_split_ids,
    load_vod_odom_from_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--stack-sizes", nargs="+", type=int, default=(10, 20))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "train_val", "full"),
        help="Generate only target frames from this official VoD split.",
    )
    parser.add_argument(
        "--max-step-translation-m",
        type=float,
        default=5.0,
        help="Break history at recording boundaries or implausible pose jumps.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-suffix",
        default="",
        choices=("", "temporal_filtered"),
        help=(
            "Optional dataset-directory suffix, for example temporal_filtered "
            "writes radar_20frames_temporal_filtered."
        ),
    )
    parser.add_argument(
        "--basic-validity-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--temporal-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--radar-min-range-m", type=float, default=1.0)
    parser.add_argument("--radar-max-range-m", type=float, default=80.0)
    parser.add_argument("--radar-min-height-m", type=float, default=-5.0)
    parser.add_argument("--radar-max-height-m", type=float, default=5.0)
    parser.add_argument("--radar-min-rcs", type=float)
    parser.add_argument("--radar-max-abs-velocity-mps", type=float)
    parser.add_argument("--temporal-support-radius-m", type=float, default=0.75)
    parser.add_argument("--temporal-min-support-scans", type=int, default=2)
    parser.add_argument(
        "--preserve-current-scan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _public_root(vod_root: Path) -> Path:
    public = vod_root / "view_of_delft_PUBLIC"
    return public if public.is_dir() else vod_root


def _calibration_path(public: Path, frame_id: int) -> Path:
    return public / "radar" / "training" / "calib" / f"{frame_id:05d}.txt"


def _pose_path(public: Path, frame_id: int) -> Path:
    return public / "lidar" / "training" / "pose" / f"{frame_id:05d}.json"


def _radar_path(public: Path, frame_id: int) -> Path:
    return public / "radar" / "training" / "velodyne" / f"{frame_id:05d}.bin"


def _same_recording(public: Path, previous: int, current: int, maximum: float) -> bool:
    if current != previous + 1:
        return False
    try:
        previous_pose = load_vod_odom_from_camera(_pose_path(public, previous))
        current_pose = load_vod_odom_from_camera(_pose_path(public, current))
    except (OSError, ValueError):
        return False
    relative = np.linalg.inv(current_pose) @ previous_pose
    return float(np.linalg.norm(relative[:3, 3])) <= maximum


def _histories(
    public: Path,
    frame_ids: list[int],
    maximum_stack: int,
    max_step_translation_m: float,
) -> dict[int, list[int]]:
    histories: dict[int, list[int]] = {}
    active: list[int] = []
    for frame_id in frame_ids:
        if active and not _same_recording(
            public, active[-1], frame_id, max_step_translation_m
        ):
            active = []
        active.append(frame_id)
        active = active[-maximum_stack:]
        histories[frame_id] = active.copy()
    return histories


def _valid_existing(path: Path) -> bool:
    row_bytes = 7 * np.dtype(np.float32).itemsize
    return (
        path.is_file()
        and path.stat().st_size >= row_bytes
        and path.stat().st_size % row_bytes == 0
    )


def _atomic_tofile(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        points.astype(np.float32, copy=False).tofile(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _generate_one(task: tuple) -> tuple[int, int, int, bool]:
    public_text, frame_id, history, stack_size, overwrite, suffix, filter_values = task
    public = Path(public_text)
    variant = f"radar_{stack_size}frames"
    if suffix:
        variant = f"{variant}_{suffix}"
    destination = (
        public
        / variant
        / "training"
        / "velodyne"
        / f"{frame_id:05d}.bin"
    )
    if not overwrite and _valid_existing(destination):
        rows = destination.stat().st_size // (7 * np.dtype(np.float32).itemsize)
        return stack_size, frame_id, rows, True

    selected = history[-stack_size:]
    points = accumulate_vod_radar_scans(
        [_radar_path(public, item) for item in selected],
        [_pose_path(public, item) for item in selected],
        [_calibration_path(public, item) for item in selected],
        filter_config=(
            RadarTemporalFilterConfig(**filter_values)
            if filter_values is not None
            else None
        ),
    )
    _atomic_tofile(destination, points)
    return stack_size, frame_id, len(points), False


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("num-workers must be at least one")
    if not args.stack_sizes or any(size < 1 for size in args.stack_sizes):
        raise ValueError("stack sizes must be positive")
    if args.max_step_translation_m <= 0.0:
        raise ValueError("max-step-translation-m must be positive")
    suffix = args.output_suffix.strip("_")

    filter_values = None
    if args.basic_validity_filter or args.temporal_filter:
        filter_config = RadarTemporalFilterConfig(
            min_range_m=(
                args.radar_min_range_m if args.basic_validity_filter else 0.0
            ),
            max_range_m=(
                args.radar_max_range_m if args.basic_validity_filter else 1.0e9
            ),
            min_height_m=(
                args.radar_min_height_m if args.basic_validity_filter else -1.0e9
            ),
            max_height_m=(
                args.radar_max_height_m if args.basic_validity_filter else 1.0e9
            ),
            min_rcs=(args.radar_min_rcs if args.basic_validity_filter else None),
            max_abs_compensated_velocity_mps=(
                args.radar_max_abs_velocity_mps
                if args.basic_validity_filter
                else None
            ),
            temporal_radius_m=(
                args.temporal_support_radius_m if args.temporal_filter else None
            ),
            temporal_min_scans=args.temporal_min_support_scans,
            preserve_current_scan=args.preserve_current_scan,
        )
        filter_config.validate()
        filter_values = filter_config.__dict__

    public = _public_root(args.vod_root)
    radar_root = public / "radar" / "training" / "velodyne"
    all_frame_ids = sorted(int(path.stem) for path in radar_root.glob("*.bin"))
    if not all_frame_ids:
        raise FileNotFoundError(f"No single-frame VoD radar files found in {radar_root}")

    stack_sizes = sorted(set(args.stack_sizes))
    histories = _histories(
        public,
        all_frame_ids,
        max(stack_sizes),
        args.max_step_translation_m,
    )
    if args.split is None:
        frame_ids = all_frame_ids
    else:
        requested_ids = set(load_vod_split_ids(public, args.split))
        frame_ids = [
            frame_id
            for frame_id in all_frame_ids
            if f"{frame_id:05d}" in requested_ids
        ]
    if args.limit is not None:
        frame_ids = frame_ids[: args.limit]
    if not frame_ids:
        raise FileNotFoundError("No requested VoD radar target frames were found")
    tasks = [
        (
            str(public),
            frame_id,
            histories[frame_id],
            size,
            args.overwrite,
            suffix,
            filter_values,
        )
        for size in stack_sizes
        for frame_id in frame_ids
    ]

    created = cached = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for index, (size, frame_id, rows, was_cached) in enumerate(
            executor.map(_generate_one, tasks, chunksize=8), 1
        ):
            cached += int(was_cached)
            created += int(not was_cached)
            if index % 500 == 0 or index == len(tasks):
                print(
                    f"Processed {index}/{len(tasks)}; created={created}; "
                    f"cached={cached}; latest={size} scans/{frame_id:05d} "
                    f"({rows} points)",
                    flush=True,
                )

    for size in stack_sizes:
        variant = f"radar_{size}frames" + (f"_{suffix}" if suffix else "")
        destination = public / variant / "training" / "velodyne"
        count = sum(1 for _ in destination.glob("*.bin"))
        print(f"{variant}: {count} files in {destination}")
        manifest = {
            "variant": variant,
            "source": "radar",
            "stack_size": size,
            "ego_motion_compensated": True,
            "basic_validity_filter": args.basic_validity_filter,
            "temporal_filter": args.temporal_filter,
            "filter": filter_values,
            "frames": count,
            "target_split": args.split,
        }
        (destination.parent.parent / "filter_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
