"""Generate ego-motion aligned 10- and 20-scan VoD radar releases."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import tempfile

import numpy as np

from Fault_Localization_Model.vod_dataset import (
    accumulate_vod_radar_scans,
    load_vod_odom_from_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--stack-sizes", nargs="+", type=int, default=(10, 20))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-step-translation-m",
        type=float,
        default=5.0,
        help="Break history at recording boundaries or implausible pose jumps.",
    )
    parser.add_argument("--overwrite", action="store_true")
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
    public_text, frame_id, history, stack_size, overwrite = task
    public = Path(public_text)
    destination = (
        public
        / f"radar_{stack_size}frames"
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

    public = _public_root(args.vod_root)
    radar_root = public / "radar" / "training" / "velodyne"
    frame_ids = sorted(int(path.stem) for path in radar_root.glob("*.bin"))
    if args.limit is not None:
        frame_ids = frame_ids[: args.limit]
    if not frame_ids:
        raise FileNotFoundError(f"No single-frame VoD radar files found in {radar_root}")

    stack_sizes = sorted(set(args.stack_sizes))
    histories = _histories(
        public,
        frame_ids,
        max(stack_sizes),
        args.max_step_translation_m,
    )
    tasks = [
        (str(public), frame_id, histories[frame_id], size, args.overwrite)
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
        destination = public / f"radar_{size}frames" / "training" / "velodyne"
        count = sum(1 for _ in destination.glob("*.bin"))
        print(f"radar_{size}frames: {count} files in {destination}")


if __name__ == "__main__":
    main()
