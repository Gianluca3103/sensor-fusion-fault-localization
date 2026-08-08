"""Build the K-Radar adaptation of the four-channel RadarV2 cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import sys

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # noqa: N801
        def __init__(self, iterable=None, total=None, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def update(self, _count):
            return None

        def close(self):
            return None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.concurrency_utils import iter_bounded_futures
from Fault_Localization_Model.io_utils import atomic_write_text
from PFS_Radar_v2.radar_data import (
    DopplerConfig,
    build_radar_cache_entry,
    kradar_cache_path,
    load_sequence_index,
    radar_cache_is_compatible,
)
from PFS_Radar_v2.radar_types import RadarAlignmentUnavailableError


def _build(task):
    return build_radar_cache_entry(*task)


def _build_batch(tasks):
    outcomes = []
    for task in tasks:
        try:
            _build(task)
            outcomes.append(("created", task[5], task[4], ""))
        except RadarAlignmentUnavailableError as exc:
            outcomes.append(("skipped", task[5], task[4], str(exc)))
        except Exception as exc:
            outcomes.append(
                ("failed", task[5], task[4], f"{type(exc).__name__}: {exc}")
            )
    return outcomes


def _sequence_batches(keyed_tasks, batch_size):
    batches = []
    current = []
    current_sequence = None
    for (sequence, timestamp), task in sorted(keyed_tasks, key=lambda item: item[0]):
        if current and (sequence != current_sequence or len(current) >= batch_size):
            batches.append(current)
            current = []
        current_sequence = sequence
        current.append(task)
    if current:
        batches.append(current)
    return batches


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build single-frame, Doppler-compensated RadarV2 BEVs from K-Radar "
            "tensor-derived pc10p files."
        )
    )
    parser.add_argument("--kradar-root", required=True)
    parser.add_argument(
        "--radar-point-root",
        required=True,
        help="Directory containing <sequence>/rpc_<radar-index>.npy pc10p files.",
    )
    parser.add_argument(
        "--odometry-root",
        required=True,
        help="K-Radar resources/odometry or resources/odometry/gt directory.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-pending-frames",
        type=int,
        default=0,
        help="Process at most this many uncached frames; 0 processes all.",
    )
    parser.add_argument("--x-range", type=float, nargs=2, default=(0.0, 64.0))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-32.0, 32.0))
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--radar-to-lidar-z-m", type=float, default=0.7)
    parser.add_argument("--dynamic-threshold-mps", type=float, default=1.0)
    parser.add_argument("--dynamic-power-quantile", type=float, default=0.9)
    parser.add_argument(
        "--doppler-period-mps",
        type=float,
        default=3.865182436611008,
        help="K-Radar unambiguous Doppler interval used to wrap ego residuals.",
    )
    parser.add_argument("--doppler-sign", choices=("auto", "1", "-1"), default="auto")
    parser.add_argument("--sign-inference-min-speed-mps", type=float, default=0.5)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=30.0)
    args = parser.parse_args()
    if args.num_workers < 1 or args.batch_size < 1:
        parser.error("--num-workers and --batch-size must be at least 1")
    if args.max_pending_frames < 0:
        parser.error("--max-pending-frames must be non-negative")
    if args.resolution <= 0.0 or not math.isfinite(args.resolution):
        parser.error("--resolution must be finite and positive")
    if args.x_range[0] >= args.x_range[1] or args.y_range[0] >= args.y_range[1]:
        parser.error("BEV ranges must be increasing")
    return parser, args


def main():
    parser, args = _parse_args()
    doppler_config = DopplerConfig(
        dynamic_threshold_mps=args.dynamic_threshold_mps,
        doppler_sign=args.doppler_sign,
        sign_inference_min_speed_mps=args.sign_inference_min_speed_mps,
        max_abs_velocity_mps=args.max_abs_velocity_mps,
        doppler_period_mps=args.doppler_period_mps,
        dynamic_power_quantile=args.dynamic_power_quantile,
    )
    try:
        doppler_config.validate()
    except ValueError as exc:
        parser.error(str(exc))

    kradar_root = Path(args.kradar_root)
    radar_point_root = Path(args.radar_point_root)
    odometry_root = Path(args.odometry_root)
    output_root = Path(args.output_root)
    for label, path in (
        ("K-Radar", kradar_root),
        ("pc10p", radar_point_root),
        ("odometry", odometry_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} root does not exist: {path}")
    output_root.mkdir(parents=True, exist_ok=True)

    sequence_dirs = sorted(
        (path for path in radar_point_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not sequence_dirs:
        raise FileNotFoundError(f"No numeric K-Radar sequence directories under {radar_point_root}")

    x_range = tuple(args.x_range)
    y_range = tuple(args.y_range)
    keyed_tasks = []
    available_count = 0
    cached_count = 0
    for sequence_dir in sequence_dirs:
        index = load_sequence_index(
            kradar_root, radar_point_root, odometry_root, sequence_dir.name
        )
        available_count += len(index.radar_indices)
        for position, radar_index in enumerate(index.radar_indices):
            destination = kradar_cache_path(output_root, index.sequence, radar_index)
            compatible = radar_cache_is_compatible(
                destination,
                sequence=index.sequence,
                radar_index=radar_index,
                lidar_index=index.lidar_indices[position],
                timestamp=index.timestamps[position],
                x_range=x_range,
                y_range=y_range,
                resolution=args.resolution,
                doppler_config=doppler_config,
            )
            if compatible:
                cached_count += 1
                continue
            task = (
                kradar_root,
                radar_point_root,
                odometry_root,
                output_root,
                index.sequence,
                radar_index,
                x_range,
                y_range,
                args.resolution,
                args.radar_to_lidar_z_m,
                doppler_config,
            )
            keyed_tasks.append(((int(index.sequence), index.timestamps[position]), task))

    total_pending = len(keyed_tasks)
    if args.max_pending_frames:
        keyed_tasks = keyed_tasks[: args.max_pending_frames]
    batches = _sequence_batches(keyed_tasks, args.batch_size)
    scheduled = sum(len(batch) for batch in batches)
    print(
        f"K-Radar pc10p frames: {available_count} | already cached: {cached_count} | "
        f"pending: {total_pending} | scheduled now: {scheduled} | batches: {len(batches)}"
    )
    skipped_path = output_root / "skipped_alignment_frames.txt"
    failure_path = output_root / "cache_failures.txt"
    if not batches:
        skipped_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        return

    skipped = []

    def handle_outcomes(outcomes):
        for status, radar_index, sequence, message in outcomes:
            if status == "skipped":
                skipped.append((sequence, radar_index, message))
            elif status == "failed":
                report = f"sequence={sequence}\tradar={radar_index}\t{message}"
                atomic_write_text(failure_path, report)
                raise RuntimeError(
                    f"K-Radar RadarV2 cache failed. Details: {failure_path}"
                )

    progress = tqdm(total=scheduled, desc="K-Radar RadarV2 cache")
    try:
        if args.num_workers == 1:
            for batch in batches:
                handle_outcomes(_build_batch(batch))
                progress.update(len(batch))
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = iter_bounded_futures(
                    executor,
                    _build_batch,
                    batches,
                    max_pending=max(args.num_workers * 3, 1),
                )
                for future, batch in futures:
                    handle_outcomes(future.result())
                    progress.update(len(batch))
    finally:
        progress.close()

    failure_path.unlink(missing_ok=True)
    if skipped:
        atomic_write_text(
            skipped_path,
            "\n".join(
                f"sequence={sequence}\tradar={radar_index}\t{message}"
                for sequence, radar_index, message in skipped
            ),
        )
        print(f"Skipped {len(skipped)} frames without valid alignment data.")
    else:
        skipped_path.unlink(missing_ok=True)
    print(f"K-Radar RadarV2 cache complete: {output_root}")


if __name__ == "__main__":
    main()
