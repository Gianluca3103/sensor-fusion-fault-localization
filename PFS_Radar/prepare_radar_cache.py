from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import sys

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PFS_Radar.radar_data import (
    RadarAlignmentUnavailableError,
    build_radar_cache_entry,
    load_sample_metadata,
    radar_cache_path,
    radar_cache_is_compatible,
)
from Fault_Localization_Model.io_utils import atomic_write_text
from Fault_Localization_Model.concurrency_utils import iter_bounded_futures


def _build(task):
    return build_radar_cache_entry(*task)


def main():
    parser = argparse.ArgumentParser(description="Cache clean, time-aligned Continental radar BEVs for PFS-Radar.")
    parser.add_argument("--dataset-root", required=True, help="Reliability dataset root containing .npz samples recursively.")
    parser.add_argument("--hercules-root", required=True)
    parser.add_argument("--output-root", required=True, help="Radar cache root.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--max-abs-velocity", type=float, default=30.0)
    parser.add_argument(
        "--radar-frame-count",
        type=int,
        default=1,
        help="Causal radar frames accumulated before BEV projection. Use 20 for temporal stacking.",
    )
    parser.add_argument(
        "--require-full-stack",
        action="store_true",
        help="Fail instead of using empty leading history at the start of a sequence.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if not math.isfinite(args.max_delta_ms) or args.max_delta_ms < 0.0:
        parser.error("--max-delta-ms must be non-negative")
    if not math.isfinite(args.max_abs_velocity) or args.max_abs_velocity <= 0.0:
        parser.error("--max-abs-velocity must be positive")
    if args.radar_frame_count < 1:
        parser.error("--radar-frame-count must be at least 1")

    dataset_root = Path(args.dataset_root)
    hercules_root = Path(args.hercules_root)
    output_root = Path(args.output_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not hercules_root.is_dir():
        raise FileNotFoundError(f"HeRCULES root does not exist: {hercules_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sample_paths = sorted(dataset_root.rglob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No reliability .npz samples found under {dataset_root}")

    unique = {}
    cache_geometry = {}
    for sample_path in sample_paths:
        metadata = load_sample_metadata(sample_path)
        destination = radar_cache_path(output_root, metadata)
        geometry = (
            tuple(metadata.get("x_range", [0.0, 64.0])),
            tuple(metadata.get("y_range", [-32.0, 32.0])),
            float(metadata.get("resolution", 0.2)),
        )
        if destination in cache_geometry and cache_geometry[destination] != geometry:
            raise ValueError(
                f"Samples mapping to {destination} request conflicting BEV geometries: "
                f"{cache_geometry[destination]} and {geometry}"
            )
        cache_geometry[destination] = geometry
        unique.setdefault(destination, (sample_path, metadata))

    pending = []
    for destination, (sample_path, metadata) in unique.items():
        if radar_cache_is_compatible(
            destination,
            metadata,
            max_delta_ms=args.max_delta_ms,
            max_abs_velocity=args.max_abs_velocity,
            radar_frame_count=args.radar_frame_count,
            require_full_stack=args.require_full_stack,
        ):
            continue
        pending.append(
            (
                sample_path,
                hercules_root,
                output_root,
                args.max_delta_ms,
                args.max_abs_velocity,
                args.radar_frame_count,
                args.require_full_stack,
            )
        )
    print(
        f"Reliability samples: {len(sample_paths)} | unique LiDAR frames: {len(unique)} | "
        f"already cached: {len(unique) - len(pending)} | pending: {len(pending)}"
    )
    skipped_report_path = output_root / "skipped_alignment_samples.txt"
    failure_report_path = output_root / "cache_failures.txt"
    if not pending:
        skipped_report_path.unlink(missing_ok=True)
        failure_report_path.unlink(missing_ok=True)
        return

    skipped_alignment = []
    workers = max(1, args.num_workers)

    def raise_cache_failure(sample_path, exc):
        report = f"{sample_path}\t{type(exc).__name__}: {exc}"
        atomic_write_text(failure_report_path, report)
        raise RuntimeError(
            f"Radar cache failed for {sample_path}. Details: "
            f"{failure_report_path}"
        ) from exc

    if workers == 1:
        for task in tqdm(pending, desc="Radar cache"):
            try:
                _build(task)
            except RadarAlignmentUnavailableError as exc:
                skipped_alignment.append((task[0], exc))
            except Exception as exc:
                raise_cache_failure(task[0], exc)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            completed_futures = iter_bounded_futures(
                executor,
                _build,
                pending,
                max_pending=max(workers * 3, 1),
            )
            for future, task in tqdm(
                completed_futures,
                total=len(pending),
                desc="Radar cache",
            ):
                try:
                    future.result()
                except RadarAlignmentUnavailableError as exc:
                    skipped_alignment.append((task[0], exc))
                except Exception as exc:
                    raise_cache_failure(task[0], exc)

    failure_report_path.unlink(missing_ok=True)
    if skipped_alignment:
        report = "\n".join(f"{path}\t{exc}" for path, exc in skipped_alignment)
        atomic_write_text(skipped_report_path, report)
        print(
            f"Skipped {len(skipped_alignment)} samples without a complete, pose-aligned "
            f"{args.radar_frame_count}-frame causal history."
        )
    else:
        skipped_report_path.unlink(missing_ok=True)
    print(f"Radar cache complete: {output_root}")


if __name__ == "__main__":
    main()
