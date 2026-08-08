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


def _build_batch(tasks):
    """Build a chronological scene batch while retaining worker-local caches."""

    outcomes = []
    for task in tasks:
        try:
            _build(task)
            outcomes.append(("created", task[0], ""))
        except RadarAlignmentUnavailableError as exc:
            outcomes.append(("skipped", task[0], str(exc)))
        except Exception as exc:
            outcomes.append(("failed", task[0], f"{type(exc).__name__}: {exc}"))
    return outcomes


def _scene_batches(keyed_tasks, batch_size):
    """Keep each worker job chronological and confined to one scene/session."""

    batches = []
    current = []
    current_scene = None
    for scene_key, task in sorted(keyed_tasks, key=lambda item: item[0]):
        task_scene = scene_key[:2]
        if current and (task_scene != current_scene or len(current) >= batch_size):
            batches.append(current)
            current = []
        current_scene = task_scene
        current.append(task)
    if current:
        batches.append(current)
    return batches


def main():
    parser = argparse.ArgumentParser(description="Cache clean, time-aligned Continental radar BEVs for PFS-Radar.")
    parser.add_argument("--dataset-root", required=True, help="Reliability dataset root containing .npz samples recursively.")
    parser.add_argument("--hercules-root", required=True)
    parser.add_argument("--output-root", required=True, help="Radar cache root.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Chronological samples processed per worker job. Larger batches "
            "reduce scheduling overhead and improve per-worker cache reuse."
        ),
    )
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument("--max-abs-velocity", type=float, default=30.0)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not math.isfinite(args.max_delta_ms) or args.max_delta_ms < 0.0:
        parser.error("--max-delta-ms must be non-negative")
    if not math.isfinite(args.max_abs_velocity) or args.max_abs_velocity <= 0.0:
        parser.error("--max-abs-velocity must be positive")

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
        ):
            continue
        pending.append(
            (
                (
                    str(metadata.get("scene") or metadata.get("day") or ""),
                    str(metadata.get("session") or ""),
                    int(metadata["timestamp"]),
                ),
                (
                    sample_path,
                    hercules_root,
                    output_root,
                    args.max_delta_ms,
                    args.max_abs_velocity,
                ),
            )
        )
    batches = _scene_batches(pending, args.batch_size)
    pending_count = sum(len(batch) for batch in batches)
    print(
        f"Reliability samples: {len(sample_paths)} | unique LiDAR frames: {len(unique)} | "
        f"already cached: {len(unique) - len(pending)} | pending: {len(pending)} | "
        f"chronological batches: {len(batches)}"
    )
    skipped_report_path = output_root / "skipped_alignment_samples.txt"
    failure_report_path = output_root / "cache_failures.txt"
    if not pending:
        skipped_report_path.unlink(missing_ok=True)
        failure_report_path.unlink(missing_ok=True)
        return

    skipped_alignment = []
    workers = max(1, args.num_workers)

    def raise_cache_failure(sample_path, message):
        report = f"{sample_path}\t{message}"
        atomic_write_text(failure_report_path, report)
        raise RuntimeError(
            f"Radar cache failed for {sample_path}. Details: "
            f"{failure_report_path}"
        )

    if workers == 1:
        progress = tqdm(total=pending_count, desc="Radar cache")
        try:
            for batch in batches:
                for status, sample_path, message in _build_batch(batch):
                    if status == "skipped":
                        skipped_alignment.append((sample_path, message))
                    elif status == "failed":
                        raise_cache_failure(sample_path, message)
                    progress.update(1)
        finally:
            progress.close()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            completed_futures = iter_bounded_futures(
                executor,
                _build_batch,
                batches,
                max_pending=max(workers * 3, 1),
            )
            progress = tqdm(total=pending_count, desc="Radar cache")
            try:
                for future, batch in completed_futures:
                    try:
                        outcomes = future.result()
                    except Exception as exc:
                        raise_cache_failure(
                            batch[0][0],
                            f"{type(exc).__name__}: {exc}",
                        )
                    for status, sample_path, message in outcomes:
                        if status == "skipped":
                            skipped_alignment.append((sample_path, message))
                        elif status == "failed":
                            raise_cache_failure(sample_path, message)
                    progress.update(len(batch))
            finally:
                progress.close()

    failure_report_path.unlink(missing_ok=True)
    if skipped_alignment:
        report = "\n".join(f"{path}\t{exc}" for path, exc in skipped_alignment)
        atomic_write_text(skipped_report_path, report)
        print(
            f"Skipped {len(skipped_alignment)} samples without an aligned radar frame."
        )
    else:
        skipped_report_path.unlink(missing_ok=True)
    print(f"Radar cache complete: {output_root}")


if __name__ == "__main__":
    main()
