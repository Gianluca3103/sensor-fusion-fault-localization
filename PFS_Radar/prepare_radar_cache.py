from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
)


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

    dataset_root = Path(args.dataset_root)
    hercules_root = Path(args.hercules_root)
    output_root = Path(args.output_root)
    sample_paths = sorted(dataset_root.rglob("*.npz"))
    if not sample_paths:
        raise FileNotFoundError(f"No reliability .npz samples found under {dataset_root}")

    unique = {}
    for sample_path in sample_paths:
        metadata = load_sample_metadata(sample_path)
        destination = radar_cache_path(output_root, metadata)
        unique.setdefault(destination, sample_path)

    pending = [
        (
            sample_path,
            hercules_root,
            output_root,
            args.max_delta_ms,
            args.max_abs_velocity,
            args.radar_frame_count,
            args.require_full_stack,
        )
        for destination, sample_path in unique.items()
        if not destination.exists()
    ]
    print(
        f"Reliability samples: {len(sample_paths)} | unique LiDAR frames: {len(unique)} | "
        f"already cached: {len(unique) - len(pending)} | pending: {len(pending)}"
    )
    if not pending:
        return

    failures = []
    skipped_alignment = []
    workers = max(1, args.num_workers)
    if workers == 1:
        for task in tqdm(pending, desc="Radar cache"):
            try:
                _build(task)
            except RadarAlignmentUnavailableError as exc:
                skipped_alignment.append((task[0], exc))
            except Exception as exc:
                failures.append((task[0], exc))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_build, task): task[0] for task in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Radar cache"):
                try:
                    future.result()
                except RadarAlignmentUnavailableError as exc:
                    skipped_alignment.append((futures[future], exc))
                except Exception as exc:
                    failures.append((futures[future], exc))

    if failures:
        preview = "\n".join(f"  {path}: {exc}" for path, exc in failures[:10])
        raise RuntimeError(f"Failed to cache {len(failures)} radar frames:\n{preview}")
    if skipped_alignment:
        report = "\n".join(f"{path}\t{exc}" for path, exc in skipped_alignment)
        (output_root / "skipped_alignment_samples.txt").write_text(report, encoding="utf-8")
        print(
            f"Skipped {len(skipped_alignment)} samples without a complete, pose-aligned "
            f"{args.radar_frame_count}-frame causal history."
        )
    print(f"Radar cache complete: {output_root}")


if __name__ == "__main__":
    main()
