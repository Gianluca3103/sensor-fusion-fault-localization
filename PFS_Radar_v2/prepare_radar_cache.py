from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import sys

try:
    from tqdm import tqdm
except ImportError:  # Keep cache preparation usable in minimal environments.
    def tqdm(iterable, **_kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.concurrency_utils import iter_bounded_futures
from Fault_Localization_Model.io_utils import atomic_write_text
from Fault_Localization_Model.sample_utils import load_sample_metadata
from PFS_Radar.radar_data import RadarAlignmentUnavailableError, radar_cache_path
from PFS_Radar_v2.radar_data import (
    AdaptiveStackConfig,
    DopplerTrackingConfig,
    build_radar_cache_entry,
    radar_cache_is_compatible,
)


def _build(task):
    return build_radar_cache_entry(*task)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build adaptive, ego-compensated and Doppler-tracked Continental "
            "radar BEVs for PFS-Radar v2."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--hercules-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-delta-ms", type=float, default=30.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Maximum accepted causal frames. Use 0 (the default) for no "
            "frame-count cap; history, translation, and rotation gates still apply."
        ),
    )
    parser.add_argument("--max-history-s", type=float, default=1.0)
    parser.add_argument("--max-translation-m", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--weight-time-s", type=float, default=0.5)
    parser.add_argument("--weight-translation-m", type=float, default=2.0)
    parser.add_argument("--weight-rotation-deg", type=float, default=3.0)
    parser.add_argument("--dynamic-threshold-mps", type=float, default=1.0)
    parser.add_argument(
        "--doppler-sign",
        choices=("auto", "1", "-1"),
        default="auto",
        help=(
            "Raw Doppler convention. Auto chooses the sign whose ego-compensated "
            "residual best fits the static majority in each frame."
        ),
    )
    parser.add_argument("--sign-inference-min-speed-mps", type=float, default=0.5)
    parser.add_argument("--cluster-eps-m", type=float, default=1.2)
    parser.add_argument("--cluster-min-samples", type=int, default=2)
    parser.add_argument("--association-distance-m", type=float, default=3.0)
    parser.add_argument("--min-track-hits", type=int, default=2)
    parser.add_argument("--velocity-smoothing", type=float, default=0.5)
    parser.add_argument("--max-abs-velocity-mps", type=float, default=30.0)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if not math.isfinite(args.max_delta_ms) or args.max_delta_ms < 0.0:
        parser.error("--max-delta-ms must be finite and non-negative")
    if args.max_frames < 0:
        parser.error("--max-frames must be 0 (unlimited) or a positive integer")
    return parser, args


def main():
    parser, args = _parse_args()
    stack_config = AdaptiveStackConfig(
        max_frames=None if args.max_frames == 0 else args.max_frames,
        max_age_s=args.max_history_s,
        max_translation_m=args.max_translation_m,
        max_rotation_deg=args.max_rotation_deg,
        weight_time_s=args.weight_time_s,
        weight_translation_m=args.weight_translation_m,
        weight_rotation_deg=args.weight_rotation_deg,
    )
    tracking_config = DopplerTrackingConfig(
        dynamic_threshold_mps=args.dynamic_threshold_mps,
        doppler_sign=args.doppler_sign,
        sign_inference_min_speed_mps=args.sign_inference_min_speed_mps,
        cluster_eps_m=args.cluster_eps_m,
        cluster_min_samples=args.cluster_min_samples,
        association_distance_m=args.association_distance_m,
        min_track_hits=args.min_track_hits,
        velocity_smoothing=args.velocity_smoothing,
        max_abs_velocity_mps=args.max_abs_velocity_mps,
    )
    try:
        stack_config.validate()
        tracking_config.validate()
    except ValueError as exc:
        parser.error(str(exc))

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
        raise FileNotFoundError(
            f"No reliability .npz samples found under {dataset_root}"
        )

    unique: dict[Path, tuple[Path, dict]] = {}
    cache_geometry: dict[Path, tuple] = {}
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
                f"Samples mapping to {destination} request conflicting BEV "
                f"geometries: {cache_geometry[destination]} and {geometry}"
            )
        cache_geometry[destination] = geometry
        unique.setdefault(destination, (sample_path, metadata))

    pending = []
    for destination, (sample_path, metadata) in unique.items():
        if radar_cache_is_compatible(
            destination,
            metadata,
            max_delta_ms=args.max_delta_ms,
            stack_config=stack_config,
            tracking_config=tracking_config,
        ):
            continue
        pending.append(
            (
                sample_path,
                hercules_root,
                output_root,
                args.max_delta_ms,
                stack_config,
                tracking_config,
            )
        )
    print(
        f"Reliability samples: {len(sample_paths)} | "
        f"unique LiDAR frames: {len(unique)} | "
        f"already cached: {len(unique) - len(pending)} | pending: {len(pending)}"
    )
    skipped_path = output_root / "skipped_alignment_samples.txt"
    failure_path = output_root / "cache_failures.txt"
    if not pending:
        skipped_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        return

    skipped = []

    def raise_cache_failure(sample_path, exc):
        report = f"{sample_path}\t{type(exc).__name__}: {exc}"
        atomic_write_text(failure_path, report)
        raise RuntimeError(
            f"Radar v2 cache failed for {sample_path}. Details: {failure_path}"
        ) from exc

    if args.num_workers == 1:
        for task in tqdm(pending, desc="Radar v2 cache"):
            try:
                _build(task)
            except RadarAlignmentUnavailableError as exc:
                skipped.append((task[0], exc))
            except Exception as exc:
                raise_cache_failure(task[0], exc)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = iter_bounded_futures(
                executor,
                _build,
                pending,
                max_pending=max(args.num_workers * 3, 1),
            )
            for future, task in tqdm(
                futures, total=len(pending), desc="Radar v2 cache"
            ):
                try:
                    future.result()
                except RadarAlignmentUnavailableError as exc:
                    skipped.append((task[0], exc))
                except Exception as exc:
                    raise_cache_failure(task[0], exc)

    failure_path.unlink(missing_ok=True)
    if skipped:
        atomic_write_text(
            skipped_path,
            "\n".join(f"{path}\t{exc}" for path, exc in skipped),
        )
        print(f"Skipped {len(skipped)} samples without valid causal pose history.")
    else:
        skipped_path.unlink(missing_ok=True)
    print(f"Radar v2 cache complete: {output_root}")


if __name__ == "__main__":
    main()
