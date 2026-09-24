"""Create exact-size HeRCULES full-scan range-view fault/radar datasets.

Unlike the legacy generator, this path never builds BEV reliability maps or
selector crops. It writes unprojected full LiDAR fault points plus a separate
causally stacked, LiDAR-aligned radar cache. The model's forward-FOV selection
is performed later by the range-view loader.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time

import numpy as np

from Fault_Localization_Model.config.defaults import DEFAULT_FOG_ROOT, DEFAULT_INJECTOR_ROOT
from Fault_Localization_Model.fault_injector import build_fault_plan, inject_fault, load_fault_injector
from Fault_Localization_Model.hercules_dataset import (
    HerculesSynchronizationError, discover_hercules_frames, load_frame_radar,
    load_hercules_lidar,
)
from Fault_Localization_Model.io_utils import atomic_savez, atomic_write_json
from Fault_Localization_Model.vod_dataset.vod_io import VODFrame


FORMAT_VERSION = 1
DEFAULT_FAULT_PLAN = (("fov_filter", 1), ("fov_filter", 2),
                      ("fov_filter", 3), ("total_loss", 1))
_INJECTOR = None


def _init_worker() -> None:
    global _INJECTOR
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    _INJECTOR = load_fault_injector(DEFAULT_INJECTOR_ROOT)


def _metadata(path: Path, *, required: tuple[str, ...]) -> dict | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not set(required).issubset(archive.files):
                return None
            if "radar_points" in required and (archive["radar_points"].ndim != 2
                    or archive["radar_points"].shape[1] != 5):
                return None
            if "faulty_lidar_points" in required:
                points = archive["faulty_lidar_points"]
                source_ids = archive["faulty_source_ids"]
                if points.ndim != 2 or points.shape[1] != 4 or source_ids.shape != (len(points),):
                    return None
            return json.loads(str(archive["metadata_json"].item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def _process(task: dict) -> dict:
    frame = VODFrame(**task["frame"])
    source = str(frame.lidar_path)
    radar_path = Path(task["radar_cache_root"]) / frame.split / f"{int(frame.frame_id):05d}.npz"
    sample_path = Path(task["output_root"]) / frame.split / (
        f"{int(frame.frame_id):05d}_{task['fault']}_s{task['severity']}.npz"
    )
    signature = task["signature"]
    radar_metadata = _metadata(radar_path, required=("radar_points",))
    radar_cached = bool(radar_metadata
                        and radar_metadata.get("range_view_radar_cache_version") == FORMAT_VERSION
                        and radar_metadata.get("generator_signature") == signature
                        and radar_metadata.get("source_relative_path") == source)
    sample_metadata = _metadata(sample_path, required=("faulty_lidar_points", "faulty_source_ids"))
    sample_cached = bool(sample_metadata
                         and sample_metadata.get("range_view_full_scan") is True
                         and sample_metadata.get("range_view_format_version") == FORMAT_VERSION
                         and sample_metadata.get("generator_signature") == signature
                         and sample_metadata.get("source_relative_path") == source
                         and sample_metadata.get("injection_seed") == task["injection_seed"])
    if radar_cached and sample_cached:
        return {"status": "cached", "sample": str(sample_path), "radar": str(radar_path),
                "frame_id": frame.frame_id}
    config = dict(task["radar_config"])
    if not radar_cached:
        try:
            _raw, aligned, _radar_to_lidar = load_frame_radar(frame, config)
        except HerculesSynchronizationError as error:
            return {"status": "skipped_synchronization", "frame_id": frame.frame_id,
                    "source": source, "reason": str(error)}
        radar_points = np.column_stack((aligned[:, :3], aligned[:, 3], aligned[:, 5])).astype(np.float32)
        radar_metadata = {
            "range_view_radar_cache_version": FORMAT_VERSION,
            "generator_signature": signature,
            "dataset": "HeRCULES", "split": frame.split, "frame_id": frame.frame_id,
            "source_relative_path": source, "radar_source": str(frame.radar_path),
            "radar_variant": frame.radar_variant,
            "radar_fields": ["x_lidar", "y_lidar", "z_lidar", "rcs", "compensated_radial_velocity"],
            "hercules_alignment": config.get("_hercules_alignment", {}),
        }
        atomic_savez(radar_path, compression_level=task["compression_level"],
                     radar_points=radar_points, metadata_json=np.asarray(json.dumps(radar_metadata)))
    if not sample_cached:
        clean = np.asarray(load_hercules_lidar(frame.lidar_path), dtype=np.float32)
        if clean.ndim != 2 or clean.shape[1] != 4 or not len(clean):
            raise ValueError(f"Expected nonempty HeRCULES Aeva [x,y,z,velocity] scan: {source}")
        if _INJECTOR is None:
            _init_worker()
        injection, injection_metadata = inject_fault(
            task["fault"], clean.copy(), np.arange(len(clean), dtype=np.int64),
            int(task["severity"]), DEFAULT_INJECTOR_ROOT, DEFAULT_FOG_ROOT,
            lidar_corruptions=_INJECTOR, rng_seed=int(task["injection_seed"]),
        )
        metadata = {
            "dataset": "HeRCULES", "representation": "range_view",
            "range_view_full_scan": True, "range_view_format_version": FORMAT_VERSION,
            "generator_signature": signature, "split": frame.split, "frame_id": frame.frame_id,
            "source_relative_path": source, "radar_relative_path": str(frame.radar_path),
            "radar_variant": frame.radar_variant,
            "fault": task["fault"], "severity": int(task["severity"]),
            "injection_seed": int(task["injection_seed"]),
            "injection_metadata": injection_metadata,
            "source_point_count": len(clean), "faulty_point_count": len(injection.points),
            "lidar_field_4": "radial_velocity_not_reflectivity",
            "radar_preprocessing": radar_metadata.get("hercules_alignment", {}),
        }
        atomic_savez(sample_path, compression_level=task["compression_level"],
                     faulty_lidar_points=np.asarray(injection.points[:, :4], dtype=np.float32),
                     faulty_source_ids=np.asarray(injection.source_ids, dtype=np.int64),
                     metadata_json=np.asarray(json.dumps(metadata)))
    return {"status": "created", "sample": str(sample_path), "radar": str(radar_path),
            "frame_id": frame.frame_id}


def _candidate_tasks(frames: list[VODFrame], *, split: str, seed: int,
                     fault_plan: list[tuple[str, int]], common: dict) -> list[dict]:
    # Random frame selection with short chronological blocks retains useful
    # per-worker temporal cache locality without taking only early scenes.
    shuffled = list(frames)
    random.Random(seed).shuffle(shuffled)
    tasks = []
    for rank, frame in enumerate(shuffled):
        fault, severity = fault_plan[rank % len(fault_plan)]
        injection_seed = int(np.random.SeedSequence([seed, int(frame.frame_id)]).generate_state(1)[0])
        tasks.append({"frame": asdict(frame), "fault": fault, "severity": severity,
                      "injection_seed": injection_seed, **common})
    ordered = []
    for start in range(0, len(tasks), 256):
        block = tasks[start:start + 256]
        ordered.extend(sorted(block, key=lambda item: (
            str(item["frame"]["radar_path"]), int(Path(item["frame"]["lidar_path"]).stem)
        )))
    return ordered


def _exclude_split_boundary_history(frames: list[VODFrame], *, split: str,
                                    history_s: float) -> tuple[list[VODFrame], int]:
    """Prevent causal radar accumulation from reaching the previous split."""
    if split == "train":
        return frames, 0
    first_timestamp: dict[Path, int] = {}
    for frame in frames:
        timestamp = int(frame.lidar_path.stem)
        session = frame.radar_path
        first_timestamp[session] = min(first_timestamp.get(session, timestamp), timestamp)
    buffer_ns = int(history_s * 1e9)
    eligible = [frame for frame in frames
                if int(frame.lidar_path.stem) >= first_timestamp[frame.radar_path] + buffer_ns]
    return eligible, len(frames) - len(eligible)


def _fill_split(tasks: list[dict], target: int, *, workers: int,
                progress_every: int = 25) -> dict:
    if target > len(tasks):
        raise ValueError(f"Requested {target} samples from only {len(tasks)} candidate frames")
    accepted: list[dict] = []
    skipped: list[dict] = []
    started = time.perf_counter()
    iterator = iter(tasks)
    last_reported = 0

    def record(result: dict) -> None:
        nonlocal last_reported
        if result["status"] == "skipped_synchronization":
            skipped.append(result)
        else:
            accepted.append(result)
        if len(accepted) != last_reported and (len(accepted) == 1
                or len(accepted) % progress_every == 0 or len(accepted) == target):
            print(f"{tasks[0]['frame']['split']}: {len(accepted)}/{target} "
                  f"skipped={len(skipped)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            last_reported = len(accepted)

    if workers == 1:
        _init_worker()
        for task in iterator:
            record(_process(task))
            if len(accepted) == target:
                break
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_worker) as executor:
            pending = deque()

            def refill() -> None:
                while len(pending) < workers and len(accepted) + len(pending) < target:
                    task = next(iterator, None)
                    if task is None:
                        break
                    pending.append(executor.submit(_process, task))

            refill()
            while pending:
                # Consume in candidate order. The exact accepted set then
                # remains stable across resumptions regardless of CPU timing.
                record(pending.popleft().result())
                refill()
    if len(accepted) != target:
        raise RuntimeError(f"Only {len(accepted)}/{target} synchronized samples available; "
                           f"skipped {len(skipped)} from {len(tasks)} candidates")
    return {"count": len(accepted), "created": sum(row["status"] == "created" for row in accepted),
            "cached": sum(row["status"] == "cached" for row in accepted),
            "skipped_count": len(skipped), "samples": accepted, "skipped": skipped}


def _check_output_roots(data_root: Path, radar_root: Path,
                        quotas: dict[str, int], signature: str) -> None:
    """Refuse mixed experiments rather than silently exceeding split quotas."""
    for root, label in ((data_root, "sample"), (radar_root, "radar")):
        for split, target in quotas.items():
            paths = sorted((root / split).glob("*.npz"))
            if len(paths) > target:
                raise ValueError(f"{root / split} already has {len(paths)} {label} archives; "
                                 f"requested quota is {target}. Use a fresh output root.")
            for path in paths:
                metadata = _metadata(path, required=("radar_points",) if label == "radar"
                                     else ("faulty_lidar_points", "faulty_source_ids"))
                if metadata is None or metadata.get("generator_signature") != signature:
                    raise ValueError(f"{path} is incomplete or from a different generation policy. "
                                     "Use a fresh output root, or resume the matching run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hercules-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-cache-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path,
                        help="Optional existing scene-held-out manifest; must provide enough frames per split")
    parser.add_argument("--train-count", type=int, default=7000)
    parser.add_argument("--val-count", type=int, default=1500)
    parser.add_argument("--test-count", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fault-plan", nargs="+",
                        default=["fov_filter:1", "fov_filter:2",
                                 "fov_filter:3", "total_loss:1"])
    parser.add_argument("--allow-velocity-as-reflectivity", action="store_true",
                        help="Explicitly permit intensity-dependent faults on HeRCULES velocity values")
    parser.add_argument("--radar-frames", type=int, default=20,
                        help="Maximum causal radar scans (adaptive pose/time gates may select fewer)")
    parser.add_argument("--max-radar-age-ms", type=float, default=50.0)
    parser.add_argument("--max-pose-gap-ms", type=float, default=200.0)
    parser.add_argument("--max-history-s", type=float, default=1.0)
    parser.add_argument("--max-translation-m", type=float, default=4.0)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--temporal-radius-m", type=float, default=0.75)
    parser.add_argument("--doppler-sign", choices=("auto", "1", "-1"), default="auto")
    parser.add_argument("--compression-level", type=int, choices=range(10), default=1)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    quotas = {"train": args.train_count, "val": args.val_count, "test": args.test_count}
    if any(count < 0 for count in quotas.values()) or sum(quotas.values()) < 1:
        parser.error("split counts must be nonnegative and total at least one sample")
    if args.workers < 1 or args.radar_frames < 1:
        parser.error("workers and radar-frames must be positive")
    if min(args.max_radar_age_ms, args.max_pose_gap_ms, args.max_history_s,
           args.max_translation_m, args.max_rotation_deg, args.temporal_radius_m) <= 0:
        parser.error("radar synchronization/stack limits must be positive")
    plan = build_fault_plan(args.fault_plan, None, None, DEFAULT_FAULT_PLAN)
    # Only these two injectors are known to preserve Aeva's velocity field
    # while operating solely on return geometry/existence. Other injectors
    # were designed for a four-channel XYZ+reflectivity LiDAR contract.
    risky = sorted({name for name, _severity in plan} - {"fov_filter", "total_loss"})
    if risky and not args.allow_velocity_as_reflectivity:
        parser.error(f"HeRCULES Aeva field 4 is velocity, not reflectivity; {risky} "
                     "were not validated for this four-channel contract. Use "
                     "--allow-velocity-as-reflectivity only for a legacy-compatible "
                     "but physically questionable experiment")
    radar_config = {
        "hercules_radar_frames": args.radar_frames,
        "hercules_max_radar_age_ms": args.max_radar_age_ms,
        "hercules_max_pose_gap_ms": args.max_pose_gap_ms,
        "hercules_temporal_radius": args.temporal_radius_m,
        "hercules_stack": {"max_frames": args.radar_frames,
                           "max_age_s": args.max_history_s,
                           "max_translation_m": args.max_translation_m,
                           "max_rotation_deg": args.max_rotation_deg},
        "hercules_tracking": {"doppler_sign": args.doppler_sign},
    }
    manifest_digest = (hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
                       if args.split_manifest else None)
    signature = hashlib.sha256(json.dumps({"version": FORMAT_VERSION,
        "radar": radar_config, "fault_plan": plan,
        "seed": args.seed, "split_manifest_digest": manifest_digest},
        sort_keys=True).encode()).hexdigest()[:16]
    common = {"output_root": str(args.output_root),
              "radar_cache_root": str(args.radar_cache_root),
              "radar_config": radar_config, "signature": signature,
              "compression_level": args.compression_level}
    available = {}
    boundary_excluded = {}
    all_tasks = {}
    for offset, (split, target) in enumerate(quotas.items()):
        if target == 0:
            available[split] = 0
            boundary_excluded[split] = 0
            all_tasks[split] = []
            continue
        frames = discover_hercules_frames(args.hercules_root, split,
            radar_variant=f"hercules_range_view_v{FORMAT_VERSION}_{signature}",
            split_manifest=args.split_manifest)
        frames, boundary_excluded[split] = _exclude_split_boundary_history(
            frames, split=split, history_s=args.max_history_s)
        available[split] = len(frames)
        all_tasks[split] = _candidate_tasks(frames, split=split, seed=args.seed + offset,
                                            fault_plan=plan, common=common)
        if len(frames) < target:
            raise ValueError(f"{split} has only {len(frames)} frames; {target} requested")
    print(json.dumps({"requested": quotas, "available_before_sync": available,
                      "cross_split_history_excluded": boundary_excluded,
                      "generator_signature": signature, "fault_plan": plan,
                      "radar_frames_cap": args.radar_frames}, indent=2), flush=True)
    if args.plan_only:
        return
    _check_output_roots(args.output_root, args.radar_cache_root, quotas, signature)
    results = {}
    for split, target in quotas.items():
        if target:
            results[split] = _fill_split(all_tasks[split], target, workers=args.workers)
        else:
            results[split] = {"count": 0, "created": 0, "cached": 0,
                              "skipped_count": 0, "samples": [], "skipped": []}
        atomic_write_json(args.output_root / f"range_view_generation_{split}.json", results[split])
    summary = {"requested": quotas, "actual": {split: row["count"] for split, row in results.items()},
               "generator_signature": signature, "radar_frames_cap": args.radar_frames,
               "cross_split_history_excluded": boundary_excluded,
               "fault_plan": plan, "data_root": str(args.output_root),
               "radar_root": str(args.radar_cache_root)}
    atomic_write_json(args.output_root / "range_view_generation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
