"""Generate uncropped VoD range-view faults paired with up to five radar scans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from Fault_Localization_Model.config.defaults import DEFAULT_FOG_ROOT, DEFAULT_INJECTOR_ROOT
from Fault_Localization_Model.fault_injector import build_fault_plan, inject_fault, load_fault_injector
from Fault_Localization_Model.io_utils import atomic_savez
from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar, discover_vod_frames, load_vod_lidar, load_vod_radar,
    load_vod_radar_to_lidar,
)


DEFAULT_PLAN = (("fog_sim", 4), ("fog_sim", 5), ("fov_filter", 1), ("total_loss", 1))
VERSION = 1


def _ensure_radar_cache(frame, radar_cache_root: Path,
                        min_frames: int) -> tuple[str, int] | None:
    raw = load_vod_radar(frame.radar_path)
    scan_count = int(np.unique(raw[:, 6]).size)
    if scan_count < min_frames:
        return None
    if scan_count > 5:
        raise ValueError(f"Expected at most five radar scans in {frame.radar_path}; found {scan_count}")
    cache_path = radar_cache_root / frame.split / f"{int(frame.frame_id):05d}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if archive["radar_points"].ndim != 2 or archive["radar_points"].shape[1] != 5:
                raise ValueError(f"Expected five-column aligned radar in {cache_path}")
        if (metadata.get("radar_variant") != frame.radar_variant
                or metadata.get("radar_source") != str(frame.radar_path)):
            raise ValueError(f"Radar cache does not match the five-frame raw source: {cache_path}")
        return "cached", scan_count

    lidar_from_radar = load_vod_radar_to_lidar(
        frame.lidar_calibration_path, frame.radar_calibration_path,
    )
    aligned = align_radar_to_lidar(raw, lidar_from_radar)
    points = np.column_stack((aligned[:, :3], aligned[:, 3], aligned[:, 5])).astype(np.float32)
    metadata = {
        "cache_format_version": 1, "dataset": "View-of-Delft", "split": frame.split,
        "frame_id": frame.frame_id, "radar_variant": frame.radar_variant,
        "radar_source": str(frame.radar_path),
        "radar_fields": ["x_lidar", "y_lidar", "z_lidar", "rcs", "compensated_radial_velocity"],
        "radar_stack_frames": scan_count, "radar_stack_max_frames": 5,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(cache_path, compression_level=1, radar_points=points,
                 metadata_json=np.asarray(json.dumps(metadata)))
    return "created", scan_count


def _create(frame, *, destination: Path, fault: str, severity: int,
            injection_seed: int, signature: str, radar_frames: int, injector) -> str:
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("generator_signature") == signature
                    and metadata.get("injection_seed") == injection_seed
                    and metadata.get("range_view_full_scan") is True
                    and "faulty_source_ids" in archive.files):
                return "cached"
        raise ValueError(f"Existing sample uses another generation policy: {destination}")
    clean = np.asarray(load_vod_lidar(frame.lidar_path), dtype=np.float32)
    injection, injection_metadata = inject_fault(
        fault, clean.copy(), np.arange(len(clean), dtype=np.int64), severity,
        DEFAULT_INJECTOR_ROOT, DEFAULT_FOG_ROOT,
        lidar_corruptions=injector, rng_seed=injection_seed,
    )
    metadata = {
        "dataset": "View-of-Delft", "representation": "range_view",
        "range_view_full_scan": True, "range_view_format_version": VERSION,
        "generator_signature": signature, "split": frame.split,
        "frame_id": frame.frame_id, "source_relative_path": str(frame.lidar_path),
        "radar_relative_path": str(frame.radar_path), "radar_variant": frame.radar_variant,
        "radar_stack_frames": radar_frames, "radar_stack_max_frames": 5,
        "fault": fault, "severity": severity,
        "injection_seed": injection_seed, "injection_metadata": injection_metadata,
        "source_point_count": len(clean), "faulty_point_count": len(injection.points),
        "lidar_field_4": "reflectivity",
    }
    atomic_savez(
        destination, compression_level=1,
        faulty_lidar_points=np.asarray(injection.points[:, :4], dtype=np.float32),
        faulty_source_ids=np.asarray(injection.source_ids, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--radar-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int,
                        help="Optional smoke-test cap; omit to process the entire split")
    parser.add_argument("--radar-variant", choices=("radar_5frames", "radar_5frames_rangeview"),
                        default="radar_5frames")
    parser.add_argument("--min-radar-frames", type=int, default=5,
                        help="1 includes recording warm-up frames; 5 requires an exact five-scan stack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fault-plan", nargs="+",
                        default=["fog_sim:4", "fog_sim:5", "fov_filter:1", "total_loss:1"])
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 1 <= args.min_radar_frames <= 5:
        parser.error("--min-radar-frames must be between 1 and 5")
    plan = build_fault_plan(args.fault_plan, None, None, DEFAULT_PLAN)
    signature_policy = {"version": VERSION, "plan": plan,
                        "seed": args.seed, "radar_variant": args.radar_variant}
    if args.min_radar_frames != 5:
        signature_policy["min_radar_frames"] = args.min_radar_frames
    signature = hashlib.sha256(json.dumps(signature_policy, sort_keys=True).encode()).hexdigest()[:16]
    frames = discover_vod_frames(args.vod_root, args.split, radar_variant=args.radar_variant)
    random.Random(args.seed + {"train": 0, "val": 1, "test": 2}[args.split]).shuffle(frames)
    injector = load_fault_injector(DEFAULT_INJECTOR_ROOT)
    created = cached = skipped = radar_created = radar_cached = 0
    selected = 0
    processed = 0
    fault_offset = 2 if args.split == "val" else 0
    for frame in frames:
        processed += 1
        radar_result = _ensure_radar_cache(frame, args.radar_cache_root, args.min_radar_frames)
        if radar_result is None:
            skipped += 1
            continue
        radar_status, radar_frames = radar_result
        radar_created += radar_status == "created"
        radar_cached += radar_status == "cached"
        fault, severity = plan[(selected + fault_offset) % len(plan)]
        injection_seed = int(np.random.SeedSequence([args.seed, int(frame.frame_id),
            selected + fault_offset]).generate_state(1)[0])
        destination = args.output_root / args.split / f"{int(frame.frame_id):05d}_{fault}_s{severity}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = _create(frame, destination=destination, fault=fault,
                         severity=severity, injection_seed=injection_seed,
                         signature=signature, radar_frames=radar_frames, injector=injector)
        created += result == "created"
        cached += result == "cached"
        selected += 1
        if selected == 1 or selected % 25 == 0 or processed == len(frames) or selected == args.limit:
            print(f"{args.split}: processed={processed}/{len(frames)} samples={selected} "
                  f"created={created} cached={cached} radar_created={radar_created} "
                  f"radar_cached={radar_cached} skipped_short_stacks={skipped}", flush=True)
        if args.limit is not None and selected == args.limit:
            break
    if args.limit is not None and selected != args.limit:
        raise RuntimeError(f"Only {selected}/{args.limit} frames had at least {args.min_radar_frames} radar scans")
    summary = {
        "split": args.split, "available_frames": len(frames), "processed_frames": processed,
        "eligible_samples": selected, "created": created, "cached": cached,
        "radar_created": radar_created, "radar_cached": radar_cached,
        "skipped_short_stacks": skipped, "generator_signature": signature,
        "radar_variant": args.radar_variant, "radar_stack_max_frames": 5,
        "min_radar_frames": args.min_radar_frames, "fault_plan": plan,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"range_view_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
