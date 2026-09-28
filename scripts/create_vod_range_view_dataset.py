"""Generate uncropped VoD range-view faults paired with exactly five radar scans."""

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
from Fault_Localization_Model.vod_dataset.vod_io import discover_vod_frames, load_vod_lidar, load_vod_radar


DEFAULT_PLAN = (("fog_sim", 4), ("fog_sim", 5), ("fov_filter", 1), ("total_loss", 1))
VERSION = 1


def _radar_is_five(frame, radar_cache_root: Path) -> bool:
    cache_path = radar_cache_root / frame.split / f"{int(frame.frame_id):05d}.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Aligned radar cache is missing: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if archive["radar_points"].ndim != 2 or archive["radar_points"].shape[1] != 5:
            raise ValueError(f"Expected five-column aligned radar in {cache_path}")
    if metadata.get("radar_variant") != "radar_5frames" or metadata.get("radar_source") != str(frame.radar_path):
        raise ValueError(f"Radar cache does not match the five-frame raw source: {cache_path}")
    return np.unique(load_vod_radar(frame.radar_path)[:, 6]).size == 5


def _create(frame, *, destination: Path, fault: str, severity: int,
            injection_seed: int, signature: str, injector) -> str:
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
        "radar_relative_path": str(frame.radar_path), "radar_variant": "radar_5frames",
        "radar_stack_frames": 5, "fault": fault, "severity": severity,
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
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fault-plan", nargs="+",
                        default=["fog_sim:4", "fog_sim:5", "fov_filter:1", "total_loss:1"])
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    plan = build_fault_plan(args.fault_plan, None, None, DEFAULT_PLAN)
    signature = hashlib.sha256(json.dumps({"version": VERSION, "plan": plan,
        "seed": args.seed, "radar_variant": "radar_5frames"}, sort_keys=True).encode()).hexdigest()[:16]
    frames = discover_vod_frames(args.vod_root, args.split, radar_variant="radar_5frames")
    random.Random(args.seed + {"train": 0, "val": 1, "test": 2}[args.split]).shuffle(frames)
    injector = load_fault_injector(DEFAULT_INJECTOR_ROOT)
    created = cached = skipped = 0
    selected = 0
    fault_offset = 2 if args.split == "val" else 0
    for frame in frames:
        if not _radar_is_five(frame, args.radar_cache_root):
            skipped += 1
            continue
        fault, severity = plan[(selected + fault_offset) % len(plan)]
        injection_seed = int(np.random.SeedSequence([args.seed, int(frame.frame_id),
            selected + fault_offset]).generate_state(1)[0])
        destination = args.output_root / args.split / f"{int(frame.frame_id):05d}_{fault}_s{severity}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = _create(frame, destination=destination, fault=fault,
                         severity=severity, injection_seed=injection_seed,
                         signature=signature, injector=injector)
        created += result == "created"
        cached += result == "cached"
        selected += 1
        if selected == 1 or selected % 25 == 0 or selected == args.limit:
            print(f"{args.split}: {selected}/{args.limit} created={created} "
                  f"cached={cached} skipped_short_stacks={skipped}", flush=True)
        if selected == args.limit:
            break
    if selected != args.limit:
        raise RuntimeError(f"Only {selected}/{args.limit} frames had five radar scans")


if __name__ == "__main__":
    main()
