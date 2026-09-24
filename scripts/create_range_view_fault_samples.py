"""Regenerate injected faults on full central LiDAR scans for range-view training.

Legacy reconstruction artifacts identify each raw scan/fault/seed but contain
BEV-cropped faulty points. This script keeps their split and metadata while
re-running the same fault on the entire sensor scan into a separate root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.config.defaults import DEFAULT_FOG_ROOT, DEFAULT_INJECTOR_ROOT
from Fault_Localization_Model.fault_injector import inject_fault, load_fault_injector
from Fault_Localization_Model.hercules_dataset import load_hercules_lidar
from Fault_Localization_Model.io_utils import atomic_savez
from Fault_Localization_Model.vod_dataset.vod_io import load_vod_lidar


FORMAT_VERSION = 1


def _one(path: Path, destination: Path, injector) -> str:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as archive:
            previous = json.loads(str(archive["metadata_json"].item()))
            if (previous.get("range_view_full_scan") is True
                    and previous.get("range_view_format_version") == FORMAT_VERSION
                    and previous.get("source_relative_path") == metadata.get("source_relative_path")):
                return "cached"
    raw_path = Path(str(metadata["source_relative_path"]))
    dataset = str(metadata.get("dataset", "")).strip().lower()
    if dataset in {"view-of-delft", "view of delft", "vod"}:
        clean = load_vod_lidar(raw_path)
    elif dataset == "hercules":
        clean = load_hercules_lidar(raw_path)
    else:
        raise ValueError(f"Unsupported LiDAR dataset in {path}: {dataset!r}")
    if "injection_seed" not in metadata:
        raise ValueError(f"{path} lacks the injection seed required to reproduce its fault")
    clean = np.asarray(clean, dtype=np.float32)
    ids = np.arange(len(clean), dtype=np.int64)
    injection, injection_metadata = inject_fault(
        metadata["fault"], clean.copy(), ids, int(metadata["severity"]),
        DEFAULT_INJECTOR_ROOT, DEFAULT_FOG_ROOT,
        lidar_corruptions=injector, rng_seed=int(metadata["injection_seed"]),
    )
    new_metadata = dict(metadata)
    new_metadata.pop("point_filter", None)
    new_metadata.update({
        "representation": "range_view", "range_view_full_scan": True,
        "range_view_format_version": FORMAT_VERSION,
        "remove_added_points": False,
        "source_point_count": len(clean),
        "faulty_point_count": len(injection.points),
        "injection_metadata": injection_metadata,
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        destination, compression_level=6,
        faulty_lidar_points=np.asarray(injection.points[:, :4], dtype=np.float32),
        faulty_source_ids=np.asarray(injection.source_ids, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(new_metadata)),
    )
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-name", help="Regenerate exactly one source artifact by filename")
    args = parser.parse_args()
    if args.sample_name:
        if Path(args.sample_name).name != args.sample_name or not args.sample_name.endswith(".npz"):
            parser.error("--sample-name must be a single .npz filename")
        paths = [args.source_root / args.split / args.sample_name]
        if not paths[0].is_file():
            raise FileNotFoundError(paths[0])
    else:
        paths = sorted((args.source_root / args.split).glob("*.npz"))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"No source artifacts in {args.source_root / args.split}")
    injector = load_fault_injector(DEFAULT_INJECTOR_ROOT)
    counts = {"created": 0, "cached": 0}
    for index, path in enumerate(paths, start=1):
        result = _one(path, args.output_root / args.split / path.name, injector)
        counts[result] += 1
        if index == 1 or index % 25 == 0 or index == len(paths):
            print(f"{args.split}: {index}/{len(paths)} created={counts['created']} cached={counts['cached']}", flush=True)


if __name__ == "__main__":
    main()
