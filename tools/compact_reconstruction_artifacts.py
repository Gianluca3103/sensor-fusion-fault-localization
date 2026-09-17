"""Compact generated reconstruction artifacts in place without changing inputs.

The training profile retains every array read by coarse/fine training,
evaluation, PointPillars, visualisation, and fault-selector cache generation.
Generation-only provenance and duplicate diagnostic grids are discarded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez
from Fault_Localization_Model.reliability_maps import TRAINING_RELIABILITY_MAP_KEYS


SAMPLE_KEYS = TRAINING_RELIABILITY_MAP_KEYS | {
    "clean_rgb",
    "faulty_rgb",
    "faulty_lidar_points",
    "observability_confidence",
    "metadata_json",
    "faulty_lidar_input_bev",
}
RADAR_KEYS = {
    "radar_bev",
    "radar_points",
    "radar_point_weights",
    "metadata_json",
}


def _load_compact_arrays(
    path: Path, compression_level: int | None = None
) -> tuple[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        if {"clean_rgb", "faulty_rgb", "faulty_lidar_points"} <= names:
            kind = "sample"
            required = TRAINING_RELIABILITY_MAP_KEYS | {
                "clean_rgb",
                "faulty_rgb",
                "faulty_lidar_points",
                "metadata_json",
            }
            missing = required - names
            if missing:
                raise ValueError("missing sample arrays: " + ", ".join(sorted(missing)))
            keep = SAMPLE_KEYS & names
        elif {"radar_bev", "radar_points"} <= names:
            kind = "radar"
            keep = RADAR_KEYS & names
        else:
            raise ValueError("archive is neither a generated sample nor radar cache")
        arrays = {name: np.asarray(archive[name]) for name in sorted(keep)}

    if kind == "sample":
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["artifact_profile"] = "training"
        if compression_level is not None:
            metadata["artifact_compression_level"] = compression_level
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    elif "metadata_json" in arrays:
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["artifact_profile"] = "training"
        if compression_level is not None:
            metadata["artifact_compression_level"] = compression_level
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    return kind, arrays


def compact_file(path: Path, compression_level: int) -> tuple[int, int, str]:
    before = path.stat().st_size
    kind, arrays = _load_compact_arrays(path, compression_level)
    atomic_savez(path, compression_level=compression_level, **arrays)
    # Re-open the replacement before considering the destructive conversion
    # successful. atomic_savez itself preserves the old file until replace.
    checked_kind, checked = _load_compact_arrays(path)
    if checked_kind != kind or set(checked) != set(arrays):
        raise RuntimeError(f"post-write validation failed for {path}")
    return before, path.stat().st_size, kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losslessly compact reconstruction samples/radar caches in place."
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--compression-level", type=int, choices=range(1, 10), default=6)
    parser.add_argument("--report-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        path
        for root in args.roots
        for path in root.rglob("*.npz")
        if path.is_file()
    )
    if not paths:
        raise FileNotFoundError("No .npz artifacts found under the supplied roots")

    before_total = 0
    after_total = 0
    counts = {"sample": 0, "radar": 0}
    for index, path in enumerate(paths, 1):
        before, after, kind = compact_file(path, args.compression_level)
        before_total += before
        after_total += after
        counts[kind] += 1
        if index == 1 or index % args.report_every == 0 or index == len(paths):
            saved = before_total - after_total
            print(
                f"Compacted {index}/{len(paths)} | samples={counts['sample']} "
                f"radar={counts['radar']} | saved={saved / 2**30:.2f} GiB",
                flush=True,
            )

    reduction = 100.0 * (before_total - after_total) / max(before_total, 1)
    print(
        f"Before: {before_total / 2**30:.2f} GiB\n"
        f"After:  {after_total / 2**30:.2f} GiB\n"
        f"Saved:  {(before_total - after_total) / 2**30:.2f} GiB ({reduction:.1f}%)"
    )


if __name__ == "__main__":
    main()
