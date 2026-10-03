"""Preproject full-scan reconstruction training samples onto a fitted ray grid.

The cache contains training inputs and clean-derived training targets. It must
never be used as a detector input or for unlabeled test evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

from tqdm import tqdm

from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.range_view.input_cache import (
    cache_manifest, cache_settings, write_cached_sample,
)


def _build_one(task: tuple[Path, Path, RangeGeometry, Path, dict, bool]) -> bool:
    return write_cached_sample(*task[:5], resume=task[5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-rear", action="store_true")
    parser.add_argument("--radar-floor-band-m", type=float, default=0.0)
    parser.add_argument("--require-lidar-intensity", action="store_true")
    parser.add_argument("--object-targets", action="store_true",
                        help="Cache clean VoD box classes and radar-supported regions for object-focused training")
    parser.add_argument("--radar-region-row-radius", type=int, default=8)
    parser.add_argument("--radar-region-col-radius", type=int, default=32)
    parser.add_argument("--rebuild", action="store_true", help="Recompute entries even when valid")
    args = parser.parse_args()
    if args.workers < 1 or args.radar_floor_band_m < 0:
        parser.error("workers must be positive and radar floor band must be nonnegative")
    paths = sorted((args.data_root / "train").glob("*.npz"))
    if not paths:
        parser.error(f"No training samples in {args.data_root / 'train'}")
    geometry = RangeGeometry.from_json(args.geometry)
    settings = cache_settings(geometry, forward_only=not args.include_rear,
                              radar_floor_band_m=args.radar_floor_band_m,
                              require_lidar_intensity=args.require_lidar_intensity,
                              include_object_targets=args.object_targets,
                              radar_region_row_radius=args.radar_region_row_radius,
                              radar_region_col_radius=args.radar_region_col_radius)
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = ((path, args.radar_root, geometry, args.output_root, settings, not args.rebuild)
             for path in paths)
    built = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        iterator = executor.map(_build_one, tasks, chunksize=1)
        for count, did_build in enumerate(tqdm(iterator, total=len(paths), desc="Range-view train cache",
                                                unit="sample", dynamic_ncols=True,
                                                disable=not sys.stderr.isatty()), start=1):
            built += int(did_build)
            if not sys.stderr.isatty() and (count % 250 == 0 or count == len(paths)):
                print(f"Cached {count}/{len(paths)} training samples", flush=True)
    manifest = cache_manifest(paths, args.radar_root, settings)
    temporary = args.output_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(args.output_root / "manifest.json")
    print(json.dumps({"samples": len(paths), "built": built,
                      "reused": len(paths) - built, "shape": geometry.shape,
                      "cache_root": str(args.output_root)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
