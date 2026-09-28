"""Write an explicit forward VoD angular-bin grid and audit LiDAR coverage.

This is an angular image grid, not a claim about calibrated Velodyne ring
centres. VoD's four-float point files do not retain per-point ring IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import load_vod_lidar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Full-scan range-view sample root; audit uses train only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-samples", type=int, default=16)
    parser.add_argument("--elevation-bins", type=int, default=128)
    parser.add_argument("--azimuth-bins", type=int, default=512)
    parser.add_argument("--min-elevation-deg", type=float, default=-25.0)
    parser.add_argument("--max-elevation-deg", type=float, default=10.0)
    parser.add_argument("--min-range-m", type=float, default=0.5)
    parser.add_argument("--max-range-m", type=float, default=120.0)
    args = parser.parse_args()
    if (args.audit_samples < 1 or args.elevation_bins < 2 or args.azimuth_bins < 2
            or not -90 < args.min_elevation_deg < args.max_elevation_deg < 90
            or not 0 < args.min_range_m < args.max_range_m):
        parser.error("invalid audit count or angular/range bounds")
    paths = sorted((args.data_root / "train").glob("*.npz"))[:args.audit_samples]
    if not paths:
        raise FileNotFoundError(f"No full-scan train samples under {args.data_root}")

    total = kept = 0
    sources = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("range_view_full_scan") is not True:
            raise ValueError(f"Not a full-scan range-view sample: {path}")
        source = Path(str(metadata["source_relative_path"]))
        if source in sources:
            continue
        sources.add(source)
        xyz = load_vod_lidar(source)[:, :3]
        xyz = xyz[xyz[:, 0] >= 0]
        radius = np.linalg.norm(xyz, axis=1)
        elevation = np.degrees(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1])))
        valid = ((radius >= args.min_range_m) & (radius <= args.max_range_m)
                 & (elevation >= args.min_elevation_deg)
                 & (elevation <= args.max_elevation_deg))
        total += len(xyz)
        kept += int(valid.sum())
    if kept / total < 0.99:
        raise ValueError(f"Angular/range bounds retain only {kept/total:.2%} of forward LiDAR points")

    width = (args.max_elevation_deg - args.min_elevation_deg) / args.elevation_bins
    centres_deg = args.min_elevation_deg + (np.arange(args.elevation_bins) + 0.5) * width
    payload = {
        "beam_elevations_rad": np.deg2rad(centres_deg).tolist(),
        "azimuth_bins": args.azimuth_bins,
        "azimuth_span_rad": float(np.pi),
        "azimuth_offset_rad": float(-np.pi / 2),
        "min_range_m": args.min_range_m,
        "max_range_m": args.max_range_m,
        "max_beam_error_rad": float(np.deg2rad(width / 2)),
        "geometry_mode": "uniform_angular_bins_not_calibrated_lidar_rings",
        "elevation_bounds_deg": [args.min_elevation_deg, args.max_elevation_deg],
        "train_audit": {"unique_scans": len(sources), "forward_points": total,
                        "retained_points": kept, "retained_fraction": kept / total},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Geometry: {args.output} | {args.elevation_bins} x {args.azimuth_bins} "
          f"angular cells | retained {kept/total:.2%} forward LiDAR points "
          f"across {len(sources)} train scans", flush=True)


if __name__ == "__main__":
    main()
