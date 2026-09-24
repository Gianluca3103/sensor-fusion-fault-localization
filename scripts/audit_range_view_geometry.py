"""Audit configured sensor rays against one uncropped raw LiDAR scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from models.two_stage_reconstruction_head.range_view.geometry import (
    RangeGeometry, angular_indices, backproject,
)
from models.two_stage_reconstruction_head.voxelization.inputs import load_clean_lidar_from_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True,
                        help="A full-scan range-view artifact, not a legacy BEV-cropped artifact")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    geometry = RangeGeometry.from_json(args.geometry)
    with np.load(args.artifact, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata.get("range_view_full_scan") is not True:
        raise ValueError("Audit requires a full-scan range-view artifact")
    points = load_clean_lidar_from_metadata(metadata)
    rows, cols, distances, valid = angular_indices(points, geometry)
    reconstructed = backproject(rows[valid], cols[valid], distances[valid], geometry)
    errors = np.linalg.norm(reconstructed - points[valid, :3], axis=1)
    summary = {
        "source": str(metadata["source_relative_path"]),
        "point_count": len(points), "projected_count": int(valid.sum()),
        "coverage_fraction": float(valid.mean()) if len(valid) else 0.0,
        "xyz_round_trip_error_m": {
            "median": float(np.median(errors)) if len(errors) else None,
            "p95": float(np.percentile(errors, 95)) if len(errors) else None,
            "p99": float(np.percentile(errors, 99)) if len(errors) else None,
            "max": float(np.max(errors)) if len(errors) else None,
        },
        "geometry": vars(geometry),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
