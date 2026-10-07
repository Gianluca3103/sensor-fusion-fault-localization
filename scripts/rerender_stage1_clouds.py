"""Rebuild Stage-I HTML from saved PLYs without rerunning the model or using CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.visualize_stage1_confidence_cloud import save_viewer


def read_ply(path: Path, *, scored: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            line = stream.readline()
            if not line or len(header) > 4096:
                raise ValueError(f"Invalid PLY header: {path}")
            header.extend(line)
        lines = header.decode("ascii").splitlines()
        expected = ["property float x", "property float y", "property float z"]
        if scored:
            expected.append("property float confidence")
        if (lines[:2] != ["ply", "format binary_little_endian 1.0"]
                or [line for line in lines if line.startswith("property ")] != expected):
            raise ValueError(f"Unexpected PLY format: {path}")
        count = int(next(line.split()[-1] for line in lines if line.startswith("element vertex ")))
        raw = stream.read()
    width = 4 if scored else 3
    values = np.frombuffer(raw, dtype="<f4")
    if values.size != count * width:
        raise ValueError(f"PLY vertex count and payload differ: {path}")
    rows = values.reshape(count, width)
    return rows[:, :3], rows[:, 3] if scored else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="Defaults to input root; existing HTML views will be replaced")
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--max-plot-points", type=int, default=30000,
                        help="Cap for radar and model points only; clean LiDAR is never capped")
    args = parser.parse_args()
    if args.max_plot_points < 1:
        parser.error("--max-plot-points must be positive")
    destination = args.output_root or args.input_root
    destination.mkdir(parents=True, exist_ok=True)
    for frame_id in args.frame_id:
        prefix = args.input_root / f"{frame_id}_stage1"
        metadata_path = prefix.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        radar, _ = read_ply(prefix.with_name(prefix.name + "_radar.ply"))
        clean, _ = read_ply(prefix.with_name(prefix.name + "_clean_lidar.ply"))
        region_path = prefix.with_name(prefix.name + "_support_region.ply")
        surface_path = prefix.with_name(prefix.name + "_surface_proposals.ply")
        legacy_path = prefix.with_name(prefix.name + "_confidence.ply")
        region = region_path.exists()
        surface = surface_path.exists()
        sites, confidence = read_ply(region_path if region else
                                     (surface_path if surface else legacy_path), scored=True)
        limits = tuple(metadata.get("display_limits_xyz_m", (0,80,-40,40,-5,7)))
        output = destination / f"{frame_id}_stage1.html"
        counts = save_viewer(output, frame_id=frame_id, epoch=metadata.get("epoch"),
                             radar=radar, lidar=clean, sites=sites, confidence=confidence,
                             limits=limits,
                             max_points=max(args.max_plot_points,len(sites)) if region else args.max_plot_points,
                             trained=bool(metadata.get("confidence_trained", False)) or surface or region,
                             calibrated=False,
                             third_name="Predicted LiDAR support region" if region else
                                        ("Predicted LiDAR surface candidates" if surface else
                                         "Stage-I radar-site confidence (legacy)"),
                             description="All panels share one metric camera. "
                             +( "The third panel shows active fine voxels in learned support patches. "
                                if region else ("The third panel shows radar-only predicted LiDAR surface locations. "
                                if surface else "The third panel shows scores at occupied radar voxels, not LiDAR surfaces. "))
                             +"Clean LiDAR is displayed for comparison only.",
                             notice=("Support regions are predicted from radar only, and are not confirmed LiDAR returns. "
                                     if region else ("Proposed LiDAR locations are predicted from radar only. "
                                     if surface else "Legacy confidence is scored at radar voxel centers. "))+
                                    "Raw radar overlay is off by default; scores are not calibrated.")
        print(f"{output} | clean LiDAR: {counts['Clean LiDAR']['shown']:,} of "
              f"{counts['Clean LiDAR']['in_view']:,} in crop (uncapped)", flush=True)


if __name__ == "__main__":
    main()
