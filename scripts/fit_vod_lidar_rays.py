"""Fit and audit a VoD forward virtual-ray grid using clean training scans.

VoD releases motion-compensated XYZI rather than ring IDs or laser calibration.
Quantile rows preserve the released XYZ geometry; they are not factory laser
beam IDs. An optional peak mode estimates beam centres for a separate audit.
Keep the old geometry and checkpoints for the existing benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_lidar, load_vod_split_ids, resolve_vod_public_root,
    vod_partition_for_split,
)
from models.two_stage_reconstruction_head.range_view.geometry import (
    RangeGeometry, angular_indices, project_lidar,
)


def _training_sources(root: Path) -> list[Path]:
    sources: dict[str, Path] = {}
    for sample in sorted((root / "train").glob("*.npz")):
        with np.load(sample, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("range_view_full_scan") is not True or metadata.get("split") != "train":
            raise ValueError(f"Expected a full-scan training artifact: {sample}")
        frame_id = str(metadata["frame_id"])
        source = Path(str(metadata["source_relative_path"]))
        if frame_id in sources and sources[frame_id] != source:
            raise ValueError(f"Conflicting clean LiDAR paths for frame {frame_id}")
        sources[frame_id] = source
    if not sources:
        raise FileNotFoundError(f"No training samples under {root / 'train'}")
    return [sources[key] for key in sorted(sources)]


def _official_training_sources(vod_root: Path) -> list[Path]:
    root = resolve_vod_public_root(vod_root)
    identifiers = load_vod_split_ids(root, "train")
    partition = vod_partition_for_split(root, "train", identifiers)
    return [root / "lidar" / partition / "velodyne" / f"{frame_id}.bin"
            for frame_id in identifiers]


def _even_subset(paths: list[Path], count: int) -> list[Path]:
    if not paths or count < 1:
        return []
    return [paths[index] for index in np.linspace(
        0, len(paths) - 1, min(count, len(paths)), dtype=np.int64)]


def _front_points(path: Path, min_range_m: float, max_range_m: float,
                  *, unique: bool = True) -> np.ndarray:
    points = load_vod_lidar(path)
    radius = np.linalg.norm(points[:, :3], axis=1)
    keep = ((points[:, 0] >= 0) & np.isfinite(points).all(axis=1)
            & (radius >= min_range_m) & (radius <= max_range_m))
    front = points[keep]
    return np.unique(front, axis=0) if unique else front


def _elevations_deg(points: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))


def fit_beam_elevations(histogram: np.ndarray, edges_deg: np.ndarray, *,
                        beams: int, min_separation_deg: float,
                        smoothing_deg: float) -> np.ndarray:
    """Select separated elevation-density peaks and refine each local centre."""
    histogram = np.asarray(histogram, dtype=np.float64)
    edges_deg = np.asarray(edges_deg, dtype=np.float64)
    if (histogram.ndim != 1 or edges_deg.shape != (len(histogram) + 1,)
            or beams < 2 or min_separation_deg <= 0 or smoothing_deg <= 0):
        raise ValueError("Invalid histogram or beam-fitting parameters")
    width = float(edges_deg[1] - edges_deg[0])
    if not np.allclose(np.diff(edges_deg), width) or width <= 0:
        raise ValueError("Histogram elevation bins must have constant positive width")
    sigma_bins = smoothing_deg / width
    half_width = max(2, int(np.ceil(4 * sigma_bins)))
    offsets = np.arange(-half_width, half_width + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(histogram, kernel, mode="same")
    candidates = np.flatnonzero(
        (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])) + 1
    candidates = candidates[np.argsort(smooth[candidates])[::-1]]
    centres = (edges_deg[:-1] + edges_deg[1:]) / 2
    selected: list[int] = []
    for candidate in candidates:
        if all(abs(centres[candidate] - centres[other]) >= min_separation_deg
               for other in selected):
            selected.append(int(candidate))
            if len(selected) == beams:
                break
    if len(selected) != beams:
        raise ValueError(
            f"Found only {len(selected)}/{beams} separated elevation peaks. "
            "Inspect the histogram, increase clean training scans, or supply a beam table."
        )
    refined = []
    local_half_width = max(1, int(round(min_separation_deg / (3 * width))))
    for peak in selected:
        lo = max(0, peak - local_half_width)
        hi = min(len(histogram), peak + local_half_width + 1)
        weights = histogram[lo:hi]
        refined.append(float(np.average(centres[lo:hi], weights=weights)))
    result = np.sort(np.asarray(refined, dtype=np.float64))
    if not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise ValueError("Fitted beam elevations are not finite and strictly increasing")
    return result


def fit_virtual_elevations(histogram: np.ndarray, edges_deg: np.ndarray, *,
                           rows: int, tail_mass: float = 0.001,
                           iterations: int = 30) -> np.ndarray:
    """Fit nonuniform angular rows by weighted 1D quantiles and Lloyd updates.

    Fixed tail rows retain the low-density extremes that peak ranking can miss.
    This minimizes projection error in motion-compensated XYZI, not laser-ring
    calibration error.
    """
    histogram = np.asarray(histogram, dtype=np.float64)
    edges_deg = np.asarray(edges_deg, dtype=np.float64)
    if (histogram.ndim != 1 or edges_deg.shape != (len(histogram) + 1,)
            or rows < 2 or not 0 < tail_mass < 0.5 or iterations < 1
            or not np.isfinite(histogram).all() or np.any(histogram < 0)
            or histogram.sum() <= 0):
        raise ValueError("Invalid elevation histogram or virtual-ray fitting settings")
    centres = (edges_deg[:-1] + edges_deg[1:]) / 2
    cdf = np.cumsum(histogram) / histogram.sum()
    beams = np.interp((np.arange(rows) + 0.5) / rows, cdf, centres)
    lower = float(np.interp(tail_mass, cdf, centres))
    upper = float(np.interp(1 - tail_mass, cdf, centres))
    beams[0], beams[-1] = lower, upper
    for _ in range(iterations):
        labels = np.searchsorted((beams[:-1] + beams[1:]) / 2, centres)
        counts = np.bincount(labels, weights=histogram, minlength=rows)
        sums = np.bincount(labels, weights=histogram * centres, minlength=rows)
        updated = np.where(counts > 0, sums / np.maximum(counts, 1), beams)
        updated[0], updated[-1] = lower, upper
        beams = updated
    if not np.isfinite(beams).all() or np.any(np.diff(beams) <= 0):
        raise ValueError("Could not fit strictly increasing virtual elevation rows")
    return beams


def audit_geometry(sources: list[Path], geometry: RangeGeometry) -> dict:
    """Measure clean-point capture, ray collisions, and XYZ round-trip error."""
    rays = geometry.ray_directions()
    rows = []
    all_errors = []
    for source in sources:
        raw = _front_points(source, geometry.min_range_m, geometry.max_range_m,
                            unique=False)
        points = np.unique(raw, axis=0)
        if not len(points):
            raise ValueError(f"No forward clean LiDAR points in {source}")
        row, col, distance, valid = angular_indices(points, geometry)
        occupied = project_lidar(points, geometry).valid.sum()
        assigned = int(valid.sum())
        approximate = rays[row[valid], col[valid]] * distance[valid, None]
        errors = np.linalg.norm(approximate - points[valid, :3], axis=1)
        all_errors.append(errors)
        rows.append({
            "source": str(source), "forward_points": len(points),
            "raw_forward_points": len(raw),
            "duplicate_fraction": 1 - len(points) / len(raw),
            "assigned_points": assigned, "occupied_ray_cells": int(occupied),
            "assigned_fraction": assigned / len(points),
            "collision_fraction": (assigned - int(occupied)) / max(assigned, 1),
            "roundtrip_p95_m": float(np.quantile(errors, 0.95)) if len(errors) else None,
        })
    total = sum(row["forward_points"] for row in rows)
    assigned = sum(row["assigned_points"] for row in rows)
    occupied = sum(row["occupied_ray_cells"] for row in rows)
    raw_total = sum(row["raw_forward_points"] for row in rows)
    errors = np.concatenate(all_errors)
    return {
        "scans": len(rows), "forward_points": total,
        "raw_forward_points": raw_total,
        "duplicate_fraction": 1 - total / raw_total,
        "assigned_fraction": assigned / total,
        "collision_fraction": (assigned - occupied) / max(assigned, 1),
        "roundtrip_p95_m": float(np.quantile(errors, 0.95)) if len(errors) else None,
        "maximum_forward_points_per_scan": max(row["forward_points"] for row in rows),
        "ray_cells_per_scan": int(np.prod(geometry.shape)),
        "per_scan": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--data-root", type=Path,
                              help="Full-scan samples root; only its train split is read")
    source_group.add_argument("--vod-public", type=Path,
                              help="Official VoD root; only its train ImageSets IDs are read")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-scans", type=int, default=128)
    parser.add_argument("--audit-scans", type=int, default=32)
    parser.add_argument("--azimuth-bins", type=int, default=2048)
    parser.add_argument("--row-mode", choices=("quantile", "beam-peaks"), default="quantile")
    parser.add_argument("--rows", type=int, default=128,
                        help="Angular rows; 128 virtual rows are the audited default")
    parser.add_argument("--min-elevation-deg", type=float, default=-25.0)
    parser.add_argument("--max-elevation-deg", type=float, default=10.0)
    parser.add_argument("--min-range-m", type=float, default=0.5)
    parser.add_argument("--max-range-m", type=float, default=120.0)
    parser.add_argument("--fit-min-range-m", type=float, default=0.5,
                        help="Minimum clean range for row fitting; beam-peaks may benefit from distant returns")
    parser.add_argument("--histogram-width-deg", type=float, default=0.01)
    parser.add_argument("--peak-separation-deg", type=float, default=0.15)
    parser.add_argument("--smoothing-deg", type=float, default=0.025)
    parser.add_argument("--min-assigned-fraction", type=float, default=0.98)
    parser.add_argument("--max-collision-fraction", type=float, default=0.10)
    parser.add_argument("--max-roundtrip-p95-m", type=float, default=0.50)
    args = parser.parse_args()
    if (args.fit_scans < 1 or args.audit_scans < 1 or args.azimuth_bins < 2
            or args.rows < 2 or not -90 < args.min_elevation_deg < args.max_elevation_deg < 90
            or not 0 < args.min_range_m <= args.fit_min_range_m < args.max_range_m
            or args.histogram_width_deg <= 0 or args.peak_separation_deg <= 0
            or args.smoothing_deg <= 0 or not 0 < args.min_assigned_fraction <= 1
            or not 0 <= args.max_collision_fraction < 1
            or args.max_roundtrip_p95_m <= 0):
        parser.error("Invalid ray fit or audit settings")
    sources = (_training_sources(args.data_root) if args.data_root is not None
               else _official_training_sources(args.vod_public))
    if len(sources) < args.fit_scans + args.audit_scans:
        raise ValueError(
            f"Need {args.fit_scans + args.audit_scans} distinct training scans; found {len(sources)}")
    selected = _even_subset(sources, args.fit_scans + args.audit_scans)
    audit_indices = set(np.linspace(0, len(selected) - 1,
                                    args.audit_scans, dtype=np.int64).tolist())
    fit_sources = [source for index, source in enumerate(selected)
                   if index not in audit_indices]
    audit_sources = [source for index, source in enumerate(selected)
                     if index in audit_indices]
    edges = np.linspace(args.min_elevation_deg, args.max_elevation_deg,
                        int(np.ceil((args.max_elevation_deg - args.min_elevation_deg)
                                    / args.histogram_width_deg)) + 1)
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    fitting_points = 0
    for source in fit_sources:
        points = _front_points(source, args.fit_min_range_m, args.max_range_m)
        elevation = _elevations_deg(points)
        histogram += np.histogram(elevation, bins=edges)[0]
        fitting_points += len(points)
    if fitting_points == 0:
        raise ValueError("No clean returns within the fitting range")
    if args.row_mode == "quantile":
        centres_deg = fit_virtual_elevations(histogram, edges, rows=args.rows)
    else:
        centres_deg = fit_beam_elevations(
            histogram, edges, beams=args.rows,
            min_separation_deg=args.peak_separation_deg,
            smoothing_deg=args.smoothing_deg)
    geometry = RangeGeometry(
        beam_elevations_rad=tuple(np.deg2rad(centres_deg)),
        azimuth_bins=args.azimuth_bins,
        min_range_m=args.min_range_m, max_range_m=args.max_range_m,
        azimuth_span_rad=float(np.pi), azimuth_offset_rad=float(-np.pi / 2),
    )
    audit = audit_geometry(audit_sources, geometry)
    reasons = []
    if audit["assigned_fraction"] < args.min_assigned_fraction:
        reasons.append("insufficient clean-point assignment")
    if audit["collision_fraction"] > args.max_collision_fraction:
        reasons.append("too many clean points collide into the same ray")
    if audit["roundtrip_p95_m"] > args.max_roundtrip_p95_m:
        reasons.append("clean-to-ray-to-XYZ geometric error is too large")
    if audit["maximum_forward_points_per_scan"] > audit["ray_cells_per_scan"]:
        reasons.append("grid capacity is smaller than a clean forward scan")
    report = {
        "mode": ("train_fitted_virtual_angular_rows_not_calibrated_rings"
                 if args.row_mode == "quantile" else
                 "train_fitted_elevation_peaks_not_factory_calibration"),
        "row_mode": args.row_mode,
        "fit_scans": len(fit_sources), "fit_points": fitting_points,
        "estimated_beam_elevations_deg": centres_deg.tolist(),
        "minimum_beam_spacing_deg": float(np.diff(centres_deg).min()),
        "median_beam_spacing_deg": float(np.median(np.diff(centres_deg))),
        "audit": audit, "passes": not reasons, "failures": reasons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output.with_name(args.output.stem + "_audit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if reasons:
        raise ValueError(f"Ray geometry audit failed: {', '.join(reasons)}. See {report_path}")
    payload = {
        "beam_elevations_rad": list(geometry.beam_elevations_rad),
        "azimuth_bins": geometry.azimuth_bins,
        "azimuth_span_rad": geometry.azimuth_span_rad,
        "azimuth_offset_rad": geometry.azimuth_offset_rad,
        "min_range_m": geometry.min_range_m,
        "max_range_m": geometry.max_range_m,
        "max_beam_error_rad": None,
        "geometry_mode": report["mode"], "fit_train_scans": len(fit_sources),
        "audit_train_scans": len(audit_sources),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Fitted {len(centres_deg)} {args.row_mode} rows x "
          f"{geometry.azimuth_bins} azimuth bins", flush=True)
    print(f"Held-out train audit: assigned {audit['assigned_fraction']:.2%}, "
          f"collisions {audit['collision_fraction']:.2%}, "
          f"XYZ p95 {audit['roundtrip_p95_m']:.3f} m, "
          f"exact duplicates {audit['duplicate_fraction']:.2%}", flush=True)
    print(f"Geometry: {args.output}\nAudit: {report_path}", flush=True)


if __name__ == "__main__":
    main()
