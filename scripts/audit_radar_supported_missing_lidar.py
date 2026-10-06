"""Audit where radar can support genuinely missing VoD LiDAR returns.

The support mask uses radar and observed faulty LiDAR only. Clean LiDAR and
annotated boxes are loaded afterward, exclusively to measure that mask.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar,
    discover_vod_frames,
    load_vod_lidar,
    load_vod_lidar_to_camera,
    load_vod_radar,
    load_vod_radar_to_lidar,
)
from models.two_stage_reconstruction_head.cross_modal_data import observed_lidar_height_mask
from models.two_stage_reconstruction_head.voxelization.inputs import read_sample_metadata


CLASSES = ("Car", "Pedestrian", "Cyclist", "Background")
CLASS_ALIASES = {"Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist", "bicycle": "Cyclist"}
BOUNDS = np.asarray(((0.0, 64.0), (-32.0, 32.0), (-3.0, 5.0)), dtype=np.float32)


def in_working_volume(xyz: np.ndarray) -> np.ndarray:
    """Use the repository's forward 3D voxel working volume."""
    return np.isfinite(xyz).all(axis=1) & np.all(
        (xyz >= BOUNDS[:, 0]) & (xyz < BOUNDS[:, 1]), axis=1
    )


def radar_supported_seeds(
    radar_xyz: np.ndarray, *, neighbor_radius_m: float, min_neighbors: int,
    radar_scan_ids: np.ndarray | None = None, min_scans: int = 1,
) -> np.ndarray:
    """Select locally repeated radar evidence; one return counts as itself."""
    if min_neighbors < 1 or min_scans < 1 or neighbor_radius_m <= 0:
        raise ValueError("Support settings must be positive")
    if len(radar_xyz) == 0 or len(radar_xyz) < min_neighbors:
        return np.empty((0, 3), dtype=np.float32)
    tree = cKDTree(radar_xyz)
    if min_scans == 1:
        distances, _ = tree.query(radar_xyz, k=min_neighbors, workers=1)
        kth_distance = distances if min_neighbors == 1 else distances[:, -1]
        return radar_xyz[kth_distance <= neighbor_radius_m]
    if radar_scan_ids is None or len(radar_scan_ids) != len(radar_xyz):
        raise ValueError("Scan IDs are required for multi-scan support")
    neighborhoods = tree.query_ball_point(radar_xyz, neighbor_radius_m, workers=1)
    keep = np.fromiter(
        (len(indices) >= min_neighbors and
         np.unique(radar_scan_ids[indices]).size >= min_scans
         for indices in neighborhoods),
        dtype=bool, count=len(radar_xyz),
    )
    return radar_xyz[keep]


def supported_points(xyz: np.ndarray, seeds: np.ndarray, radius_m: float) -> np.ndarray:
    """Union of metric balls around radar seeds, with no target information."""
    if radius_m <= 0:
        raise ValueError("Support radius must be positive")
    if len(seeds) == 0:
        return np.zeros(len(xyz), dtype=bool)
    distance, _ = cKDTree(seeds).query(xyz, k=1, workers=1)
    return distance <= radius_m


def missing_clean_indices(clean_count: int, faulty_source_ids: np.ndarray) -> np.ndarray:
    """Source IDs identify exact clean returns removed by the fault injector."""
    ids = np.asarray(faulty_source_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("faulty_source_ids must be a 1D integer array")
    retained = ids[ids >= 0]
    if np.any(retained >= clean_count):
        raise ValueError("faulty_source_ids contain a clean index out of range")
    missing = np.ones(clean_count, dtype=bool)
    missing[retained] = False
    return np.flatnonzero(missing)


def missing_point_objects(
    xyz: np.ndarray, labels_path: Path, calibration_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Assign each missing clean point to one annotated 3D object, if any."""
    camera_from_lidar = load_vod_lidar_to_camera(calibration_path)
    camera = xyz.astype(np.float64) @ camera_from_lidar[:3, :3].T + camera_from_lidar[:3, 3]
    object_id = np.full(len(xyz), -1, dtype=np.int32)
    names: list[str] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] not in CLASS_ALIASES:
            continue
        if len(fields) < 15:
            raise ValueError(f"Malformed VoD label: {labels_path}: {line!r}")
        height, width, length, x, y, z, yaw = map(float, fields[8:15])
        if not np.isfinite([height, width, length, x, y, z, yaw]).all() or min(height, width, length) <= 0:
            raise ValueError(f"Invalid VoD 3D box: {labels_path}: {line!r}")
        delta = camera - (x, y, z)
        local_x = np.cos(yaw) * delta[:, 0] - np.sin(yaw) * delta[:, 2]
        local_z = np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 2]
        inside = ((np.abs(local_x) <= width / 2) & (np.abs(local_z) <= length / 2)
                  & (delta[:, 1] >= -height) & (delta[:, 1] <= 0))
        object_id[inside & (object_id < 0)] = len(names)
        names.append(CLASS_ALIASES[fields[0]])
    return object_id, names


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--neighbor-radius-m", type=float, default=1.5,
                        help="Radius used to decide whether a radar return has nearby radar evidence")
    parser.add_argument("--min-neighbors", type=int, default=3,
                        help="Minimum radar returns in the neighbor radius, counting the center return")
    parser.add_argument("--min-scans", type=int, default=1,
                        help="Optional minimum distinct scan ordinals in a radar neighborhood")
    parser.add_argument("--support-radii-m", type=float, nargs="+", default=(0.5, 1.0, 2.0),
                        help="3D distance from a supported radar return to an eligible reconstruction site")
    parser.add_argument("--radar-height-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--volume-probes", type=int, default=4096,
                        help="Uniform 3D sites per frame for estimating the selected volume fraction")
    parser.add_argument("--limit", type=int, help="Evenly spaced pilot sample count; omit for the full split")
    args = parser.parse_args()
    if (args.neighbor_radius_m <= 0 or args.min_neighbors < 1 or args.min_scans < 1 or
            not args.support_radii_m or any(radius <= 0 for radius in args.support_radii_m) or
            args.volume_probes < 1 or (args.limit is not None and args.limit < 1)):
        parser.error("Radii, neighbor count, scan count, and limit must be positive")
    return args


def main() -> None:
    args = _arguments()
    paths = sorted((args.samples_root / args.split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No reconstruction samples in {args.samples_root / args.split}")
    if args.limit and args.limit < len(paths):
        paths = [paths[index] for index in np.linspace(0, len(paths) - 1, args.limit, dtype=int)]
    metadata = [read_sample_metadata(path) for path in paths]
    ids = [str(item["frame_id"]).zfill(5) for item in metadata]
    if len(ids) != len(set(ids)):
        raise ValueError("Audit requires one fault sample per physical frame")
    for item in metadata:
        if item.get("split") != args.split or not item.get("range_view_full_scan"):
            raise ValueError("Sample split or full-scan metadata does not match the audit")
    frames = {frame.frame_id: frame for frame in discover_vod_frames(
        args.vod_root, args.split, radar_variant=args.radar_variant, frame_ids=ids)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    totals: dict[tuple[float, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    radii = sorted(set(args.support_radii_m))
    volume_fractions: dict[float, list[float]] = defaultdict(list)
    probe_rng = np.random.default_rng(42)
    volume_probes = probe_rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1],
                                      size=(args.volume_probes, 3)).astype(np.float32)

    for index, (path, item, frame_id) in enumerate(zip(paths, metadata, ids), 1):
        frame = frames[frame_id]
        with np.load(path, allow_pickle=False) as archive:
            faulty = np.asarray(archive["faulty_lidar_points"], dtype=np.float32)
            source_ids = np.asarray(archive["faulty_source_ids"])
        if len(faulty) != len(source_ids) or faulty.ndim != 2 or faulty.shape[1] < 3:
            raise ValueError(f"Malformed faulty source mapping: {path}")
        radar = load_vod_radar(frame.radar_path)
        radar = align_radar_to_lidar(radar, load_vod_radar_to_lidar(
            frame.lidar_calibration_path, frame.radar_calibration_path))
        if args.radar_height_filter:
            radar = radar[observed_lidar_height_mask(radar, faulty)]
        radar = radar[in_working_volume(radar[:, :3])]
        seeds = radar_supported_seeds(
            radar[:, :3], neighbor_radius_m=args.neighbor_radius_m,
            min_neighbors=args.min_neighbors, radar_scan_ids=radar[:, 6],
            min_scans=args.min_scans,
        )
        # Uniform volume probes and the support tree use inference inputs only.
        seed_tree = cKDTree(seeds) if len(seeds) else None
        probe_distances = (seed_tree.query(volume_probes, k=1, workers=1)[0]
                           if seed_tree is not None else np.full(len(volume_probes), np.inf))

        # No target or annotation is read until the radar-only candidate set is fixed.
        clean = load_vod_lidar(frame.lidar_path)
        missing_indices = missing_clean_indices(len(clean), source_ids)
        missing_xyz = clean[missing_indices, :3]
        within = in_working_volume(missing_xyz)
        missing_xyz = missing_xyz[within]
        object_ids, object_names = missing_point_objects(
            missing_xyz, frame.lidar_path.parent.parent / "label_2" / f"{frame_id}.txt",
            frame.lidar_calibration_path,
        )
        fault = str(item.get("fault", "unknown"))
        point_classes = np.asarray([object_names[object_id] if object_id >= 0 else "Background"
                                    for object_id in object_ids])
        target_distances = (seed_tree.query(missing_xyz, k=1, workers=1)[0]
                            if seed_tree is not None and len(missing_xyz)
                            else np.full(len(missing_xyz), np.inf))

        for radius in radii:
            supported = target_distances <= radius
            volume_fraction = float(np.mean(probe_distances <= radius))
            volume_fractions[radius].append(volume_fraction)
            for name in (*CLASSES, "All"):
                selected = np.ones(len(missing_xyz), dtype=bool) if name == "All" else point_classes == name
                count = int(selected.sum())
                hit = int(np.count_nonzero(selected & supported))
                group = totals[(radius, fault, name)]
                group["missing"] += count
                group["supported_missing"] += hit
                if name != "All" and name != "Background":
                    for object_id, object_name in enumerate(object_names):
                        if object_name != name:
                            continue
                        on_object = object_ids == object_id
                        if np.any(on_object):
                            group["objects_with_missing"] += 1
                            group["objects_with_supported_missing"] += int(np.any(on_object & supported))
                rows.append({
                    "frame_id": frame_id, "fault": fault, "class": name,
                    "support_radius_m": radius, "radar_returns": len(radar),
                    "supported_radar_seeds": len(seeds),
                    "estimated_support_volume_fraction": volume_fraction,
                    "missing_returns": count, "supported_missing_returns": hit,
                    "within_region_fraction": hit / count if count else "",
                })
        if index % 50 == 0 or index == len(paths):
            print(f"Audited {index}/{len(paths)} {args.split} samples", flush=True)

    columns = list(rows[0])
    with (args.output_root / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "definition": {
            "split": args.split, "frames": len(paths), "radar_variant": args.radar_variant,
            "working_volume_xyz_m": BOUNDS.tolist(), "radar_height_filter": args.radar_height_filter,
            "neighbor_radius_m": args.neighbor_radius_m, "min_neighbors": args.min_neighbors,
            "min_scans": args.min_scans, "support_radii_m": radii,
            "volume_probes_per_frame": args.volume_probes,
            "missing_definition": "clean source index absent from faulty_source_ids",
            "candidate_definition": "union of 3D metric balls around radar returns with sufficient local radar neighbors",
            "annotations_and_clean_used_for_candidates": False,
        },
        "estimated_support_volume": [
            {"support_radius_m": radius,
             "mean_fraction": float(np.mean(volume_fractions[radius])),
             "median_fraction": float(np.median(volume_fractions[radius]))}
            for radius in radii
        ],
        "groups": [],
    }
    for (radius, fault, name), counts in sorted(totals.items()):
        count = counts["missing"]
        entry = {
            "support_radius_m": radius, "fault": fault, "class": name,
            "missing_returns": count, "supported_missing_returns": counts["supported_missing"],
            "within_region_fraction": counts["supported_missing"] / count if count else None,
        }
        if name in CLASSES[:3]:
            entry["objects_with_missing"] = counts["objects_with_missing"]
            entry["objects_with_supported_missing"] = counts["objects_with_supported_missing"]
        summary["groups"].append(entry)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'} and per_frame.csv")


if __name__ == "__main__":
    main()
