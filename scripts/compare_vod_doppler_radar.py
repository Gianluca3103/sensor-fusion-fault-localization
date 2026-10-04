"""Compare VoD radar variants against target-time 3D annotation boxes.

Counts measure radar concentration, not object detection AP or correct LiDAR
surface recovery. A return inside a box can still be multipath or clutter.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from Fault_Localization_Model.vod_dataset.doppler_accumulation import (
    radial_compensate_verified_stack,
)
from Fault_Localization_Model.vod_dataset.vod_io import (
    _named_transform, load_vod_radar, load_vod_split_ids,
    resolve_vod_public_root, vod_partition_for_split,
)


VARIANTS = {
    "ego": "radar_20frames_verified",
    "old_gate": "radar_20frames_verified_motion_aware",
    "radial": "radar_20frames_verified_doppler_radial",
    "radial_window": "radar_20frames_verified_doppler_window",
}


def _target_box_masks(public: Path, partition: str, frame_id: str,
                      radar_points: np.ndarray) -> dict[str, np.ndarray]:
    transform = _named_transform(
        public / "radar" / partition / "calib" / f"{frame_id}.txt"
    )
    camera = radar_points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    result = {name: np.zeros(len(camera), dtype=bool)
              for name in ("Cyclist", "Car", "Pedestrian")}
    labels = public / "lidar" / partition / "label_2" / f"{frame_id}.txt"
    for line in labels.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        name = "Cyclist" if fields[0] in {"Cyclist", "bicycle", "rider"} else fields[0]
        if name not in result:
            continue
        height, width, length, x, y, z, yaw = map(float, fields[8:15])
        delta = camera - np.asarray((x, y, z))
        local_x = np.cos(yaw) * delta[:, 0] - np.sin(yaw) * delta[:, 2]
        local_z = np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 2]
        result[name] |= (
            (np.abs(local_x) <= width / 2)
            & (np.abs(local_z) <= length / 2)
            & (delta[:, 1] >= -height)
            & (delta[:, 1] <= 0)
        )
    return result


def _metrics(public: Path, partition: str, frame_id: str,
             condition: str, points: np.ndarray) -> dict[str, int | str]:
    old = points[:, 6] < 0
    fast_old = old & (np.abs(points[:, 5]) >= 1)
    masks = _target_box_masks(public, partition, frame_id, points)
    row: dict[str, int | str] = {
        "frame_id": frame_id,
        "condition": condition,
        "points": len(points),
        "old_points": int(old.sum()),
        "fast_old_points": int(fast_old.sum()),
    }
    for name, mask in masks.items():
        row[f"{name.lower()}_box_points"] = int(mask.sum())
        row[f"{name.lower()}_box_old_points"] = int((mask & old).sum())
        row[f"{name.lower()}_box_fast_old_points"] = int((mask & fast_old).sum())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--frame-ids", nargs="+",
                        help="Official split IDs to compare; default is the full split")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", choices=tuple(VARIANTS),
                        default=tuple(VARIANTS),
                        help="Compare only the listed available radar variants")
    parser.add_argument("--frame-period-s", type=float, default=0.10022)
    parser.add_argument("--opposite-sign-diagnostic", action="store_true")
    args = parser.parse_args()
    public = resolve_vod_public_root(args.vod_root)
    official_ids = load_vod_split_ids(public, args.split)
    frame_ids = args.frame_ids or official_ids
    if not set(frame_ids) <= set(official_ids):
        raise ValueError("Requested frames must belong to the official split")
    partition = vod_partition_for_split(public, args.split, frame_ids)
    rows = []
    for frame_id in frame_ids:
        for condition in dict.fromkeys(args.conditions):
            variant = VARIANTS[condition]
            path = public / variant / partition / "velodyne" / f"{frame_id}.bin"
            if not path.is_file():
                raise FileNotFoundError(path)
            points = load_vod_radar(path)
            rows.append(_metrics(public, partition, frame_id, condition, points))
        if args.opposite_sign_diagnostic:
            verified = load_vod_radar(
                public / VARIANTS["ego"] / partition / "velodyne" / f"{frame_id}.bin"
            )
            ages = sorted(set(verified[:, 6].astype(int)))
            source = {
                age: load_vod_radar(
                    public / "radar" / partition / "velodyne" /
                    f"{int(frame_id) + age:05d}.bin"
                ) for age in ages
            }
            reversed_points, _ = radial_compensate_verified_stack(
                verified, source, frame_period_s=args.frame_period_s,
                doppler_sign=-1, dispersion_tolerance_m=None,
            )
            rows.append(_metrics(public, partition, frame_id,
                                 "opposite_radial_sign", reversed_points))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_condition = {}
    for row in rows:
        sums = by_condition.setdefault(row["condition"],
                                       {key: 0 for key in row if key not in ("frame_id", "condition")})
        for key in sums:
            sums[key] += int(row[key])
    for condition, sums in by_condition.items():
        print(condition, sums)
    print(args.output)


if __name__ == "__main__":
    main()
