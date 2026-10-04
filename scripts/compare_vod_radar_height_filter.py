"""Measure the observed-LiDAR height gate on VoD radar and fault samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar, discover_vod_frames, load_vod_radar,
    load_vod_radar_to_lidar, load_vod_split_ids, resolve_vod_public_root,
)
from models.two_stage_reconstruction_head.cross_modal_data import (
    observed_lidar_height_mask,
)
from scripts.compare_vod_doppler_radar import _target_box_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--fault-samples-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--radar-variant",
                        default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = resolve_vod_public_root(args.vod_root)
    ids = load_vod_split_ids(public, args.split)
    frames = discover_vod_frames(public, args.split,
                                 radar_variant=args.radar_variant,
                                 frame_ids=ids)
    rows = []
    for index, frame in enumerate(frames, 1):
        matches = list((args.fault_samples_root / args.split).glob(
            f"{frame.frame_id}_*.npz"
        ))
        if len(matches) != 1:
            raise ValueError(f"Expected one faulty sample for {frame.frame_id}: {matches}")
        with np.load(matches[0], allow_pickle=False) as archive:
            observed = np.asarray(archive["faulty_lidar_points"][:, :4],
                                  dtype=np.float32)
        raw = load_vod_radar(frame.radar_path)
        transform = load_vod_radar_to_lidar(
            frame.lidar_calibration_path, frame.radar_calibration_path,
        )
        radar_lidar = align_radar_to_lidar(raw, transform)
        keep = observed_lidar_height_mask(radar_lidar, observed)
        below = int(np.count_nonzero(radar_lidar[:, 2] < observed[:, 2].min())) if len(observed) >= 2 else 0
        above = int(np.count_nonzero(radar_lidar[:, 2] > observed[:, 2].max())) if len(observed) >= 2 else 0
        boxes = _target_box_masks(public, frame.lidar_path.parent.parent.name,
                                  frame.frame_id, raw)
        row: dict[str, int | float | str] = {
            "frame_id": frame.frame_id,
            "fault_sample": str(matches[0]),
            "observed_lidar_points": len(observed),
            "lidar_z_min": float(observed[:, 2].min()) if len(observed) else float("nan"),
            "lidar_z_max": float(observed[:, 2].max()) if len(observed) else float("nan"),
            "radar_before": len(raw),
            "radar_after": int(keep.sum()),
            "removed_below": below,
            "removed_above": above,
            "removed_current": int(np.count_nonzero(~keep & (raw[:, 6] == 0))),
            "removed_history": int(np.count_nonzero(~keep & (raw[:, 6] < 0))),
        }
        for name, inside in boxes.items():
            row[f"{name.lower()}_before"] = int(inside.sum())
            row[f"{name.lower()}_after"] = int(np.count_nonzero(inside & keep))
        rows.append(row)
        if index % 250 == 0 or index == len(frames):
            print(f"Measured {index}/{len(frames)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ("radar_before", "radar_after", "removed_below", "removed_above",
                "removed_current", "removed_history", "cyclist_before",
                "cyclist_after", "car_before", "car_after",
                "pedestrian_before", "pedestrian_after"):
        print(f"{key}: {sum(int(row[key]) for row in rows):,}")
    print(args.output)


if __name__ == "__main__":
    main()
