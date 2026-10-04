"""Plot four VoD radar stacks with the same current-time 3D box outlines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    _named_transform, load_vod_radar, resolve_vod_public_root,
)
from scripts.compare_vod_doppler_radar import VARIANTS, _target_box_masks


def _boxes_in_radar(public: Path, partition: str, frame_id: str):
    camera_from_radar = _named_transform(
        public / "radar" / partition / "calib" / f"{frame_id}.txt"
    )
    radar_from_camera = np.linalg.inv(camera_from_radar)
    labels = public / "lidar" / partition / "label_2" / f"{frame_id}.txt"
    for line in labels.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        name = "Cyclist" if fields[0] in {"Cyclist", "bicycle", "rider"} else fields[0]
        if name not in {"Cyclist", "Pedestrian", "Car"}:
            continue
        _, width, length, x, y, z, yaw = map(float, fields[8:15])
        corners = np.array([
            [-width / 2, -length / 2], [width / 2, -length / 2],
            [width / 2, length / 2], [-width / 2, length / 2],
            [-width / 2, -length / 2],
        ])
        camera = np.column_stack((
            x + np.cos(yaw) * corners[:, 0] + np.sin(yaw) * corners[:, 1],
            np.full(len(corners), y),
            z - np.sin(yaw) * corners[:, 0] + np.cos(yaw) * corners[:, 1],
        ))
        radar = camera @ radar_from_camera[:3, :3].T + radar_from_camera[:3, 3]
        yield name, radar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--partition", choices=("training", "testing"), default="training")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    public = resolve_vod_public_root(args.vod_root)
    labels = {
        "ego": "Ego alignment",
        "old_gate": "Old velocity-age filter",
        "radial": "Doppler radial shift",
        "radial_window": "Radial shift + age window",
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.16, top=0.82, wspace=0.08)
    for axis, (condition, variant) in zip(axes, VARIANTS.items()):
        points = load_vod_radar(
            public / variant / args.partition / "velodyne" / f"{args.frame_id}.bin"
        )
        old = points[:, 6] < 0
        moving = old & (np.abs(points[:, 5]) >= 1)
        current = ~old
        slow_old = old & ~moving
        axis.scatter(points[slow_old, 0], points[slow_old, 1], s=1.0,
                     color="#9ca3af", alpha=0.18, rasterized=True)
        axis.scatter(points[moving, 0], points[moving, 1], s=2.5,
                     color="#e38a22", alpha=0.62, rasterized=True)
        axis.scatter(points[current, 0], points[current, 1], s=3.0,
                     color="#1976d2", alpha=0.7, rasterized=True)
        for name, radar in _boxes_in_radar(public, args.partition, args.frame_id):
            color = {"Cyclist": "#19a65b", "Pedestrian": "#d43786", "Car": "#7a31bf"}[name]
            axis.plot(radar[:, 0], radar[:, 1], color=color, linewidth=1.3)
        cyclist_count = int(_target_box_masks(
            public, args.partition, args.frame_id, points,
        )["Cyclist"].sum())
        axis.set_title(f"{labels[condition]}\n{len(points):,} returns; "
                       f"{cyclist_count} in cyclist boxes", fontsize=11)
        axis.set_xlim(0, 40)
        axis.set_ylim(-23, 23)
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
        axis.set_xlabel("Radar forward x (m)")
    axes[0].set_ylabel("Radar lateral y (m)")
    fig.suptitle(f"VoD {args.frame_id}: radar motion alignment", y=0.98, fontsize=14)
    fig.text(0.5, 0.90, "Orange = older high-Doppler radar; blue = current radar; "
             "green = cyclist boxes", ha="center", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
