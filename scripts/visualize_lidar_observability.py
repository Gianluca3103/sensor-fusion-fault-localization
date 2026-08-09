"""Render clean K-Radar LiDAR occupancy and deterministic observability maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.bev_utils import HEIGHT_RANGE_M, project_lidar_bev
from Fault_Localization_Model.data_injection_utils import filter_pointcloud
from Fault_Localization_Model.kradar_dataset import (
    load_radar_from_lidar_transform,
    radar_overlap_mask,
    read_kradar_lidar_pcd,
)
from Fault_Localization_Model.lidar_observability import (
    LIDAR_SENSOR_ORIGIN,
    create_observability_map,
    save_observability_debug_figure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-pcd", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=64.0)
    parser.add_argument("--y-min", type=float, default=-32.0)
    parser.add_argument("--y-max", type=float, default=32.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--min-range", type=float, default=1.0)
    parser.add_argument("--max-range", type=float, default=118.037109375)
    parser.add_argument("--num-z-bins", type=int, default=16)
    parser.add_argument("--ray-support-tau", type=float, default=4.0)
    args = parser.parse_args()

    raw = read_kradar_lidar_pcd(args.lidar_pcd)
    _, range_mask = filter_pointcloud(
        raw, args.min_range, args.max_range, return_mask=True
    )
    radar_from_lidar = load_radar_from_lidar_transform(args.calibration)
    points = raw[
        range_mask & radar_overlap_mask(raw, radar_from_lidar),
        :4,
    ]
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    clean = project_lidar_bev(points, x_range, y_range, args.resolution)
    observations = create_observability_map(
        points,
        LIDAR_SENSOR_ORIGIN,
        x_range,
        y_range,
        args.resolution,
        z_range=HEIGHT_RANGE_M,
        num_z_bins=args.num_z_bins,
        ray_support_tau=args.ray_support_tau,
    )
    save_observability_debug_figure(
        args.output,
        clean["occupancy"],
        observations,
    )
    print(f"Filtered clean points: {len(points)}")
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
