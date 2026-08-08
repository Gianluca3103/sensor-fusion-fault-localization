from __future__ import annotations

from bisect import bisect_left
import math

import numpy as np

from PFS_Radar_v2.radar_types import RadarAlignmentUnavailableError


def nearest_pose_index(timestamps: tuple[int, ...], timestamp: int) -> int:
    insertion = bisect_left(timestamps, timestamp)
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(timestamps)
    ]
    if not candidates:
        raise RadarAlignmentUnavailableError(f"No pose is available for {timestamp}")
    return min(candidates, key=lambda index: abs(timestamps[index] - timestamp))


def pose_velocity(
    pose_timestamps: tuple[int, ...],
    poses: np.ndarray,
    timestamp: int,
) -> tuple[np.ndarray, float]:
    """Estimate sensor velocity in its local axes and ego yaw rate."""

    center = nearest_pose_index(pose_timestamps, timestamp)
    left = max(0, center - 1)
    right = min(len(pose_timestamps) - 1, center + 1)
    if left == right:
        return np.zeros(3, dtype=np.float64), 0.0
    dt_s = (pose_timestamps[right] - pose_timestamps[left]) / 1_000_000_000.0
    if dt_s <= 0.0:
        return np.zeros(3, dtype=np.float64), 0.0
    velocity_world = (poses[right, :3, 3] - poses[left, :3, 3]) / dt_s
    velocity_sensor = poses[center, :3, :3].T @ velocity_world
    relative_rotation = poses[left, :3, :3].T @ poses[right, :3, :3]
    yaw_delta = math.atan2(relative_rotation[1, 0], relative_rotation[0, 0])
    return velocity_sensor, yaw_delta / dt_s
