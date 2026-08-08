from __future__ import annotations

from bisect import bisect_left, bisect_right
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from PFS_Radar_v2.radar_types import (
    AdaptiveStackConfig,
    RadarAlignmentUnavailableError,
)


def rotation_angle_rad(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def interpolated_pose_from_sequence(
    timestamps: tuple[int, ...],
    poses: np.ndarray,
    timestamp: int,
    max_extrapolation_delta_ms: float = 20.0,
) -> tuple[np.ndarray, float]:
    """Estimate pose at timestamp with linear translation and rotational SLERP."""

    if max_extrapolation_delta_ms < 0.0:
        raise ValueError("max_extrapolation_delta_ms must be non-negative")
    if not timestamps:
        raise RadarAlignmentUnavailableError(f"No pose is available for {timestamp}")
    pose_array = np.asarray(poses, dtype=np.float64)
    if pose_array.shape != (len(timestamps), 4, 4):
        raise ValueError(
            f"poses must have shape [{len(timestamps)},4,4], got {pose_array.shape}"
        )

    target = int(timestamp)
    insertion = bisect_left(timestamps, target)
    if insertion < len(timestamps) and timestamps[insertion] == target:
        return pose_array[insertion].copy(), 0.0

    if 0 < insertion < len(timestamps):
        left_time = timestamps[insertion - 1]
        right_time = timestamps[insertion]
        span = right_time - left_time
        if span <= 0:
            raise ValueError("pose timestamps must be strictly increasing")
        alpha = (target - left_time) / span
        left_pose = pose_array[insertion - 1]
        right_pose = pose_array[insertion]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (1.0 - alpha) * left_pose[:3, 3] + alpha * right_pose[:3, 3]
        transform[:3, :3] = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix([left_pose[:3, :3], right_pose[:3, :3]]),
        )([alpha]).as_matrix()[0]
        return transform, 0.0

    nearest_index = 0 if insertion == 0 else len(timestamps) - 1
    delta_ms = (timestamps[nearest_index] - target) / 1_000_000.0
    if abs(delta_ms) > max_extrapolation_delta_ms:
        raise RadarAlignmentUnavailableError(
            f"Nearest pose is {abs(delta_ms):.1f} ms from timestamp {timestamp}"
        )
    return pose_array[nearest_index].copy(), delta_ms


def _frame_weight(
    age_s: float,
    translation_m: float,
    rotation_deg: float,
    config: AdaptiveStackConfig,
) -> float:
    exponent = (
        -age_s / config.weight_time_s
        - translation_m / config.weight_translation_m
        - rotation_deg / config.weight_rotation_deg
    )
    return float(np.exp(exponent))


def select_adaptive_indices(
    radar_timestamps: tuple[int, ...] | list[int],
    radar_poses: np.ndarray,
    lidar_timestamp: int,
    config: AdaptiveStackConfig,
    reference_pose: np.ndarray | None = None,
) -> list[dict]:
    """Select the longest newest-first causal history satisfying all pose gates."""

    config.validate()
    timestamps = tuple(int(value) for value in radar_timestamps)
    poses = np.asarray(radar_poses, dtype=np.float64)
    if poses.shape != (len(timestamps), 4, 4):
        raise ValueError(
            f"radar_poses must have shape [{len(timestamps)},4,4], got {poses.shape}"
        )
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("radar_timestamps must be strictly increasing")
    current_index = bisect_right(timestamps, int(lidar_timestamp)) - 1
    if current_index < 0:
        raise RadarAlignmentUnavailableError(
            f"No causal radar frame exists at or before LiDAR timestamp {lidar_timestamp}"
        )

    if reference_pose is None:
        reference_pose = poses[current_index]
    reference_pose = np.asarray(reference_pose, dtype=np.float64)
    if reference_pose.shape != (4, 4) or not np.isfinite(reference_pose).all():
        raise ValueError("reference_pose must be a finite 4x4 matrix")
    reference_from_world = np.linalg.inv(reference_pose)
    rows: list[dict] = []
    for index in range(current_index, -1, -1):
        if config.max_frames is not None and len(rows) >= config.max_frames:
            break
        age_s = (int(lidar_timestamp) - timestamps[index]) / 1_000_000_000.0
        relative = reference_from_world @ poses[index]
        translation_m = float(np.linalg.norm(relative[:3, 3]))
        rotation_deg = math.degrees(rotation_angle_rad(relative[:3, :3]))
        if (
            age_s > config.max_age_s
            or translation_m > config.max_translation_m
            or rotation_deg > config.max_rotation_deg
        ):
            break
        rows.append(
            {
                "index": index,
                "timestamp": timestamps[index],
                "age_s": age_s,
                "translation_m": translation_m,
                "rotation_deg": rotation_deg,
                "weight": _frame_weight(
                    age_s, translation_m, rotation_deg, config
                ),
            }
        )
    rows.reverse()
    return rows


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
    imu_from_radar_rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate radar-origin velocity in radar axes plus ego yaw rate."""

    center = nearest_pose_index(pose_timestamps, timestamp)
    left = max(0, center - 1)
    right = min(len(pose_timestamps) - 1, center + 1)
    if left == right:
        return np.zeros(3, dtype=np.float64), 0.0
    dt_s = (pose_timestamps[right] - pose_timestamps[left]) / 1_000_000_000.0
    if dt_s <= 0.0:
        return np.zeros(3, dtype=np.float64), 0.0
    velocity_world = (poses[right, :3, 3] - poses[left, :3, 3]) / dt_s
    world_from_radar_rotation = poses[center, :3, :3] @ imu_from_radar_rotation
    velocity_radar = world_from_radar_rotation.T @ velocity_world
    relative_rotation = poses[left, :3, :3].T @ poses[right, :3, :3]
    yaw_delta = math.atan2(relative_rotation[1, 0], relative_rotation[0, 0])
    return velocity_radar, yaw_delta / dt_s
