from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez_compressed
from Fault_Localization_Model.sample_utils import load_sample_metadata
from PFS_Radar.radar_data import (
    RadarAlignmentUnavailableError,
    find_named_file as _find_named_file,
    load_ground_truth_poses,
    load_lidar_to_radar_transform as _load_lidar_to_radar_transform,
    load_named_transform as _load_named_transform,
    nearest_ground_truth_pose,
    radar_cache_path,
    read_continental_bin as _read_continental_bin,
    scene_name_from_metadata,
    scene_radar_resources,
    scene_session_root,
    session_name_from_metadata,
    transform_xyz,
)


RADAR_CACHE_VERSION = 4
POLICY_NAME = "adaptive_pose_doppler_tracking_v2"
CHANNELS = [
    "static_occupancy",
    "support_normalized_static_density",
    "tracked_dynamic_speed",
    "dynamic_track_occupancy",
]
SCENE_RESOURCE_NAMES = {
    "continental_gt.txt",
    "aeva_gt.txt",
    "imu_lidar.txt",
    "continental_lidar.txt",
}


@lru_cache(maxsize=256)
def _scene_resource_index(root: Path) -> dict[str, Path]:
    """Discover all Radar v2 scene resources in one recursive traversal."""

    candidates: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        lowered = path.name.lower()
        if lowered in SCENE_RESOURCE_NAMES and path.is_file():
            candidates.setdefault(lowered, []).append(path)
    return {
        name: min(paths, key=lambda path: len(path.parts))
        for name, paths in candidates.items()
    }


def find_named_file(root: Path, name: str) -> Path:
    """Reuse a single scene-resource index within each worker."""

    lowered = name.lower()
    if lowered not in SCENE_RESOURCE_NAMES:
        return _find_named_file(root, name)
    try:
        return _scene_resource_index(root)[lowered]
    except KeyError as exc:
        raise FileNotFoundError(f"No {name} under {root}") from exc


@lru_cache(maxsize=256)
def load_named_transform(path: Path, key: str) -> np.ndarray:
    """Cache immutable calibration matrices within each worker."""

    return _load_named_transform(path, key)


@lru_cache(maxsize=128)
def load_lidar_to_radar_transform(path: Path) -> np.ndarray:
    """Cache the immutable Continental-to-LiDAR calibration."""

    return _load_lidar_to_radar_transform(path)


@lru_cache(maxsize=256)
def read_continental_bin(path: Path) -> np.ndarray:
    """Reuse raw radar frames across overlapping chronological windows."""

    return _read_continental_bin(path)


@dataclass(frozen=True)
class AdaptiveStackConfig:
    """Pose gates and soft weights for causal radar-frame accumulation."""

    max_frames: int | None = None
    max_age_s: float = 1.0
    max_translation_m: float = 4.0
    max_rotation_deg: float = 5.0
    weight_time_s: float = 0.5
    weight_translation_m: float = 2.0
    weight_rotation_deg: float = 3.0

    def validate(self) -> None:
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be None or at least 1")
        positive = {
            "max_age_s": self.max_age_s,
            "max_translation_m": self.max_translation_m,
            "max_rotation_deg": self.max_rotation_deg,
            "weight_time_s": self.weight_time_s,
            "weight_translation_m": self.weight_translation_m,
            "weight_rotation_deg": self.weight_rotation_deg,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DopplerTrackingConfig:
    """Ego-Doppler compensation, clustering, and short-window tracking."""

    dynamic_threshold_mps: float = 1.0
    doppler_sign: str = "auto"
    sign_inference_min_speed_mps: float = 0.5
    cluster_eps_m: float = 1.2
    cluster_min_samples: int = 2
    association_distance_m: float = 3.0
    min_track_hits: int = 2
    velocity_smoothing: float = 0.5
    max_abs_velocity_mps: float = 30.0

    def validate(self) -> None:
        if self.doppler_sign not in {"auto", "1", "-1"}:
            raise ValueError("doppler_sign must be one of: auto, 1, -1")
        positive = {
            "dynamic_threshold_mps": self.dynamic_threshold_mps,
            "sign_inference_min_speed_mps": self.sign_inference_min_speed_mps,
            "cluster_eps_m": self.cluster_eps_m,
            "association_distance_m": self.association_distance_m,
            "max_abs_velocity_mps": self.max_abs_velocity_mps,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.cluster_min_samples < 1:
            raise ValueError("cluster_min_samples must be at least 1")
        if self.min_track_hits < 1:
            raise ValueError("min_track_hits must be at least 1")
        if not 0.0 <= self.velocity_smoothing < 1.0:
            raise ValueError("velocity_smoothing must lie in [0, 1)")


@dataclass
class SelectedFrame:
    path: Path
    timestamp: int
    pose: np.ndarray
    age_s: float
    translation_m: float
    rotation_deg: float
    weight: float
    pose_delta_ms: float


@dataclass
class ProcessedFrame:
    timestamp: int
    points: np.ndarray
    doppler_residual_mps: np.ndarray
    dynamic_mask: np.ndarray
    cluster_labels: np.ndarray
    weight: float
    doppler_sign: int
    sensor_speed_mps: float
    yaw_rate_dps: float


@dataclass
class ClusterObservation:
    frame_index: int
    timestamp: int
    label: int
    point_indices: np.ndarray
    centroid: np.ndarray
    doppler_velocity: np.ndarray
    track_id: int = -1


@dataclass
class TrackState:
    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    last_timestamp: int
    hits: int


def rotation_angle_rad(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


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

    reference_pose = poses[current_index]
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


def _nearest_pose_index(timestamps: tuple[int, ...], timestamp: int) -> int:
    insertion = bisect_left(timestamps, timestamp)
    candidates = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < len(timestamps)
    ]
    if not candidates:
        raise RadarAlignmentUnavailableError(f"No pose is available for {timestamp}")
    return min(candidates, key=lambda index: abs(timestamps[index] - timestamp))


def _pose_velocity(
    pose_timestamps: tuple[int, ...],
    poses: np.ndarray,
    timestamp: int,
    imu_from_radar_rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate radar-origin velocity in radar axes plus ego yaw rate."""

    center = _nearest_pose_index(pose_timestamps, timestamp)
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


def compensate_doppler(
    points: np.ndarray,
    sensor_velocity: np.ndarray,
    doppler_sign: str = "auto",
    sign_inference_min_speed_mps: float = 0.5,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Subtract the radial Doppler expected from ego motion.

    Returns residual velocity, the applied raw-measurement sign, and expected
    static radial velocity. Auto sign selection assumes static detections are
    the majority and chooses the convention with the lower median residual.
    """

    points = np.asarray(points, dtype=np.float64)
    sensor_velocity = np.asarray(sensor_velocity, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"points must have shape [N,>=4], got {points.shape}")
    if sensor_velocity.shape != (3,) or not np.isfinite(sensor_velocity).all():
        raise ValueError("sensor_velocity must be a finite 3-vector")
    if doppler_sign not in {"auto", "1", "-1"}:
        raise ValueError("doppler_sign must be one of: auto, 1, -1")
    ranges = np.linalg.norm(points[:, :3], axis=1)
    valid = ranges > 1e-6
    line_of_sight = np.zeros((len(points), 3), dtype=np.float64)
    line_of_sight[valid] = points[valid, :3] / ranges[valid, None]
    expected_static = -(line_of_sight @ sensor_velocity)
    measured = points[:, 3]

    if doppler_sign == "auto":
        if np.linalg.norm(sensor_velocity) < sign_inference_min_speed_mps or not np.any(valid):
            sign = 1
        else:
            positive_score = float(
                np.median(np.abs(measured[valid] - expected_static[valid]))
            )
            negative_score = float(
                np.median(np.abs(-measured[valid] - expected_static[valid]))
            )
            sign = 1 if positive_score <= negative_score else -1
    else:
        sign = int(doppler_sign)
    residual = sign * measured - expected_static
    residual[~valid] = 0.0
    return residual.astype(np.float32), sign, expected_static.astype(np.float32)


def dbscan_labels(
    xy: np.ndarray,
    candidate_mask: np.ndarray,
    eps_m: float,
    min_samples: int,
) -> np.ndarray:
    """Small dependency-free 2-D DBSCAN for sparse radar detections."""

    xy = np.asarray(xy, dtype=np.float64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape [N,2], got {xy.shape}")
    if candidate_mask.shape != (len(xy),):
        raise ValueError("candidate_mask must contain one value per point")
    if eps_m <= 0.0 or min_samples < 1:
        raise ValueError("eps_m must be positive and min_samples at least 1")

    labels = np.full(len(xy), -1, dtype=np.int32)
    candidates = np.flatnonzero(candidate_mask & np.isfinite(xy).all(axis=1))
    if not len(candidates):
        return labels
    cells: dict[tuple[int, int], list[int]] = {}
    point_cells: dict[int, tuple[int, int]] = {}
    for index in candidates:
        cell = tuple(np.floor(xy[index] / eps_m).astype(np.int64))
        point_cells[int(index)] = cell
        cells.setdefault(cell, []).append(int(index))

    eps_squared = eps_m * eps_m

    def neighbors(index: int) -> list[int]:
        cell = point_cells[index]
        nearby: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    if float(np.sum((xy[index] - xy[other]) ** 2)) <= eps_squared:
                        nearby.append(other)
        return nearby

    visited: set[int] = set()
    cluster_id = 0
    for seed in candidates:
        seed = int(seed)
        if seed in visited:
            continue
        visited.add(seed)
        seed_neighbors = neighbors(seed)
        if len(seed_neighbors) < min_samples:
            continue
        labels[seed] = cluster_id
        queue = list(seed_neighbors)
        queued = set(queue)
        cursor = 0
        while cursor < len(queue):
            point = queue[cursor]
            cursor += 1
            if point not in visited:
                visited.add(point)
                point_neighbors = neighbors(point)
                if len(point_neighbors) >= min_samples:
                    for neighbor in point_neighbors:
                        if neighbor not in queued:
                            queued.add(neighbor)
                            queue.append(neighbor)
            if labels[point] < 0:
                labels[point] = cluster_id
        cluster_id += 1
    return labels


def estimate_planar_doppler_velocity(
    radar_points: np.ndarray,
    residual_mps: np.ndarray,
    radar_to_current_rotation: np.ndarray,
) -> np.ndarray:
    radar_points = np.asarray(radar_points, dtype=np.float64)
    residual_mps = np.asarray(residual_mps, dtype=np.float64)
    if len(radar_points) == 0:
        return np.zeros(2, dtype=np.float64)
    planar_range = np.linalg.norm(radar_points[:, :2], axis=1)
    valid = planar_range > 1e-6
    if not np.any(valid):
        return np.zeros(2, dtype=np.float64)
    design = radar_points[valid, :2] / planar_range[valid, None]
    target = residual_mps[valid]
    regularization = 0.05 * np.eye(2, dtype=np.float64)
    velocity_radar = np.linalg.solve(
        design.T @ design + regularization,
        design.T @ target,
    )
    rotation = np.asarray(radar_to_current_rotation, dtype=np.float64)
    velocity_current_3d = rotation @ np.asarray(
        [velocity_radar[0], velocity_radar[1], 0.0]
    )
    return velocity_current_3d[:2]


def _make_observations(
    frames: list[ProcessedFrame],
    raw_frames: list[np.ndarray],
    radar_to_current_rotations: list[np.ndarray],
) -> list[list[ClusterObservation]]:
    observations: list[list[ClusterObservation]] = []
    for frame_index, frame in enumerate(frames):
        frame_observations: list[ClusterObservation] = []
        for label in sorted(set(frame.cluster_labels.tolist()) - {-1}):
            indices = np.flatnonzero(frame.cluster_labels == label)
            frame_observations.append(
                ClusterObservation(
                    frame_index=frame_index,
                    timestamp=frame.timestamp,
                    label=int(label),
                    point_indices=indices,
                    centroid=np.mean(frame.points[indices, :2], axis=0).astype(
                        np.float64
                    ),
                    doppler_velocity=estimate_planar_doppler_velocity(
                        raw_frames[frame_index][indices],
                        frame.doppler_residual_mps[indices],
                        radar_to_current_rotations[frame_index],
                    ),
                )
            )
        observations.append(frame_observations)
    return observations


def associate_tracks(
    observations: list[list[ClusterObservation]],
    association_distance_m: float,
    velocity_smoothing: float,
) -> dict[int, TrackState]:
    """Associate cluster centroids chronologically with a predictive greedy match."""

    tracks: dict[int, TrackState] = {}
    next_track_id = 0
    for frame_observations in observations:
        if not frame_observations:
            continue
        timestamp = frame_observations[0].timestamp
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in tracks.items():
            dt_s = (timestamp - track.last_timestamp) / 1_000_000_000.0
            if dt_s <= 0.0:
                continue
            predicted = track.position + track.velocity * dt_s
            for observation_index, observation in enumerate(frame_observations):
                distance = float(np.linalg.norm(observation.centroid - predicted))
                if distance <= association_distance_m:
                    candidate_pairs.append((distance, track_id, observation_index))

        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        for _, track_id, observation_index in sorted(candidate_pairs):
            if (
                track_id in matched_tracks
                or observation_index in matched_observations
            ):
                continue
            track = tracks[track_id]
            observation = frame_observations[observation_index]
            dt_s = (observation.timestamp - track.last_timestamp) / 1_000_000_000.0
            centroid_velocity = (
                (observation.centroid - track.position) / dt_s
                if dt_s > 1e-6
                else observation.doppler_velocity
            )
            measurement_velocity = (
                0.5 * centroid_velocity + 0.5 * observation.doppler_velocity
            )
            track.velocity = (
                velocity_smoothing * track.velocity
                + (1.0 - velocity_smoothing) * measurement_velocity
            )
            track.position = observation.centroid.copy()
            track.last_timestamp = observation.timestamp
            track.hits += 1
            observation.track_id = track_id
            matched_tracks.add(track_id)
            matched_observations.add(observation_index)

        for observation_index, observation in enumerate(frame_observations):
            if observation_index in matched_observations:
                continue
            track_id = next_track_id
            next_track_id += 1
            observation.track_id = track_id
            tracks[track_id] = TrackState(
                track_id=track_id,
                position=observation.centroid.copy(),
                velocity=observation.doppler_velocity.copy(),
                last_timestamp=observation.timestamp,
                hits=1,
            )
    return tracks


def _grid_indices(
    xyz: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    width = int(np.ceil((y_range[1] - y_range[0]) / resolution))
    valid = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= x_range[0])
        & (xyz[:, 0] < x_range[1])
        & (xyz[:, 1] >= y_range[0])
        & (xyz[:, 1] < y_range[1])
    )
    cols = np.floor((xyz[:, 1] - y_range[0]) / resolution).astype(np.int32)
    rows_from_bottom = np.floor(
        (xyz[:, 0] - x_range[0]) / resolution
    ).astype(np.int32)
    rows = height - 1 - rows_from_bottom
    valid &= (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows, cols, valid


def project_adaptive_bev(
    frames: list[ProcessedFrame],
    observations: list[list[ClusterObservation]],
    tracks: dict[int, TrackState],
    lidar_timestamp: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    tracking_config: DopplerTrackingConfig,
) -> np.ndarray:
    tracking_config.validate()
    height = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    width = int(np.ceil((y_range[1] - y_range[0]) / resolution))
    output = np.zeros((4, height, width), dtype=np.float32)
    density = np.zeros((height, width), dtype=np.float32)
    effective_support = sum(frame.weight for frame in frames)

    for frame in frames:
        static_points = frame.points[~frame.dynamic_mask, :3]
        if len(static_points):
            rows, cols, valid = _grid_indices(
                static_points, x_range, y_range, resolution
            )
            rows, cols = rows[valid], cols[valid]
            output[0, rows, cols] = 1.0
            np.add.at(density, (rows, cols), frame.weight)

    if effective_support > 0.0:
        normalized_density = density / effective_support
        output[1] = np.clip(
            np.log1p(normalized_density) / np.log(2.0), 0.0, 1.0
        )

    for frame_index, frame_observations in enumerate(observations):
        frame = frames[frame_index]
        assigned = np.zeros(len(frame.points), dtype=bool)
        for observation in frame_observations:
            track = tracks[observation.track_id]
            confirmed = track.hits >= tracking_config.min_track_hits
            age_s = (lidar_timestamp - frame.timestamp) / 1_000_000_000.0
            points = frame.points[observation.point_indices, :3].copy()
            if confirmed:
                points[:, :2] += track.velocity[None, :] * age_s
                speed = float(np.linalg.norm(track.velocity))
                confidence = min(1.0, track.hits / tracking_config.min_track_hits)
            else:
                speed = float(
                    np.max(np.abs(frame.doppler_residual_mps[observation.point_indices]))
                )
                confidence = 0.5
            rows, cols, valid = _grid_indices(points, x_range, y_range, resolution)
            rows, cols = rows[valid], cols[valid]
            np.maximum.at(
                output[2],
                (rows, cols),
                np.float32(
                    np.clip(
                        speed / tracking_config.max_abs_velocity_mps, 0.0, 1.0
                    )
                ),
            )
            np.maximum.at(
                output[3], (rows, cols), np.float32(confidence)
            )
            assigned[observation.point_indices] = True

        unclustered = frame.dynamic_mask & ~assigned
        if np.any(unclustered):
            points = frame.points[unclustered, :3]
            residual = np.abs(frame.doppler_residual_mps[unclustered])
            rows, cols, valid = _grid_indices(points, x_range, y_range, resolution)
            rows, cols = rows[valid], cols[valid]
            np.maximum.at(
                output[2],
                (rows, cols),
                np.clip(
                    residual[valid] / tracking_config.max_abs_velocity_mps,
                    0.0,
                    1.0,
                ),
            )
            np.maximum.at(output[3], (rows, cols), np.float32(0.25))
    return output


def _select_scene_frames(
    scene_root: Path,
    lidar_timestamp: int,
    max_delta_ms: float,
    config: AdaptiveStackConfig,
) -> tuple[list[SelectedFrame], float]:
    timestamps, path_texts, _ = scene_radar_resources(str(scene_root.resolve()))
    current_index = bisect_right(timestamps, lidar_timestamp) - 1
    if current_index < 0:
        raise RadarAlignmentUnavailableError(
            f"No causal radar frame exists at or before LiDAR timestamp {lidar_timestamp}"
        )
    delta_ms = (timestamps[current_index] - lidar_timestamp) / 1_000_000.0
    if -delta_ms > max_delta_ms:
        raise RadarAlignmentUnavailableError(
            f"Latest causal radar frame is {-delta_ms:.1f} ms before LiDAR frame; "
            f"limit is {max_delta_ms:.1f} ms"
        )

    continental_gt = find_named_file(scene_root, "Continental_gt.txt")
    selected_pose_rows: list[np.ndarray] = []
    pose_deltas: list[float] = []
    oldest_timestamp = lidar_timestamp - int(config.max_age_s * 1_000_000_000)
    candidate_start = bisect_left(
        timestamps,
        oldest_timestamp,
        0,
        current_index + 1,
    )
    if config.max_frames is not None:
        candidate_start = max(
            candidate_start,
            current_index - config.max_frames + 1,
        )
    for index in range(candidate_start, current_index + 1):
        pose, pose_delta_ms = nearest_ground_truth_pose(
            continental_gt, timestamps[index]
        )
        selected_pose_rows.append(pose)
        pose_deltas.append(pose_delta_ms)
    candidate_timestamps = timestamps[candidate_start : current_index + 1]
    selection_rows = select_adaptive_indices(
        candidate_timestamps,
        np.stack(selected_pose_rows),
        lidar_timestamp,
        config,
    )
    selected: list[SelectedFrame] = []
    for row in selection_rows:
        local_index = int(row["index"])
        global_index = candidate_start + local_index
        selected.append(
            SelectedFrame(
                path=Path(path_texts[global_index]),
                timestamp=timestamps[global_index],
                pose=selected_pose_rows[local_index],
                age_s=float(row["age_s"]),
                translation_m=float(row["translation_m"]),
                rotation_deg=float(row["rotation_deg"]),
                weight=float(row["weight"]),
                pose_delta_ms=float(pose_deltas[local_index]),
            )
        )
    if not selected:
        raise RadarAlignmentUnavailableError(
            f"Adaptive policy selected no radar frame for {lidar_timestamp}"
        )
    return selected, delta_ms


def _process_frames(
    scene_root: Path,
    lidar_timestamp: int,
    selected: list[SelectedFrame],
    tracking_config: DopplerTrackingConfig,
) -> tuple[
    list[ProcessedFrame],
    list[np.ndarray],
    list[np.ndarray],
    float,
    float,
]:
    continental_gt_path = find_named_file(scene_root, "Continental_gt.txt")
    pose_timestamps, continental_poses = load_ground_truth_poses(
        str(continental_gt_path.resolve())
    )
    aeva_gt = find_named_file(scene_root, "Aeva_gt.txt")
    lidar_to_imu = load_named_transform(
        find_named_file(scene_root, "IMU_LiDAR.txt"),
        "Tr_lidar_to_imu",
    )
    lidar_to_radar = load_lidar_to_radar_transform(
        find_named_file(scene_root, "Continental_LiDAR.txt")
    )
    radar_to_lidar = np.linalg.inv(lidar_to_radar)
    imu_from_lidar_rotation = lidar_to_imu[:3, :3]
    imu_from_radar_rotation = imu_from_lidar_rotation @ radar_to_lidar[:3, :3]

    lidar_pose, lidar_pose_delta_ms = nearest_ground_truth_pose(
        aeva_gt, lidar_timestamp
    )
    world_from_current_lidar = np.eye(4, dtype=np.float64)
    world_from_current_lidar[:3, :3] = (
        lidar_pose[:3, :3] @ imu_from_lidar_rotation
    )
    world_from_current_lidar[:3, 3] = lidar_pose[:3, 3]
    current_lidar_from_world = np.linalg.inv(world_from_current_lidar)

    processed: list[ProcessedFrame] = []
    raw_frames: list[np.ndarray] = []
    radar_to_current_rotations: list[np.ndarray] = []
    latest_speed = 0.0
    latest_yaw_rate = 0.0
    for selected_frame in selected:
        raw = read_continental_bin(selected_frame.path)
        velocity_radar, yaw_rate = _pose_velocity(
            pose_timestamps,
            continental_poses,
            selected_frame.timestamp,
            imu_from_radar_rotation,
        )
        residual, applied_sign, _ = compensate_doppler(
            raw,
            velocity_radar,
            doppler_sign=tracking_config.doppler_sign,
            sign_inference_min_speed_mps=tracking_config.sign_inference_min_speed_mps,
        )
        dynamic = np.abs(residual) > tracking_config.dynamic_threshold_mps
        labels = dbscan_labels(
            raw[:, :2],
            dynamic,
            tracking_config.cluster_eps_m,
            tracking_config.cluster_min_samples,
        )

        world_from_radar = np.eye(4, dtype=np.float64)
        world_from_radar[:3, :3] = (
            selected_frame.pose[:3, :3] @ imu_from_radar_rotation
        )
        world_from_radar[:3, 3] = selected_frame.pose[:3, 3]
        current_lidar_from_radar = current_lidar_from_world @ world_from_radar
        aligned = raw.copy()
        if len(aligned):
            aligned[:, :3] = transform_xyz(
                raw[:, :3], current_lidar_from_radar
            )
        processed.append(
            ProcessedFrame(
                timestamp=selected_frame.timestamp,
                points=aligned,
                doppler_residual_mps=residual,
                dynamic_mask=dynamic,
                cluster_labels=labels,
                weight=selected_frame.weight,
                doppler_sign=applied_sign,
                sensor_speed_mps=float(np.linalg.norm(velocity_radar)),
                yaw_rate_dps=math.degrees(yaw_rate),
            )
        )
        raw_frames.append(raw)
        radar_to_current_rotations.append(current_lidar_from_radar[:3, :3])
        latest_speed = float(np.linalg.norm(velocity_radar))
        latest_yaw_rate = math.degrees(yaw_rate)
    return (
        processed,
        raw_frames,
        radar_to_current_rotations,
        latest_speed,
        latest_yaw_rate,
    )


def radar_cache_is_compatible(
    cache_path: Path,
    sample_metadata: dict,
    *,
    max_delta_ms: float | None = None,
    stack_config: AdaptiveStackConfig | None = None,
    tracking_config: DopplerTrackingConfig | None = None,
) -> bool:
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if not {"radar_bev", "metadata_json"}.issubset(data.files):
                return False
            radar_bev = np.asarray(data["radar_bev"])
            metadata = json.loads(str(data["metadata_json"]))
        x_range = tuple(
            float(value)
            for value in sample_metadata.get("x_range", [0.0, 64.0])
        )
        y_range = tuple(
            float(value)
            for value in sample_metadata.get("y_range", [-32.0, 32.0])
        )
        resolution = float(sample_metadata.get("resolution", 0.2))
        expected_shape = (
            4,
            int(np.ceil((x_range[1] - x_range[0]) / resolution)),
            int(np.ceil((y_range[1] - y_range[0]) / resolution)),
        )
        if (
            radar_bev.shape != expected_shape
            or not np.isfinite(radar_bev).all()
            or (radar_bev.size and float(radar_bev.min()) < 0.0)
            or (radar_bev.size and float(radar_bev.max()) > 1.0)
        ):
            return False
        if int(metadata.get("cache_format_version", 0)) != RADAR_CACHE_VERSION:
            return False
        if metadata.get("policy") != POLICY_NAME:
            return False
        if metadata.get("channels") != CHANNELS:
            return False
        if str(metadata.get("scene")) != scene_name_from_metadata(sample_metadata):
            return False
        if str(metadata.get("session", "")) != session_name_from_metadata(
            sample_metadata
        ):
            return False
        if str(metadata.get("lidar_timestamp")) != str(sample_metadata["timestamp"]):
            return False
        if not np.allclose(metadata.get("x_range"), x_range):
            return False
        if not np.allclose(metadata.get("y_range"), y_range):
            return False
        if not np.isclose(float(metadata.get("resolution")), resolution):
            return False
        if max_delta_ms is not None:
            if abs(float(metadata.get("radar_delta_ms"))) > max_delta_ms + 1e-9:
                return False
            if not np.isclose(float(metadata.get("max_delta_ms")), max_delta_ms):
                return False
        if stack_config is not None and metadata.get("adaptive_stack") != asdict(
            stack_config
        ):
            return False
        if tracking_config is not None and metadata.get(
            "doppler_tracking"
        ) != asdict(tracking_config):
            return False
        return True
    except Exception:
        return False


def build_radar_cache_entry(
    sample_path: Path,
    hercules_root: Path,
    radar_root: Path,
    max_delta_ms: float = 30.0,
    stack_config: AdaptiveStackConfig | None = None,
    tracking_config: DopplerTrackingConfig | None = None,
) -> Path:
    stack_config = stack_config or AdaptiveStackConfig()
    tracking_config = tracking_config or DopplerTrackingConfig()
    stack_config.validate()
    tracking_config.validate()
    if not math.isfinite(max_delta_ms) or max_delta_ms < 0.0:
        raise ValueError("max_delta_ms must be finite and non-negative")

    sample_path = Path(sample_path)
    metadata = load_sample_metadata(sample_path)
    output_path = radar_cache_path(radar_root, metadata)
    if radar_cache_is_compatible(
        output_path,
        metadata,
        max_delta_ms=max_delta_ms,
        stack_config=stack_config,
        tracking_config=tracking_config,
    ):
        return output_path

    scene_root = scene_session_root(hercules_root, metadata)
    lidar_timestamp = int(metadata["timestamp"])
    selected, radar_delta_ms = _select_scene_frames(
        scene_root,
        lidar_timestamp,
        max_delta_ms,
        stack_config,
    )
    (
        frames,
        raw_frames,
        radar_to_current_rotations,
        ego_speed_mps,
        yaw_rate_dps,
    ) = _process_frames(
        scene_root,
        lidar_timestamp,
        selected,
        tracking_config,
    )
    observations = _make_observations(
        frames, raw_frames, radar_to_current_rotations
    )
    tracks = associate_tracks(
        observations,
        tracking_config.association_distance_m,
        tracking_config.velocity_smoothing,
    )

    x_range = tuple(
        float(value) for value in metadata.get("x_range", [0.0, 64.0])
    )
    y_range = tuple(
        float(value) for value in metadata.get("y_range", [-32.0, 32.0])
    )
    resolution = float(metadata.get("resolution", 0.2))
    radar_bev = project_adaptive_bev(
        frames,
        observations,
        tracks,
        lidar_timestamp,
        x_range,
        y_range,
        resolution,
        tracking_config,
    )

    confirmed_tracks = [
        {
            "track_id": track.track_id,
            "hits": track.hits,
            "velocity_xy_mps": track.velocity.tolist(),
            "speed_mps": float(np.linalg.norm(track.velocity)),
            "latest_position_xy_m": track.position.tolist(),
        }
        for track in tracks.values()
        if track.hits >= tracking_config.min_track_hits
    ]
    alignment_rows = [
        {
            "radar_timestamp": str(selection.timestamp),
            "age_ms": selection.age_s * 1000.0,
            "relative_translation_m": selection.translation_m,
            "relative_rotation_deg": selection.rotation_deg,
            "weight": selection.weight,
            "radar_pose_delta_ms": selection.pose_delta_ms,
            "doppler_sign": frame.doppler_sign,
            "sensor_speed_mps": frame.sensor_speed_mps,
            "yaw_rate_dps": frame.yaw_rate_dps,
            "point_count": int(len(frame.points)),
            "dynamic_point_count": int(np.count_nonzero(frame.dynamic_mask)),
        }
        for selection, frame in zip(selected, frames)
    ]
    cache_metadata = {
        "cache_format_version": RADAR_CACHE_VERSION,
        "policy": POLICY_NAME,
        "scene": scene_name_from_metadata(metadata),
        "session": session_name_from_metadata(metadata),
        "lidar_timestamp": str(lidar_timestamp),
        "radar_timestamp": selected[-1].path.stem,
        "radar_delta_ms": radar_delta_ms,
        "radar_sources": [str(item.path) for item in selected],
        "radar_frame_count": len(selected),
        "effective_frame_support": float(sum(item.weight for item in selected)),
        "adaptive_stack": asdict(stack_config),
        "doppler_tracking": asdict(tracking_config),
        "ego_speed_mps": ego_speed_mps,
        "yaw_rate_dps": yaw_rate_dps,
        "alignment_rows": alignment_rows,
        "confirmed_dynamic_track_count": len(confirmed_tracks),
        "dynamic_tracks": confirmed_tracks,
        "temporal_alignment": (
            "adaptive pose-gated ego compensation into the current Aeva frame"
        ),
        "doppler_processing": (
            "ego-compensated radial residuals, spatial clustering, causal "
            "association, and tracked-point motion compensation"
        ),
        "channels": CHANNELS,
        "x_range": x_range,
        "y_range": y_range,
        "resolution": resolution,
        "max_delta_ms": max_delta_ms,
    }
    atomic_savez_compressed(
        output_path,
        radar_bev=radar_bev.astype(np.float16),
        metadata_json=np.asarray(json.dumps(cache_metadata)),
    )
    return output_path
