from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from PFS_Radar_v2.radar_types import ClusterObservation, ProcessedFrame, TrackState


def compensate_doppler(
    points: np.ndarray,
    sensor_velocity: np.ndarray,
    doppler_sign: str = "auto",
    sign_inference_min_speed_mps: float = 0.5,
    doppler_period_mps: float | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Subtract expected ego Doppler, including K-Radar velocity wrapping."""

    points = np.asarray(points, dtype=np.float64)
    sensor_velocity = np.asarray(sensor_velocity, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"points must have shape [N,>=4], got {points.shape}")
    if sensor_velocity.shape != (3,) or not np.isfinite(sensor_velocity).all():
        raise ValueError("sensor_velocity must be a finite 3-vector")
    if doppler_sign not in {"auto", "1", "-1"}:
        raise ValueError("doppler_sign must be one of: auto, 1, -1")
    if doppler_period_mps is not None and doppler_period_mps <= 0.0:
        raise ValueError("doppler_period_mps must be positive when provided")
    ranges = np.linalg.norm(points[:, :3], axis=1)
    valid = ranges > 1e-6
    line_of_sight = np.zeros((len(points), 3), dtype=np.float64)
    line_of_sight[valid] = points[valid, :3] / ranges[valid, None]
    expected_static = -(line_of_sight @ sensor_velocity)
    measured = points[:, 3]

    def residual_for_sign(sign: int) -> np.ndarray:
        residual = sign * measured - expected_static
        if doppler_period_mps is not None:
            half_period = 0.5 * doppler_period_mps
            residual = (residual + half_period) % doppler_period_mps - half_period
        return residual

    if doppler_sign == "auto":
        if np.linalg.norm(sensor_velocity) < sign_inference_min_speed_mps or not np.any(valid):
            sign = 1
        else:
            positive_score = float(np.median(np.abs(residual_for_sign(1)[valid])))
            negative_score = float(np.median(np.abs(residual_for_sign(-1)[valid])))
            sign = 1 if positive_score <= negative_score else -1
    else:
        sign = int(doppler_sign)
    residual = residual_for_sign(sign)
    residual[~valid] = 0.0
    return residual.astype(np.float32), sign, expected_static.astype(np.float32)


def dbscan_labels(
    xy: np.ndarray,
    candidate_mask: np.ndarray,
    eps_m: float,
    min_samples: int,
) -> np.ndarray:
    """2-D DBSCAN using a spatial index suitable for dense K-Radar cells."""

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
    coordinates = xy[candidates]
    tree = cKDTree(coordinates)
    neighbor_counts = tree.query_ball_point(
        coordinates, eps_m, return_length=True
    )
    core = neighbor_counts >= min_samples
    visited = np.zeros(len(candidates), dtype=bool)
    local_labels = np.full(len(candidates), -1, dtype=np.int32)
    cluster_id = 0
    for seed in range(len(candidates)):
        if visited[seed]:
            continue
        visited[seed] = True
        if not core[seed]:
            continue
        local_labels[seed] = cluster_id
        queue = list(tree.query_ball_point(coordinates[seed], eps_m))
        queued = set(queue)
        cursor = 0
        while cursor < len(queue):
            point = queue[cursor]
            cursor += 1
            if not visited[point]:
                visited[point] = True
                if core[point]:
                    point_neighbors = tree.query_ball_point(coordinates[point], eps_m)
                    for neighbor in point_neighbors:
                        if neighbor not in queued:
                            queued.add(neighbor)
                            queue.append(neighbor)
            if local_labels[point] < 0:
                local_labels[point] = cluster_id
        cluster_id += 1
    labels[candidates] = local_labels
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


def make_observations(
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
