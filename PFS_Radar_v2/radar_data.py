from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict
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
    radar_cache_path,
    read_continental_bin as _read_continental_bin,
    scene_name_from_metadata,
    scene_radar_resources,
    scene_session_root,
    session_name_from_metadata,
    transform_xyz,
)
from PFS_Radar_v2.pose import (
    interpolated_pose_from_sequence,
    pose_velocity,
    select_adaptive_indices,
)
from PFS_Radar_v2.tracking import (
    associate_tracks,
    compensate_doppler,
    dbscan_labels,
    make_observations,
)
from PFS_Radar_v2.radar_types import (
    AdaptiveStackConfig,
    ClusterObservation,
    DopplerTrackingConfig,
    ProcessedFrame,
    SelectedFrame,
    TrackState,
)


RADAR_CACHE_VERSION = 5
POLICY_NAME = "adaptive_pose_doppler_tracking_interpolated_v3"
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
    pose_timestamps, continental_poses = load_ground_truth_poses(
        str(continental_gt.resolve())
    )
    lidar_reference_pose, lidar_reference_delta_ms = interpolated_pose_from_sequence(
        pose_timestamps,
        continental_poses,
        lidar_timestamp,
    )
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
        pose, pose_delta_ms = interpolated_pose_from_sequence(
            pose_timestamps,
            continental_poses,
            timestamps[index],
        )
        selected_pose_rows.append(pose)
        pose_deltas.append(pose_delta_ms)
    candidate_timestamps = timestamps[candidate_start : current_index + 1]
    selection_rows = select_adaptive_indices(
        candidate_timestamps,
        np.stack(selected_pose_rows),
        lidar_timestamp,
        config,
        reference_pose=lidar_reference_pose,
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
                pose_delta_ms=max(
                    float(pose_deltas[local_index]),
                    float(abs(lidar_reference_delta_ms)),
                ),
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

    aeva_pose_timestamps, aeva_poses = load_ground_truth_poses(
        str(aeva_gt.resolve())
    )
    lidar_pose, lidar_pose_delta_ms = interpolated_pose_from_sequence(
        aeva_pose_timestamps,
        aeva_poses,
        lidar_timestamp,
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
        velocity_radar, yaw_rate = pose_velocity(
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
    observations = make_observations(
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
        "pose_interpolation": (
            "linear translation plus SLERP rotation at radar and LiDAR timestamps"
        ),
        "temporal_alignment": (
            "adaptive pose-gated ego compensation into the interpolated current Aeva frame"
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
