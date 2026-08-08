"""K-Radar tensor-derived adaptive stacking and Doppler-tracked BEV cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez_compressed
from Fault_Localization_Model.kradar_dataset import (
    kradar_sequence_root,
    load_radar_from_lidar_transform,
    parse_label_frame,
)
from PFS_Radar_v2.pose import pose_velocity, select_adaptive_indices
from PFS_Radar_v2.radar_types import (
    AdaptiveStackConfig,
    ClusterObservation,
    DopplerTrackingConfig,
    ProcessedFrame,
    RadarAlignmentUnavailableError,
    SelectedFrame,
    TrackState,
)
from PFS_Radar_v2.tracking import (
    associate_tracks,
    compensate_doppler,
    dbscan_labels,
    make_observations,
)


RADAR_CACHE_VERSION = 7
POLICY_NAME = "kradar_pc10p_power_height_doppler_v2"
CHANNELS = [
    "static_occupancy",
    "normalized_power",
    "tracked_dynamic_speed",
    "robust_upper_height",
]
POWER_NORMALIZATION_QUANTILE = 0.99
UPPER_HEIGHT_QUANTILE = 0.90
HEIGHT_RANGE_M = (-3.0, 5.0)
K_RADAR_POINT_COLUMNS = [
    "x_m",
    "y_m",
    "z_m",
    "power",
    "doppler_mps",
    "range_m",
    "azimuth_rad",
    "elevation_rad",
    "range_index",
    "azimuth_index",
    "elevation_index",
]


def transform_xyz(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N,3], got {xyz.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape [4,4], got {transform.shape}")
    homogeneous = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float64)))
    return (transform @ homogeneous.T).T[:, :3].astype(np.float32)


@dataclass(frozen=True)
class KRadarSequenceIndex:
    sequence: str
    timestamps: tuple[int, ...]
    poses: np.ndarray
    radar_indices: tuple[str, ...]
    lidar_indices: tuple[str, ...]
    radar_paths: tuple[Path, ...]

    def position(self, radar_index: str | int) -> int:
        normalized = f"{int(radar_index):05d}"
        try:
            return self.radar_indices.index(normalized)
        except ValueError as exc:
            raise RadarAlignmentUnavailableError(
                f"K-Radar sequence {self.sequence} has no pc10p frame {normalized}"
            ) from exc


def kradar_cache_path(root: Path, sequence: str | int, radar_index: str | int) -> Path:
    sequence_text = str(int(sequence))
    radar_text = f"{int(radar_index):05d}"
    return Path(root) / sequence_text / f"{radar_text}.npz"


def _odometry_path(odometry_root: Path, sequence: str) -> Path:
    candidates = (
        Path(odometry_root) / f"gt_{int(sequence):02d}.txt",
        Path(odometry_root) / "gt" / f"gt_{int(sequence):02d}.txt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No K-Radar odometry for sequence {sequence} under {odometry_root}"
    )


@lru_cache(maxsize=64)
def _load_sequence_index_cached(
    dataset_root_text: str,
    radar_point_root_text: str,
    odometry_root_text: str,
    sequence: str,
) -> KRadarSequenceIndex:
    dataset_root = Path(dataset_root_text)
    radar_point_root = Path(radar_point_root_text)
    odometry_root = Path(odometry_root_text)
    sequence_root = kradar_sequence_root(dataset_root, sequence)
    label_paths = sorted((sequence_root / "info_label").glob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(
            f"No K-Radar labels under {sequence_root / 'info_label'}"
        )

    pose_rows = np.loadtxt(_odometry_path(odometry_root, sequence), dtype=np.float64)
    pose_rows = np.atleast_2d(pose_rows)
    if pose_rows.shape != (len(label_paths), 12):
        raise ValueError(
            f"K-Radar sequence {sequence} has {len(label_paths)} labels but odometry "
            f"shape {pose_rows.shape}; expected [{len(label_paths)},12]"
        )
    all_poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(pose_rows), axis=0)
    all_poses[:, :3, :4] = pose_rows.reshape(-1, 3, 4)

    timestamps: list[int] = []
    poses: list[np.ndarray] = []
    radar_indices: list[str] = []
    lidar_indices: list[str] = []
    radar_paths: list[Path] = []
    for row, label_path in enumerate(label_paths):
        radar_index, lidar_index, timestamp = parse_label_frame(label_path)
        radar_path = radar_point_root / sequence / f"rpc_{radar_index}.npy"
        if not radar_path.is_file():
            continue
        timestamps.append(timestamp)
        poses.append(all_poses[row])
        radar_indices.append(radar_index)
        lidar_indices.append(lidar_index)
        radar_paths.append(radar_path)
    if not radar_paths:
        raise FileNotFoundError(
            f"No pc10p frames for K-Radar sequence {sequence} under {radar_point_root}"
        )
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"K-Radar sequence {sequence} timestamps are not increasing")
    return KRadarSequenceIndex(
        sequence=sequence,
        timestamps=tuple(timestamps),
        poses=np.stack(poses),
        radar_indices=tuple(radar_indices),
        lidar_indices=tuple(lidar_indices),
        radar_paths=tuple(radar_paths),
    )


def load_sequence_index(
    dataset_root: Path,
    radar_point_root: Path,
    odometry_root: Path,
    sequence: str | int,
) -> KRadarSequenceIndex:
    return _load_sequence_index_cached(
        str(Path(dataset_root).resolve()),
        str(Path(radar_point_root).resolve()),
        str(Path(odometry_root).resolve()),
        str(int(sequence)),
    )


@lru_cache(maxsize=32)
def load_kradar_pc10p(path: Path) -> np.ndarray:
    """Load the tensor-derived K-Radar pc10p format as float32."""

    points = np.load(Path(path), allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != len(K_RADAR_POINT_COLUMNS):
        raise ValueError(
            f"K-Radar pc10p file {path} must have shape [N,11], got {points.shape}"
        )
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f"K-Radar pc10p file contains non-finite values: {path}")
    points.setflags(write=False)
    return points


@lru_cache(maxsize=64)
def load_lidar_from_radar_transform(
    calibration_path: Path,
    radar_to_lidar_z_m: float = 0.7,
) -> np.ndarray:
    """Load K-Radar's translation-only radar/LiDAR calibration."""

    transform = np.linalg.inv(
        load_radar_from_lidar_transform(calibration_path, radar_to_lidar_z_m)
    )
    transform.setflags(write=False)
    return transform


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
    rows_from_bottom = np.floor((xyz[:, 0] - x_range[0]) / resolution).astype(
        np.int32
    )
    rows = height - 1 - rows_from_bottom
    valid &= (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows, cols, valid


def _normalize_cell_power(cell_power: np.ndarray) -> np.ndarray:
    """Robustly log-normalize non-negative cell power into [0, 1]."""

    output = np.zeros_like(cell_power, dtype=np.float32)
    occupied = cell_power > 0.0
    if not np.any(occupied):
        return output
    scale = float(np.quantile(cell_power[occupied], POWER_NORMALIZATION_QUANTILE))
    if scale <= 0.0:
        return output
    output[occupied] = np.log1p(np.minimum(cell_power[occupied], scale)) / np.log1p(
        scale
    )
    return output


def _upper_height_grid(
    flat_cells: np.ndarray,
    heights_m: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return normalized per-cell upper-height quantiles with empty cells at zero."""

    output = np.zeros(shape[0] * shape[1], dtype=np.float32)
    if flat_cells.size == 0:
        return output.reshape(shape)

    z_min, z_max = HEIGHT_RANGE_M
    plausible = (heights_m >= z_min) & (heights_m <= z_max)
    flat_cells = flat_cells[plausible]
    heights_m = heights_m[plausible]
    if flat_cells.size == 0:
        return output.reshape(shape)

    order = np.lexsort((heights_m, flat_cells))
    sorted_cells = flat_cells[order]
    sorted_heights = heights_m[order]
    starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    ends = np.r_[starts[1:], len(sorted_cells)]
    counts = ends - starts
    offsets = np.ceil(UPPER_HEIGHT_QUANTILE * counts).astype(np.int64) - 1
    upper_heights = sorted_heights[starts + offsets]

    normalized = (upper_heights - z_min) / (z_max - z_min)
    output[sorted_cells[starts]] = normalized.astype(np.float32)
    return output.reshape(shape)


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
    """Project K-Radar frames as occupancy, power, speed, and upper height."""

    tracking_config.validate()
    height = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    width = int(np.ceil((y_range[1] - y_range[0]) / resolution))
    output = np.zeros((4, height, width), dtype=np.float32)
    cell_power = np.zeros((height, width), dtype=np.float32)
    height_cells = []
    height_values = []

    for frame in frames:
        static_points = frame.points[~frame.dynamic_mask, :3]
        if len(static_points):
            rows, cols, valid = _grid_indices(
                static_points, x_range, y_range, resolution
            )
            rows, cols = rows[valid], cols[valid]
            supported_cells = np.unique(rows.astype(np.int64) * width + cols)
            supported_rows = supported_cells // width
            supported_cols = supported_cells % width
            output[0, supported_rows, supported_cols] = 1.0

        rows, cols, valid = _grid_indices(
            frame.points[:, :3], x_range, y_range, resolution
        )
        if np.any(valid):
            rows, cols = rows[valid], cols[valid]
            weighted_power = np.maximum(frame.points[valid, 4], 0.0) * frame.weight
            np.maximum.at(cell_power, (rows, cols), weighted_power)
            height_cells.append(rows.astype(np.int64) * width + cols)
            height_values.append(frame.points[valid, 2])

    output[1] = _normalize_cell_power(cell_power)
    if height_cells:
        output[3] = _upper_height_grid(
            np.concatenate(height_cells),
            np.concatenate(height_values),
            (height, width),
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
            else:
                speed = float(
                    np.max(np.abs(frame.doppler_residual_mps[observation.point_indices]))
                )
            rows, cols, valid = _grid_indices(points, x_range, y_range, resolution)
            rows, cols = rows[valid], cols[valid]
            np.maximum.at(
                output[2],
                (rows, cols),
                np.float32(np.clip(speed / tracking_config.max_abs_velocity_mps, 0, 1)),
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
                np.clip(residual[valid] / tracking_config.max_abs_velocity_mps, 0, 1),
            )
    return output


def _select_sequence_frames(
    sequence_index: KRadarSequenceIndex,
    current_position: int,
    config: AdaptiveStackConfig,
) -> list[SelectedFrame]:
    current_timestamp = sequence_index.timestamps[current_position]
    rows = select_adaptive_indices(
        sequence_index.timestamps[: current_position + 1],
        sequence_index.poses[: current_position + 1],
        current_timestamp,
        config,
        reference_pose=sequence_index.poses[current_position],
    )
    selected = [
        SelectedFrame(
            path=sequence_index.radar_paths[row["index"]],
            timestamp=row["timestamp"],
            pose=sequence_index.poses[row["index"]],
            age_s=float(row["age_s"]),
            translation_m=float(row["translation_m"]),
            rotation_deg=float(row["rotation_deg"]),
            weight=float(row["weight"]),
            pose_delta_ms=0.0,
            radar_index=sequence_index.radar_indices[row["index"]],
            lidar_index=sequence_index.lidar_indices[row["index"]],
        )
        for row in rows
    ]
    if not selected or selected[-1].radar_index != sequence_index.radar_indices[current_position]:
        raise RadarAlignmentUnavailableError(
            f"Adaptive policy did not select current K-Radar frame "
            f"{sequence_index.radar_indices[current_position]}"
        )
    return selected


def _process_frames(
    sequence_index: KRadarSequenceIndex,
    current_position: int,
    selected: list[SelectedFrame],
    lidar_from_radar: np.ndarray,
    tracking_config: DopplerTrackingConfig,
) -> tuple[list[ProcessedFrame], list[np.ndarray], list[np.ndarray], float, float]:
    current_lidar_from_world = np.linalg.inv(sequence_index.poses[current_position])
    processed: list[ProcessedFrame] = []
    raw_frames: list[np.ndarray] = []
    radar_to_current_rotations: list[np.ndarray] = []
    latest_speed = 0.0
    latest_yaw_rate = 0.0

    for selected_frame in selected:
        source_position = sequence_index.position(selected_frame.radar_index)
        source_points = load_kradar_pc10p(selected_frame.path)
        # Tracking utilities expect [x,y,z,Doppler,...]. Keep power in column 4.
        raw = np.column_stack(
            (source_points[:, :3], source_points[:, 4], source_points[:, 3])
        ).astype(np.float32, copy=False)
        velocity_radar, yaw_rate = pose_velocity(
            sequence_index.timestamps,
            sequence_index.poses,
            selected_frame.timestamp,
            np.eye(3, dtype=np.float64),
        )
        residual, applied_sign, _ = compensate_doppler(
            raw,
            velocity_radar,
            doppler_sign=tracking_config.doppler_sign,
            sign_inference_min_speed_mps=tracking_config.sign_inference_min_speed_mps,
            doppler_period_mps=tracking_config.doppler_period_mps,
        )
        power_threshold = float(
            np.quantile(raw[:, 4], tracking_config.dynamic_power_quantile)
        )
        dynamic = (
            (np.abs(residual) > tracking_config.dynamic_threshold_mps)
            & (raw[:, 4] >= power_threshold)
        )
        labels = dbscan_labels(
            raw[:, :2],
            dynamic,
            tracking_config.cluster_eps_m,
            tracking_config.cluster_min_samples,
        )

        current_lidar_from_radar = (
            current_lidar_from_world
            @ sequence_index.poses[source_position]
            @ lidar_from_radar
        )
        aligned = raw.copy()
        aligned[:, :3] = transform_xyz(raw[:, :3], current_lidar_from_radar)
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
    return processed, raw_frames, radar_to_current_rotations, latest_speed, latest_yaw_rate


def radar_cache_is_compatible(
    cache_path: Path,
    *,
    sequence: str | int,
    radar_index: str | int,
    lidar_index: str | int,
    timestamp: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    stack_config: AdaptiveStackConfig | None = None,
    tracking_config: DopplerTrackingConfig | None = None,
) -> bool:
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if not {"radar_bev", "metadata_json"}.issubset(data.files):
                return False
            radar_bev = np.asarray(data["radar_bev"])
            metadata = json.loads(str(data["metadata_json"]))
        expected_shape = (
            4,
            int(np.ceil((x_range[1] - x_range[0]) / resolution)),
            int(np.ceil((y_range[1] - y_range[0]) / resolution)),
        )
        if (
            radar_bev.shape != expected_shape
            or not np.isfinite(radar_bev).all()
            or float(radar_bev.min(initial=0.0)) < 0.0
            or float(radar_bev.max(initial=0.0)) > 1.0
        ):
            return False
        if int(metadata.get("cache_format_version", 0)) != RADAR_CACHE_VERSION:
            return False
        if metadata.get("policy") != POLICY_NAME or metadata.get("channels") != CHANNELS:
            return False
        if str(metadata.get("sequence")) != str(int(sequence)):
            return False
        if str(metadata.get("radar_index")) != f"{int(radar_index):05d}":
            return False
        if str(metadata.get("lidar_index")) != f"{int(lidar_index):05d}":
            return False
        if int(metadata.get("timestamp_ns")) != int(timestamp):
            return False
        if not np.allclose(metadata.get("x_range"), x_range):
            return False
        if not np.allclose(metadata.get("y_range"), y_range):
            return False
        if not np.isclose(float(metadata.get("resolution")), resolution):
            return False
        if stack_config is not None and metadata.get("adaptive_stack") != asdict(stack_config):
            return False
        if tracking_config is not None and metadata.get("doppler_tracking") != asdict(
            tracking_config
        ):
            return False
        return True
    except Exception:
        return False


def build_radar_cache_entry(
    dataset_root: Path,
    radar_point_root: Path,
    odometry_root: Path,
    output_root: Path,
    sequence: str | int,
    radar_index: str | int,
    x_range: tuple[float, float] = (0.0, 64.0),
    y_range: tuple[float, float] = (-32.0, 32.0),
    resolution: float = 0.2,
    radar_to_lidar_z_m: float = 0.7,
    stack_config: AdaptiveStackConfig | None = None,
    tracking_config: DopplerTrackingConfig | None = None,
) -> Path:
    stack_config = stack_config or AdaptiveStackConfig()
    tracking_config = tracking_config or DopplerTrackingConfig()
    stack_config.validate()
    tracking_config.validate()
    if resolution <= 0.0 or not math.isfinite(resolution):
        raise ValueError("resolution must be finite and positive")

    sequence_index = load_sequence_index(
        dataset_root, radar_point_root, odometry_root, sequence
    )
    current_position = sequence_index.position(radar_index)
    radar_text = sequence_index.radar_indices[current_position]
    lidar_text = sequence_index.lidar_indices[current_position]
    lidar_timestamp = sequence_index.timestamps[current_position]
    output_path = kradar_cache_path(output_root, sequence_index.sequence, radar_text)
    compatibility = dict(
        sequence=sequence_index.sequence,
        radar_index=radar_text,
        lidar_index=lidar_text,
        timestamp=lidar_timestamp,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        stack_config=stack_config,
        tracking_config=tracking_config,
    )
    if radar_cache_is_compatible(output_path, **compatibility):
        return output_path

    calibration_path = (
        kradar_sequence_root(dataset_root, sequence_index.sequence)
        / "info_calib"
        / "calib_radar_lidar.txt"
    )
    lidar_from_radar = load_lidar_from_radar_transform(
        calibration_path, radar_to_lidar_z_m
    )
    selected = _select_sequence_frames(sequence_index, current_position, stack_config)
    frames, raw_frames, rotations, ego_speed_mps, yaw_rate_dps = _process_frames(
        sequence_index,
        current_position,
        selected,
        lidar_from_radar,
        tracking_config,
    )
    observations = make_observations(frames, raw_frames, rotations)
    tracks = associate_tracks(
        observations,
        tracking_config.association_distance_m,
        tracking_config.velocity_smoothing,
    )
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
            "radar_index": selection.radar_index,
            "lidar_index": selection.lidar_index,
            "timestamp_ns": selection.timestamp,
            "age_ms": selection.age_s * 1000.0,
            "relative_translation_m": selection.translation_m,
            "relative_rotation_deg": selection.rotation_deg,
            "weight": selection.weight,
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
        "dataset": "K-Radar",
        "radar_source": "from_rdr_cube_xyz/pc10p",
        "radar_source_columns": K_RADAR_POINT_COLUMNS,
        "sequence": sequence_index.sequence,
        "radar_index": radar_text,
        "lidar_index": lidar_text,
        "timestamp_ns": lidar_timestamp,
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
        "calibration": {
            "lidar_from_radar": lidar_from_radar.tolist(),
            "radar_to_lidar_z_m": radar_to_lidar_z_m,
        },
        "pose_source": "K-Radar resources/odometry/gt KITTI-format poses",
        "temporal_alignment": "adaptive pose-gated compensation into current paired LiDAR frame",
        "doppler_processing": "K-Radar ambiguity-wrapped ego residual, power gating, clustering, and causal tracking",
        "channels": CHANNELS,
        "power_normalization_quantile": POWER_NORMALIZATION_QUANTILE,
        "upper_height_quantile": UPPER_HEIGHT_QUANTILE,
        "height_range_m": HEIGHT_RANGE_M,
        "x_range": x_range,
        "y_range": y_range,
        "resolution": resolution,
    }
    atomic_savez_compressed(
        output_path,
        radar_bev=radar_bev.astype(np.float16),
        metadata_json=np.asarray(json.dumps(cache_metadata)),
    )
    return output_path
