from __future__ import annotations

from bisect import bisect_left, bisect_right
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.io_utils import atomic_savez_compressed
from Fault_Localization_Model.sample_utils import load_sample_metadata


CONTINENTAL_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "velocity", "range", "rcs", "azimuth", "elevation"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "u1", "<f4", "<f4"],
        "offsets": [0, 4, 8, 12, 16, 20, 21, 25],
        "itemsize": 29,
    }
)
RADAR_CACHE_VERSION = 3


class RadarAlignmentUnavailableError(ValueError):
    """Raised when a causal radar stack cannot be aligned without extrapolation."""


def _safe_metadata_component(value, label: str, *, required: bool = True) -> str:
    component = str(value or "").strip()
    if not component:
        if required:
            raise KeyError(f"Sample metadata does not contain {label}")
        return ""
    if (
        component in {".", ".."}
        or "/" in component
        or "\\" in component
        or Path(component).is_absolute()
    ):
        raise ValueError(
            f"Sample metadata {label} must be one path component, got {component!r}"
        )
    return component


def scene_name_from_metadata(metadata: dict) -> str:
    scene = str(metadata.get("scene") or metadata.get("day") or "").strip()
    if scene:
        return _safe_metadata_component(scene, "scene")
    relative = str(metadata.get("source_relative_path", "")).replace("\\", "/")
    if relative:
        return _safe_metadata_component(
            relative.split("/", maxsplit=1)[0],
            "scene",
        )
    raise KeyError("Sample metadata does not identify its HeRCULES scene")


def session_name_from_metadata(metadata: dict) -> str:
    return _safe_metadata_component(
        metadata.get("session"),
        "session",
        required=False,
    )


def radar_cache_path(radar_root: Path, metadata: dict) -> Path:
    timestamp = _safe_metadata_component(metadata.get("timestamp"), "timestamp")
    if not timestamp.isdigit():
        raise ValueError(
            f"Sample metadata timestamp must be a positive integer, got {timestamp!r}"
        )
    destination = Path(radar_root) / scene_name_from_metadata(metadata)
    session = session_name_from_metadata(metadata)
    if session:
        destination /= session
    return destination / f"{timestamp}.npz"


def scene_session_root(hercules_root: Path, metadata: dict) -> Path:
    scene_root = Path(hercules_root) / scene_name_from_metadata(metadata)
    session = session_name_from_metadata(metadata)
    source_root = scene_root / session if session else scene_root
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"HeRCULES source scene/session does not exist: {source_root}"
        )
    return source_root


def read_continental_bin(path: Path) -> np.ndarray:
    byte_count = path.stat().st_size
    if byte_count % CONTINENTAL_DTYPE.itemsize:
        raise ValueError(
            f"Malformed Continental radar file {path}: {byte_count} bytes is not "
            f"divisible by the {CONTINENTAL_DTYPE.itemsize}-byte record size"
        )
    records = np.fromfile(path, dtype=CONTINENTAL_DTYPE)
    if records.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    points = np.column_stack([records[name] for name in CONTINENTAL_DTYPE.names]).astype(
        np.float32
    )
    return points[np.isfinite(points).all(axis=1)]


def load_named_transform(path: Path, key: str) -> np.ndarray:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(f"{key}:"):
            values = np.fromstring(line.split(":", maxsplit=1)[1], sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(f"Expected 12 transform values in {path}, found {values.size}")
            if not np.isfinite(values).all():
                raise ValueError(f"Transform {key} in {path} contains non-finite values")
            transform = np.eye(4, dtype=np.float64)
            transform[:3] = values.reshape(3, 4)
            if abs(np.linalg.det(transform[:3, :3])) < 1e-10:
                raise ValueError(f"Transform {key} in {path} has a singular rotation block")
            return transform
    raise ValueError(f"{key} was not found in {path}")


def load_lidar_to_radar_transform(path: Path) -> np.ndarray:
    return load_named_transform(path, "Tr_lidar_to_radar")


def transform_xyz(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {points.shape}")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError("points contains non-finite XYZ coordinates")
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.column_stack([points[:, :3], np.ones(len(points), dtype=np.float64)])
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("Pose requires a 3-vector position and xyzw quaternion")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("Ground-truth pose contains non-finite values")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("Ground-truth pose contains a zero quaternion")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


@lru_cache(maxsize=128)
def load_ground_truth_poses(path_text: str) -> tuple[tuple[int, ...], np.ndarray]:
    values = np.loadtxt(path_text, dtype=str)
    if values.ndim == 1:
        values = values[None]
    if values.shape[1] != 8:
        raise ValueError(f"Expected [timestamp,x,y,z,qx,qy,qz,qw] in {path_text}")
    timestamps = tuple(int(value) for value in values[:, 0])
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(
            f"Ground-truth timestamps in {path_text} must be strictly increasing and unique"
        )
    pose_values = values[:, 1:].astype(np.float64)
    if not np.isfinite(pose_values).all():
        raise ValueError(f"Ground-truth poses in {path_text} contain non-finite values")
    transforms = np.stack(
        [pose_matrix(row[:3], row[3:7]) for row in pose_values],
        axis=0,
    )
    return timestamps, transforms


def nearest_ground_truth_pose(path: Path, timestamp: int, max_delta_ms: float = 20.0) -> tuple[np.ndarray, float]:
    if max_delta_ms < 0.0:
        raise ValueError("max_delta_ms must be non-negative")
    timestamps, transforms = load_ground_truth_poses(str(path.resolve()))
    insertion = bisect_left(timestamps, timestamp)
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    if not candidates:
        raise FileNotFoundError(f"No ground-truth poses in {path}")
    best = min(candidates, key=lambda index: abs(timestamps[index] - timestamp))
    delta_ms = (timestamps[best] - timestamp) / 1_000_000.0
    if abs(delta_ms) > max_delta_ms:
        raise RadarAlignmentUnavailableError(
            f"Nearest pose in {path.name} is {abs(delta_ms):.1f} ms from timestamp {timestamp}"
        )
    return transforms[best], delta_ms


def find_named_directory(root: Path, name: str) -> Path:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() == name.lower() and any(path.glob("*.bin"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No {name} radar directory containing .bin files under {root}")
    return min(candidates, key=lambda path: len(path.parts))


def find_named_file(root: Path, name: str) -> Path:
    candidates = [path for path in root.rglob("*") if path.is_file() and path.name.lower() == name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No {name} under {root}")
    return min(candidates, key=lambda path: len(path.parts))


@lru_cache(maxsize=64)
def scene_radar_resources(scene_root_text: str) -> tuple[tuple[int, ...], tuple[str, ...], np.ndarray]:
    scene_root = Path(scene_root_text)
    radar_dir = find_named_directory(scene_root, "continental")
    radar_paths = sorted(radar_dir.glob("*.bin"), key=lambda path: int(path.stem))
    timestamps = tuple(int(path.stem) for path in radar_paths)
    if not timestamps:
        raise FileNotFoundError(f"No radar .bin files found in {radar_dir}")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f"Duplicate radar timestamps found in {radar_dir}")
    calibration_path = find_named_file(scene_root, "Continental_LiDAR.txt")
    radar_to_lidar = np.linalg.inv(load_lidar_to_radar_transform(calibration_path))
    return timestamps, tuple(str(path) for path in radar_paths), radar_to_lidar


def nearest_radar_frame(scene_root: Path, lidar_timestamp: int, max_delta_ms: float) -> tuple[Path, float, np.ndarray]:
    if max_delta_ms < 0.0:
        raise ValueError("max_delta_ms must be non-negative")
    timestamps, path_texts, radar_to_lidar = scene_radar_resources(str(scene_root.resolve()))
    insertion = bisect_left(timestamps, lidar_timestamp)
    candidate_indices = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    if not candidate_indices:
        raise FileNotFoundError(f"No radar timestamps under {scene_root}")
    best = min(candidate_indices, key=lambda index: abs(timestamps[index] - lidar_timestamp))
    delta_ms = (timestamps[best] - lidar_timestamp) / 1_000_000.0
    if abs(delta_ms) > max_delta_ms:
        raise RadarAlignmentUnavailableError(
            f"Nearest radar frame is {abs(delta_ms):.1f} ms from LiDAR frame; "
            f"limit is {max_delta_ms:.1f} ms"
        )
    return Path(path_texts[best]), delta_ms, radar_to_lidar


def historical_radar_frames(
    scene_root: Path,
    lidar_timestamp: int,
    frame_count: int,
    max_delta_ms: float,
    require_full_stack: bool = False,
) -> tuple[list[Path], float]:
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if max_delta_ms < 0.0:
        raise ValueError("max_delta_ms must be non-negative")
    timestamps, path_texts, _ = scene_radar_resources(str(scene_root.resolve()))
    current_index = bisect_right(timestamps, lidar_timestamp) - 1
    if current_index < 0:
        raise RadarAlignmentUnavailableError(
            f"No causal radar frame exists at or before LiDAR timestamp "
            f"{lidar_timestamp}"
        )
    delta_ms = (timestamps[current_index] - lidar_timestamp) / 1_000_000.0
    if -delta_ms > max_delta_ms:
        raise RadarAlignmentUnavailableError(
            f"Latest causal radar frame is {-delta_ms:.1f} ms before LiDAR frame; "
            f"limit is {max_delta_ms:.1f} ms"
        )
    start = current_index - frame_count + 1
    if start < 0 and require_full_stack:
        raise RadarAlignmentUnavailableError(
            f"Only {current_index + 1} causal radar frames exist before LiDAR timestamp "
            f"{lidar_timestamp}; {frame_count} are required"
        )
    start = max(0, start)
    return [Path(path_texts[index]) for index in range(start, current_index + 1)], delta_ms


def stack_radar_in_current_lidar_frame(
    scene_root: Path,
    lidar_timestamp: int,
    radar_paths: list[Path],
) -> tuple[np.ndarray, list[dict]]:
    continental_gt = find_named_file(scene_root, "Continental_gt.txt")
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

    lidar_ground_truth, lidar_pose_delta_ms = nearest_ground_truth_pose(
        aeva_gt, lidar_timestamp
    )
    world_from_current_lidar = np.eye(4, dtype=np.float64)
    world_from_current_lidar[:3, :3] = (
        lidar_ground_truth[:3, :3] @ imu_from_lidar_rotation
    )
    world_from_current_lidar[:3, 3] = lidar_ground_truth[:3, 3]
    current_lidar_from_world = np.linalg.inv(world_from_current_lidar)
    aligned_clouds = []
    alignment_rows = []
    for radar_path in radar_paths:
        radar_timestamp = int(radar_path.stem)
        radar_ground_truth, radar_pose_delta_ms = nearest_ground_truth_pose(
            continental_gt, radar_timestamp
        )
        world_from_radar = np.eye(4, dtype=np.float64)
        world_from_radar[:3, :3] = (
            radar_ground_truth[:3, :3] @ imu_from_radar_rotation
        )
        world_from_radar[:3, 3] = radar_ground_truth[:3, 3]
        current_lidar_from_radar = current_lidar_from_world @ world_from_radar
        points = read_continental_bin(radar_path)
        if len(points):
            aligned = points.copy()
            aligned[:, :3] = transform_xyz(points[:, :3], current_lidar_from_radar)
            aligned_clouds.append(aligned)
        alignment_rows.append(
            {
                "radar_timestamp": str(radar_timestamp),
                "age_ms": (lidar_timestamp - radar_timestamp) / 1_000_000.0,
                "radar_pose_delta_ms": radar_pose_delta_ms,
                "lidar_pose_delta_ms": lidar_pose_delta_ms,
            }
        )
    if not aligned_clouds:
        return np.empty((0, 8), dtype=np.float32), alignment_rows
    return np.concatenate(aligned_clouds, axis=0).astype(np.float32), alignment_rows


def project_radar_bev(
    radar_points: np.ndarray,
    radar_to_lidar: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: float,
    max_abs_velocity: float = 30.0,
) -> np.ndarray:
    radar_points = np.asarray(radar_points)
    if radar_points.ndim != 2 or radar_points.shape[1] < 6:
        raise ValueError(
            f"radar_points must have shape [N,>=6], got {radar_points.shape}"
        )
    if not np.isfinite(radar_points[:, :6]).all():
        raise ValueError("radar_points contains non-finite values")
    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError("x_range and y_range must each contain two values")
    if x_range[1] <= x_range[0] or y_range[1] <= y_range[0]:
        raise ValueError("Radar BEV range maxima must exceed their minima")
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("Radar BEV resolution must be finite and positive")
    if not np.isfinite(max_abs_velocity) or max_abs_velocity <= 0.0:
        raise ValueError("max_abs_velocity must be finite and positive")
    height = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    width = int(np.ceil((y_range[1] - y_range[0]) / resolution))
    output = np.zeros((4, height, width), dtype=np.float32)
    if len(radar_points) == 0:
        return output

    xyz = transform_xyz(radar_points[:, :3], radar_to_lidar)
    valid = (
        (xyz[:, 0] >= x_range[0])
        & (xyz[:, 0] < x_range[1])
        & (xyz[:, 1] >= y_range[0])
        & (xyz[:, 1] < y_range[1])
    )
    xyz = xyz[valid]
    if len(xyz) == 0:
        return output
    velocity = np.abs(radar_points[valid, 3])
    rcs = radar_points[valid, 5]
    cols = np.floor((xyz[:, 1] - y_range[0]) / resolution).astype(np.int32)
    rows_from_bottom = np.floor((xyz[:, 0] - x_range[0]) / resolution).astype(np.int32)
    rows = height - 1 - rows_from_bottom
    in_grid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows, cols = rows[in_grid], cols[in_grid]
    velocity, rcs = velocity[in_grid], rcs[in_grid]

    density = np.zeros((height, width), dtype=np.float32)
    output[0, rows, cols] = 1.0
    np.add.at(density, (rows, cols), 1.0)
    np.maximum.at(output[2], (rows, cols), velocity)
    np.maximum.at(output[3], (rows, cols), rcs)
    logged_density = np.log1p(density)
    if logged_density.max() > 0:
        output[1] = logged_density / logged_density.max()
    output[2] = np.clip(output[2] / max(max_abs_velocity, 1e-6), 0.0, 1.0)
    output[3] = np.clip(output[3] / 255.0, 0.0, 1.0)
    return output


def radar_cache_is_compatible(
    cache_path: Path,
    sample_metadata: dict,
    *,
    max_delta_ms: float | None = None,
    max_abs_velocity: float | None = None,
    radar_frame_count: int | None = None,
    require_full_stack: bool = False,
) -> bool:
    """Return whether a radar cache is readable and matches the requested build."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if not {"radar_bev", "metadata_json"}.issubset(data.files):
                return False
            radar_bev = np.asarray(data["radar_bev"])
            cache_metadata = json.loads(str(data["metadata_json"]))

        if radar_frame_count is not None:
            if int(cache_metadata.get("cache_format_version", 0)) != RADAR_CACHE_VERSION:
                return False

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
        if str(cache_metadata.get("scene")) != scene_name_from_metadata(sample_metadata):
            return False
        if str(cache_metadata.get("session", "")) != session_name_from_metadata(
            sample_metadata
        ):
            return False
        if str(cache_metadata.get("lidar_timestamp")) != str(sample_metadata["timestamp"]):
            return False
        if not np.allclose(cache_metadata.get("x_range"), x_range):
            return False
        if not np.allclose(cache_metadata.get("y_range"), y_range):
            return False
        if not np.isclose(float(cache_metadata.get("resolution")), resolution):
            return False

        if max_abs_velocity is not None:
            cached_velocity = float(cache_metadata.get("max_abs_velocity", 30.0))
            if not np.isclose(cached_velocity, max_abs_velocity):
                return False
        if max_delta_ms is not None:
            delta_ms = abs(float(cache_metadata.get("radar_delta_ms")))
            if delta_ms > max_delta_ms + 1e-9:
                return False
        if radar_frame_count is not None:
            requested = int(
                cache_metadata.get(
                    "requested_radar_frame_count",
                    cache_metadata.get("radar_frame_count", 1),
                )
            )
            actual = int(cache_metadata.get("radar_frame_count", requested))
            if requested != radar_frame_count or actual > radar_frame_count:
                return False
            if require_full_stack and actual != radar_frame_count:
                return False
        return True
    except Exception:
        return False


def build_radar_cache_entry(
    sample_path: Path,
    hercules_root: Path,
    radar_root: Path,
    max_delta_ms: float = 30.0,
    max_abs_velocity: float = 30.0,
    radar_frame_count: int = 1,
    require_full_stack: bool = False,
) -> Path:
    metadata = load_sample_metadata(sample_path)
    scene = scene_name_from_metadata(metadata)
    output_path = radar_cache_path(radar_root, metadata)
    if radar_cache_is_compatible(
        output_path,
        metadata,
        max_delta_ms=max_delta_ms,
        max_abs_velocity=max_abs_velocity,
        radar_frame_count=radar_frame_count,
        require_full_stack=require_full_stack,
    ):
        return output_path

    scene_root = scene_session_root(hercules_root, metadata)
    lidar_timestamp = int(metadata["timestamp"])
    if radar_frame_count == 1:
        radar_path, delta_ms, radar_to_lidar = nearest_radar_frame(
            scene_root, lidar_timestamp, max_delta_ms
        )
        radar_points = read_continental_bin(radar_path)
        radar_paths = [radar_path]
        alignment_rows = [
            {
                "radar_timestamp": radar_path.stem,
                "age_ms": (lidar_timestamp - int(radar_path.stem)) / 1_000_000.0,
                "alignment": "static Continental_LiDAR extrinsic",
            }
        ]
    else:
        radar_paths, delta_ms = historical_radar_frames(
            scene_root,
            lidar_timestamp,
            radar_frame_count,
            max_delta_ms,
            require_full_stack=require_full_stack,
        )
        radar_points, alignment_rows = stack_radar_in_current_lidar_frame(
            scene_root,
            lidar_timestamp,
            radar_paths,
        )
        radar_to_lidar = np.eye(4, dtype=np.float64)
    x_range = tuple(float(value) for value in metadata.get("x_range", [0.0, 64.0]))
    y_range = tuple(float(value) for value in metadata.get("y_range", [-32.0, 32.0]))
    resolution = float(metadata.get("resolution", 0.2))
    radar_bev = project_radar_bev(
        radar_points,
        radar_to_lidar,
        x_range,
        y_range,
        resolution,
        max_abs_velocity=max_abs_velocity,
    )

    cache_metadata = {
        "cache_format_version": RADAR_CACHE_VERSION,
        "scene": scene,
        "session": session_name_from_metadata(metadata),
        "lidar_timestamp": str(lidar_timestamp),
        "radar_timestamp": radar_paths[-1].stem,
        "radar_delta_ms": delta_ms,
        "radar_sources": [str(path) for path in radar_paths],
        "radar_frame_count": len(radar_paths),
        "requested_radar_frame_count": radar_frame_count,
        "leading_empty_frame_count": radar_frame_count - len(radar_paths),
        "temporal_alignment": (
            "sensor-specific PR_GT poses into current Aeva frame"
            if radar_frame_count > 1
            else "static Continental_LiDAR extrinsic"
        ),
        "alignment_rows": alignment_rows,
        "channels": ["occupancy", "log_density", "absolute_velocity", "rcs"],
        "x_range": x_range,
        "y_range": y_range,
        "resolution": resolution,
        "max_delta_ms": max_delta_ms,
        "max_abs_velocity": max_abs_velocity,
        "require_full_stack": bool(require_full_stack),
    }
    atomic_savez_compressed(
        output_path,
        radar_bev=radar_bev.astype(np.float16),
        metadata_json=np.asarray(json.dumps(cache_metadata)),
    )
    return output_path


def radar_cache_requirements_from_checkpoint(checkpoint: dict) -> dict:
    """Recover the radar-cache contract recorded by a training checkpoint."""
    saved_args = checkpoint.get("args", {})
    return {
        "max_delta_ms": saved_args.get("max_radar_delta_ms"),
        "max_abs_velocity": saved_args.get("radar_max_abs_velocity"),
        "radar_frame_count": saved_args.get("radar_frame_count"),
        "require_full_stack": bool(
            saved_args.get("require_full_radar_stack", False)
        ),
    }


def filter_samples_with_radar_cache(
    paths,
    radar_root: Path,
    *,
    max_delta_ms: float | None = None,
    max_abs_velocity: float | None = None,
    radar_frame_count: int | None = None,
    require_full_stack: bool = False,
) -> tuple[list[Path], list[Path]]:
    available = []
    missing = []
    for path in paths:
        metadata = load_sample_metadata(path)
        destination = radar_cache_path(radar_root, metadata)
        target = (
            available
            if radar_cache_is_compatible(
                destination,
                metadata,
                max_delta_ms=max_delta_ms,
                max_abs_velocity=max_abs_velocity,
                radar_frame_count=radar_frame_count,
                require_full_stack=require_full_stack,
            )
            else missing
        )
        target.append(path)
    return available, missing
