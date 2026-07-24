from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


CONTINENTAL_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "velocity", "range", "rcs", "azimuth", "elevation"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<f4", "u1", "<f4", "<f4"],
        "offsets": [0, 4, 8, 12, 16, 20, 21, 25],
        "itemsize": 29,
    }
)


class RadarAlignmentUnavailableError(ValueError):
    """Raised when a causal radar stack cannot be aligned without extrapolation."""


def load_sample_metadata(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        if "metadata_json" not in data:
            raise KeyError(f"{npz_path} does not contain metadata_json")
        return json.loads(str(data["metadata_json"]))


def scene_name_from_metadata(metadata: dict) -> str:
    scene = str(metadata.get("scene") or metadata.get("day") or "").strip()
    if scene:
        return scene
    relative = str(metadata.get("source_relative_path", "")).replace("\\", "/")
    if relative:
        return relative.split("/", maxsplit=1)[0]
    raise KeyError("Sample metadata does not identify its HeRCULES scene")


def radar_cache_path(radar_root: Path, metadata: dict) -> Path:
    return radar_root / scene_name_from_metadata(metadata) / f"{metadata['timestamp']}.npz"


def read_continental_bin(path: Path) -> np.ndarray:
    records = np.fromfile(path, dtype=CONTINENTAL_DTYPE)
    if records.size == 0:
        return np.empty((0, 8), dtype=np.float32)
    return np.column_stack([records[name] for name in CONTINENTAL_DTYPE.names]).astype(np.float32)


def load_named_transform(path: Path, key: str) -> np.ndarray:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(f"{key}:"):
            values = np.fromstring(line.split(":", maxsplit=1)[1], sep=" ", dtype=np.float64)
            if values.size != 12:
                raise ValueError(f"Expected 12 transform values in {path}, found {values.size}")
            transform = np.eye(4, dtype=np.float64)
            transform[:3] = values.reshape(3, 4)
            return transform
    raise ValueError(f"{key} was not found in {path}")


def load_lidar_to_radar_transform(path: Path) -> np.ndarray:
    return load_named_transform(path, "Tr_lidar_to_radar")


def transform_xyz(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.column_stack([points[:, :3], np.ones(len(points), dtype=np.float64)])
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
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
    pose_values = values[:, 1:].astype(np.float64)
    transforms = np.stack(
        [pose_matrix(row[:3], row[3:7]) for row in pose_values],
        axis=0,
    )
    return timestamps, transforms


def nearest_ground_truth_pose(path: Path, timestamp: int, max_delta_ms: float = 20.0) -> tuple[np.ndarray, float]:
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
    calibration_path = find_named_file(scene_root, "Continental_LiDAR.txt")
    radar_to_lidar = np.linalg.inv(load_lidar_to_radar_transform(calibration_path))
    return timestamps, tuple(str(path) for path in radar_paths), radar_to_lidar


def nearest_radar_frame(scene_root: Path, lidar_timestamp: int, max_delta_ms: float) -> tuple[Path, float, np.ndarray]:
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
    timestamps, path_texts, _ = scene_radar_resources(str(scene_root.resolve()))
    insertion = bisect_left(timestamps, lidar_timestamp)
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    if not candidates:
        raise FileNotFoundError(f"No radar timestamps under {scene_root}")
    current_index = min(candidates, key=lambda index: abs(timestamps[index] - lidar_timestamp))
    delta_ms = (timestamps[current_index] - lidar_timestamp) / 1_000_000.0
    if abs(delta_ms) > max_delta_ms:
        raise RadarAlignmentUnavailableError(
            f"Nearest radar frame is {abs(delta_ms):.1f} ms from LiDAR frame; "
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
    continental_gt = find_named_file(scene_root / "PR_GT", "Continental_gt.txt")
    aeva_gt = find_named_file(scene_root / "PR_GT", "Aeva_gt.txt")
    lidar_to_imu = load_named_transform(
        find_named_file(scene_root / "Calibration", "IMU_LiDAR.txt"),
        "Tr_lidar_to_imu",
    )
    lidar_to_radar = load_lidar_to_radar_transform(
        find_named_file(scene_root / "Calibration", "Continental_LiDAR.txt")
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
    if output_path.exists():
        return output_path

    scene_root = hercules_root / scene
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
        "scene": scene,
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
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        radar_bev=radar_bev.astype(np.float16),
        metadata_json=np.asarray(json.dumps(cache_metadata)),
    )
    temporary.replace(output_path)
    return output_path


def filter_samples_with_radar_cache(paths, radar_root: Path) -> tuple[list[Path], list[Path]]:
    available = []
    missing = []
    for path in paths:
        destination = radar_cache_path(radar_root, load_sample_metadata(path))
        (available if destination.exists() else missing).append(path)
    return available, missing
