from __future__ import annotations

from bisect import bisect_left
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
    """Raised when a radar frame cannot be aligned without extrapolation."""


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
    if str(metadata.get("dataset", "")).strip().lower() in {
        "view-of-delft",
        "view of delft",
        "vod",
    }:
        split = _safe_metadata_component(metadata.get("split"), "split")
        frame_id = _safe_metadata_component(
            metadata.get("frame_id", metadata.get("radar_index")),
            "frame_id",
        )
        if not frame_id.isdigit():
            raise ValueError("View-of-Delft frame_id must be numeric")
        return Path(radar_root) / split / f"{int(frame_id):05d}.npz"
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


@lru_cache(maxsize=2048)
def _read_continental_bin_cached(path_text: str) -> np.ndarray:
    path = Path(path_text)
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


def read_continental_bin(path: Path) -> np.ndarray:
    return _read_continental_bin_cached(str(Path(path).resolve())).copy()


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
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower() == name.lower()
    ]
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
        return True
    except Exception:
        return False


def build_radar_cache_entry(
    sample_path: Path,
    hercules_root: Path,
    radar_root: Path,
    max_delta_ms: float = 30.0,
    max_abs_velocity: float = 30.0,
) -> Path:
    metadata = load_sample_metadata(sample_path)
    scene = scene_name_from_metadata(metadata)
    output_path = radar_cache_path(radar_root, metadata)
    if radar_cache_is_compatible(
        output_path,
        metadata,
        max_delta_ms=max_delta_ms,
        max_abs_velocity=max_abs_velocity,
    ):
        return output_path

    scene_root = scene_session_root(hercules_root, metadata)
    lidar_timestamp = int(metadata["timestamp"])
    radar_path, delta_ms, radar_to_lidar = nearest_radar_frame(
        scene_root, lidar_timestamp, max_delta_ms
    )
    radar_points = read_continental_bin(radar_path)
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
        "radar_timestamp": radar_path.stem,
        "radar_delta_ms": delta_ms,
        "radar_source": str(radar_path),
        "alignment": "static Continental_LiDAR extrinsic",
        "channels": ["occupancy", "log_density", "absolute_velocity", "rcs"],
        "x_range": x_range,
        "y_range": y_range,
        "resolution": resolution,
        "max_delta_ms": max_delta_ms,
        "max_abs_velocity": max_abs_velocity,
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
    }


def filter_samples_with_radar_cache(
    paths,
    radar_root: Path,
    *,
    max_delta_ms: float | None = None,
    max_abs_velocity: float | None = None,
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
            )
            else missing
        )
        target.append(path)
    return available, missing
