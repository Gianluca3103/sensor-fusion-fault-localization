"""Read and align the KITTI-format View-of-Delft point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


VOD_LIDAR_FIELDS = ("x", "y", "z", "reflectivity")
VOD_RADAR_FIELDS = (
    "x",
    "y",
    "z",
    "rcs",
    "radial_velocity",
    "compensated_radial_velocity",
    "time_index",
)
SUPPORTED_RADAR_VARIANTS = (
    "radar",
    "radar_3frames",
    "radar_5frames",
    "radar_10frames",
    "radar_20frames",
)


@dataclass(frozen=True)
class VODFrame:
    frame_id: str
    split: str
    lidar_path: Path
    radar_path: Path
    lidar_calibration_path: Path
    radar_calibration_path: Path
    radar_variant: str


def _read_float32_rows(path: str | Path, columns: int, label: str) -> np.ndarray:
    path = Path(path)
    byte_count = path.stat().st_size
    row_bytes = columns * np.dtype(np.float32).itemsize
    if byte_count % row_bytes:
        raise ValueError(
            f"Malformed VoD {label} file {path}: {byte_count} bytes is not "
            f"divisible by the {row_bytes}-byte row size"
        )
    points = np.fromfile(path, dtype=np.float32).reshape(-1, columns)
    if not np.isfinite(points).all():
        raise ValueError(f"VoD {label} file contains NaN or Inf: {path}")
    return points


def load_vod_lidar(path: str | Path) -> np.ndarray:
    """Return one VoD LiDAR frame as ``[x,y,z,reflectivity]``."""

    return _read_float32_rows(path, len(VOD_LIDAR_FIELDS), "LiDAR")


def load_vod_radar(path: str | Path) -> np.ndarray:
    """Return one VoD radar frame using the official seven-column contract."""

    return _read_float32_rows(path, len(VOD_RADAR_FIELDS), "radar")


def _named_transform(path: str | Path, key: str = "Tr_velo_to_cam") -> np.ndarray:
    path = Path(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        name, separator, values_text = raw_line.partition(":")
        if separator and name.strip() == key:
            values = np.fromstring(values_text, sep=" ", dtype=np.float64)
            if values.size != 12 or not np.isfinite(values).all():
                raise ValueError(f"Malformed {key} in {path}")
            transform = np.eye(4, dtype=np.float64)
            transform[:3] = values.reshape(3, 4)
            if abs(np.linalg.det(transform[:3, :3])) < 1e-10:
                raise ValueError(f"Singular {key} rotation in {path}")
            return transform
    raise ValueError(f"{key} was not found in {path}")


@lru_cache(maxsize=4096)
def _load_vod_radar_to_lidar_cached(
    lidar_calibration_text: str,
    radar_calibration_text: str,
) -> np.ndarray:
    camera_from_lidar = _named_transform(lidar_calibration_text)
    camera_from_radar = _named_transform(radar_calibration_text)
    lidar_from_radar = np.linalg.inv(camera_from_lidar) @ camera_from_radar
    lidar_from_radar.setflags(write=False)
    return lidar_from_radar


def load_vod_radar_to_lidar(
    lidar_calibration_path: str | Path,
    radar_calibration_path: str | Path,
) -> np.ndarray:
    """Derive the radar-to-LiDAR transform through the common camera frame."""

    return _load_vod_radar_to_lidar_cached(
        str(Path(lidar_calibration_path).resolve()),
        str(Path(radar_calibration_path).resolve()),
    ).copy()


def align_radar_to_lidar(
    radar_points: np.ndarray,
    lidar_from_radar: np.ndarray,
) -> np.ndarray:
    """Transform VoD radar XYZ while preserving RCS, velocities, and time."""

    radar_points = np.asarray(radar_points, dtype=np.float32)
    transform = np.asarray(lidar_from_radar, dtype=np.float64)
    if radar_points.ndim != 2 or radar_points.shape[1] != len(VOD_RADAR_FIELDS):
        raise ValueError(
            f"radar_points must have shape [N,{len(VOD_RADAR_FIELDS)}], got "
            f"{radar_points.shape}"
        )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("lidar_from_radar must be a finite 4x4 transform")
    homogeneous = np.column_stack(
        (radar_points[:, :3], np.ones(len(radar_points), dtype=np.float64))
    )
    aligned = radar_points.copy()
    aligned[:, :3] = (homogeneous @ transform.T)[:, :3].astype(np.float32)
    return aligned


def _split_ids(public_root: Path, split: str) -> list[str]:
    split_path = public_root / "lidar" / "ImageSets" / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"VoD split file is missing: {split_path}")
    identifiers = [line.strip() for line in split_path.read_text().splitlines()]
    identifiers = [identifier for identifier in identifiers if identifier]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"VoD split contains duplicate frame IDs: {split_path}")
    return identifiers


def discover_vod_frames(
    vod_root: str | Path,
    split: str,
    *,
    radar_variant: str = "radar_3frames",
) -> list[VODFrame]:
    """Return exact LiDAR/radar pairs from an official VoD ImageSets split."""

    if split not in {"train", "val", "test", "train_val", "full"}:
        raise ValueError(f"Unsupported VoD split: {split}")
    if radar_variant not in SUPPORTED_RADAR_VARIANTS:
        raise ValueError(
            f"Unsupported VoD radar variant {radar_variant!r}; expected one of "
            f"{SUPPORTED_RADAR_VARIANTS}"
        )
    public_root = Path(vod_root)
    if (public_root / "view_of_delft_PUBLIC").is_dir():
        public_root /= "view_of_delft_PUBLIC"

    lidar_root = public_root / "lidar" / "training"
    radar_root = public_root / radar_variant / "training"
    # Accumulated releases may omit duplicate calibration files. Their points
    # use the same radar frame as the single-scan release.
    radar_calibration_root = radar_root / "calib"
    if not any(radar_calibration_root.glob("*.txt")):
        radar_calibration_root = public_root / "radar" / "training" / "calib"

    frames = []
    missing = []
    for frame_id in _split_ids(public_root, split):
        paths = (
            lidar_root / "velodyne" / f"{frame_id}.bin",
            radar_root / "velodyne" / f"{frame_id}.bin",
            lidar_root / "calib" / f"{frame_id}.txt",
            radar_calibration_root / f"{frame_id}.txt",
        )
        if not all(path.is_file() for path in paths):
            missing.append((frame_id, [str(path) for path in paths if not path.is_file()]))
            continue
        frames.append(
            VODFrame(
                frame_id=frame_id,
                split=split,
                lidar_path=paths[0],
                radar_path=paths[1],
                lidar_calibration_path=paths[2],
                radar_calibration_path=paths[3],
                radar_variant=radar_variant,
            )
        )
    if missing:
        first_id, first_paths = missing[0]
        raise FileNotFoundError(
            f"VoD split {split} has {len(missing)} incomplete paired frames. "
            f"First: {first_id}: {first_paths}"
        )
    if not frames:
        raise FileNotFoundError(
            f"No paired VoD {radar_variant} frames found under {public_root}"
        )
    return frames
