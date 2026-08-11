"""Read K-Radar LiDAR PCD files and apply the radar-overlap geometry."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


K_RADAR_AZIMUTH_RANGE_RAD = (-0.9250245094299316, 0.9250245094299316)
K_RADAR_ELEVATION_RANGE_RAD = (-0.3141592741012573, 0.3141592741012573)
K_RADAR_RANGE_M = (0.0, 118.037109375)


def radar_bev_support_mask(
    shape: tuple[int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    radar_from_lidar: np.ndarray,
    *,
    azimuth_range_rad: tuple[float, float] = K_RADAR_AZIMUTH_RANGE_RAD,
    radar_range_m: tuple[float, float] = K_RADAR_RANGE_M,
) -> np.ndarray:
    """Return the nominal horizontal radar support at BEV-cell centers."""

    height, width = (int(value) for value in shape)
    if height < 1 or width < 1:
        raise ValueError(f"shape must be positive, got {shape}")
    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("BEV ranges must be increasing")
    radar_from_lidar = np.asarray(radar_from_lidar, dtype=np.float64)
    if radar_from_lidar.shape != (4, 4):
        raise ValueError(
            f"radar_from_lidar must have shape [4,4], got {radar_from_lidar.shape}"
        )

    row_centers = x_max - (
        np.arange(height, dtype=np.float64) + 0.5
    ) * ((x_max - x_min) / height)
    column_centers = y_min + (
        np.arange(width, dtype=np.float64) + 0.5
    ) * ((y_max - y_min) / width)
    lidar_x, lidar_y = np.meshgrid(
        row_centers,
        column_centers,
        indexing="ij",
    )
    radar_x = lidar_x + radar_from_lidar[0, 3]
    radar_y = lidar_y + radar_from_lidar[1, 3]
    horizontal_range = np.hypot(radar_x, radar_y)
    azimuth = np.arctan2(radar_y, radar_x)
    return (
        (horizontal_range >= radar_range_m[0])
        & (horizontal_range <= radar_range_m[1])
        & (azimuth >= azimuth_range_rad[0])
        & (azimuth <= azimuth_range_rad[1])
    )


def _pcd_header(path: Path) -> tuple[list[str], int, int]:
    fields = None
    point_count = None
    header_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, *values = stripped.split()
            key = key.upper()
            if key == "FIELDS":
                fields = values
            elif key == "POINTS":
                point_count = int(values[0])
            elif key == "DATA":
                if values != ["ascii"]:
                    raise ValueError(
                        f"K-Radar LiDAR PCD must use DATA ascii, got {values} in {path}"
                    )
                break
        else:
            raise ValueError(f"K-Radar PCD has no DATA header: {path}")
    if fields is None or point_count is None:
        raise ValueError(f"K-Radar PCD is missing FIELDS or POINTS: {path}")
    return fields, point_count, header_lines


def read_kradar_lidar_pcd(
    path: str | Path,
    *,
    include_reflectivity: bool = False,
) -> np.ndarray:
    """Load K-Radar os2-64 XYZI, optionally followed by reflectivity."""

    path = Path(path)
    fields, point_count, header_lines = _pcd_header(path)
    required = (
        ("x", "y", "z", "intensity", "reflectivity")
        if include_reflectivity
        else ("x", "y", "z", "intensity")
    )
    try:
        columns = tuple(fields.index(name) for name in required)
    except ValueError as exc:
        raise ValueError(
            f"K-Radar PCD {path} must contain fields {required}, got {fields}"
        ) from exc
    points = np.loadtxt(
        path,
        dtype=np.float32,
        skiprows=header_lines,
        usecols=columns,
        ndmin=2,
    )
    if len(points) != point_count:
        raise ValueError(
            f"K-Radar PCD {path} declares {point_count} points but contains {len(points)}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"K-Radar PCD contains non-finite XYZI values: {path}")
    return points


@lru_cache(maxsize=64)
def load_radar_from_lidar_transform(
    calibration_path: str | Path,
    radar_to_lidar_z_m: float = 0.7,
) -> np.ndarray:
    """Load K-Radar's translation-only transform from LiDAR into radar."""

    calibration_path = Path(calibration_path)
    lines = calibration_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Malformed K-Radar calibration file: {calibration_path}")
    values = [float(value.strip()) for value in lines[1].split(",")]
    if len(values) < 3:
        raise ValueError(f"Malformed K-Radar calibration row: {lines[1]!r}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (values[1], values[2], radar_to_lidar_z_m)
    transform.setflags(write=False)
    return transform


def radar_overlap_mask(
    points: np.ndarray,
    radar_from_lidar: np.ndarray,
    *,
    azimuth_range_rad: tuple[float, float] = K_RADAR_AZIMUTH_RANGE_RAD,
    elevation_range_rad: tuple[float, float] = K_RADAR_ELEVATION_RANGE_RAD,
    radar_range_m: tuple[float, float] = K_RADAR_RANGE_M,
) -> np.ndarray:
    """Return points falling inside K-Radar's horizontal and radial support."""

    points = np.asarray(points)
    radar_from_lidar = np.asarray(radar_from_lidar, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {points.shape}")
    if radar_from_lidar.shape != (4, 4):
        raise ValueError(
            f"radar_from_lidar must have shape [4,4], got {radar_from_lidar.shape}"
        )
    finite = np.isfinite(points[:, :3]).all(axis=1)
    radar_xyz = points[:, :3].astype(np.float64, copy=False) + radar_from_lidar[
        :3, 3
    ]
    ranges = np.linalg.norm(radar_xyz, axis=1)
    azimuth = np.arctan2(radar_xyz[:, 1], radar_xyz[:, 0])
    elevation = np.arctan2(
        radar_xyz[:, 2],
        np.linalg.norm(radar_xyz[:, :2], axis=1),
    )
    return (
        finite
        & (ranges >= radar_range_m[0])
        & (ranges <= radar_range_m[1])
        & (azimuth >= azimuth_range_rad[0])
        & (azimuth <= azimuth_range_rad[1])
        & (elevation >= elevation_range_rad[0])
        & (elevation <= elevation_range_rad[1])
    )
