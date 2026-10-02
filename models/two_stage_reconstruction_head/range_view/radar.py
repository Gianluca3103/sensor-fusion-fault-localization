"""Project already calibrated LiDAR-frame radar evidence into angular cells."""

from __future__ import annotations

import numpy as np

from .geometry import RangeGeometry, angular_indices


RADAR_FEATURE_NAMES = (
    "radar_valid", "radar_log_count", "radar_nearest_range",
    "radar_mean_rcs", "radar_mean_radial_velocity", "radar_mean_z",
)


def filter_radar_below_lidar(
    points_lidar_frame: np.ndarray, lidar_points: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """Keep radar at or above the lowest observed LiDAR return in this frame.

    The LiDAR argument must be the available, faulty input, never the clean
    supervision target. An empty LiDAR scan provides no safe threshold.
    """
    radar = np.asarray(points_lidar_frame, dtype=np.float32)
    lidar = np.asarray(lidar_points, dtype=np.float32)
    if radar.ndim != 2 or radar.shape[1] != 5:
        raise ValueError("aligned radar must contain [x,y,z,rcs,compensated_velocity]")
    if lidar.ndim != 2 or lidar.shape[1] < 3:
        raise ValueError("LiDAR input must contain XYZ columns")
    finite_lidar_z = lidar[np.isfinite(lidar[:, 2]), 2]
    if len(finite_lidar_z) == 0:
        return radar, None
    minimum_z = float(finite_lidar_z.min())
    return radar[radar[:, 2] >= minimum_z], minimum_z


def filter_radar_floor_band(points_lidar_frame: np.ndarray, band_m: float) -> tuple[np.ndarray, float | None]:
    """Drop radar points within ``band_m`` above this frame's minimum radar z.

    The floor proxy is deliberately the minimum *radar* height, not a clean
    LiDAR-derived plane. A zero band disables the ablation entirely.
    """
    points = np.asarray(points_lidar_frame, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError("aligned radar cache must contain [x,y,z,rcs,compensated_velocity]")
    if not np.isfinite(band_m) or band_m < 0:
        raise ValueError("radar floor band must be finite and nonnegative")
    if band_m == 0 or len(points) == 0:
        return points, None
    finite_z = np.isfinite(points[:, 2])
    if not finite_z.any():
        return points, None
    floor_z = float(points[finite_z, 2].min())
    keep = ~finite_z | (points[:, 2] > floor_z + band_m)
    return points[keep], floor_z


def project_aligned_radar(points_lidar_frame: np.ndarray, geometry: RangeGeometry) -> np.ndarray:
    """Aggregate all returns; input fields are xyz, RCS, compensated Doppler."""
    points = np.asarray(points_lidar_frame, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError("aligned radar cache must contain [x,y,z,rcs,compensated_velocity]")
    height, width = geometry.shape
    result = np.zeros((len(RADAR_FEATURE_NAMES), height, width), dtype=np.float32)
    row, col, radius, valid = angular_indices(points, geometry, require_beam_match=False)
    if not np.any(valid):
        return result
    flat = row[valid].astype(np.int64) * width + col[valid]
    cells = height * width
    count = np.bincount(flat, minlength=cells).astype(np.float32)
    used = count > 0
    result[0].reshape(-1)[used] = 1.0
    result[1].reshape(-1)[:] = np.log1p(count)
    nearest = np.full(cells, np.inf, dtype=np.float32)
    np.minimum.at(nearest, flat, radius[valid])
    result[2].reshape(-1)[used] = nearest[used] / geometry.max_range_m
    for channel, values in ((3, points[valid, 3]), (4, points[valid, 4]), (5, points[valid, 2])):
        sums = np.bincount(flat, weights=values, minlength=cells).astype(np.float32)
        result[channel].reshape(-1)[used] = sums[used] / count[used]
    return result
