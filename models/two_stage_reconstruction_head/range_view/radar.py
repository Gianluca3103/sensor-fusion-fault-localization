"""Project already calibrated LiDAR-frame radar evidence into angular cells."""

from __future__ import annotations

import numpy as np

from .geometry import RangeGeometry, angular_indices


RADAR_FEATURE_NAMES = (
    "radar_valid", "radar_log_count", "radar_nearest_range",
    "radar_mean_rcs", "radar_mean_radial_velocity", "radar_mean_z",
)


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
