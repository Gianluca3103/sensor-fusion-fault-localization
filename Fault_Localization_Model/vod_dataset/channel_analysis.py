"""Analysis-only BEV channels derived from aligned View-of-Delft points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Fault_Localization_Model.bev_utils import metric_to_grid


ENGINEERED_LIDAR_CHANNELS = (
    "occupancy",
    "log_point_count",
    "height_spread_p90_p10",
    "robust_upper_height_p90",
    "range_mean",
    "reflectivity_p90",
)
ENGINEERED_RADAR_CHANNELS = (
    "occupancy",
    "log_point_count",
    "height_spread_p90_p10",
    "robust_upper_height_p90",
    "range_mean",
    "rcs_p90",
    "compensated_radial_velocity_mean",
)

# Fixed dataset-level scales keep the meaning of every channel stable across
# samples. Empty cells remain zero and occupancy disambiguates them from valid
# measurements that happen to normalize to zero.
_LOG_COUNT_MAX = np.log1p(64.0)
_HEIGHT_MIN_M = -3.0
_HEIGHT_MAX_M = 5.0
_MAX_RANGE_M = 80.0
_LIDAR_REFLECTIVITY_MAX = 255.0
_RADAR_RCS_MIN_DB = -70.0
_RADAR_RCS_MAX_DB = 60.0
_RADAR_VELOCITY_LIMIT_MPS = 30.0


@dataclass(frozen=True)
class BEVGeometry:
    x_range: tuple[float, float] = (0.0, 64.0)
    y_range: tuple[float, float] = (-32.0, 32.0)
    resolution: float = 0.2


def _quantile_grid(
    cells: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    quantile: float,
) -> np.ndarray:
    output = np.zeros(shape[0] * shape[1], dtype=np.float32)
    if not len(cells):
        return output.reshape(shape)
    order = np.lexsort((values, cells))
    sorted_cells = cells[order]
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    ends = np.r_[starts[1:], len(sorted_cells)]
    counts = ends - starts
    offsets = np.maximum(
        np.ceil(quantile * counts).astype(np.int64) - 1,
        0,
    )
    output[sorted_cells[starts]] = sorted_values[starts + offsets]
    return output.reshape(shape)


def _moment_grids(
    cells: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = shape[0] * shape[1]
    count = np.bincount(cells, minlength=size).astype(np.float32)
    total = np.bincount(cells, weights=values, minlength=size).astype(np.float32)
    squared = np.bincount(
        cells, weights=np.square(values), minlength=size
    ).astype(np.float32)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(
        squared,
        count,
        out=np.zeros_like(squared),
        where=count > 0,
    ) - np.square(mean)
    variance = np.maximum(variance, 0.0)
    maximum = np.full(size, -np.inf, dtype=np.float32)
    if len(cells):
        np.maximum.at(maximum, cells, values)
    maximum[~np.isfinite(maximum)] = 0.0
    return (
        mean.reshape(shape),
        variance.reshape(shape),
        np.sqrt(variance).reshape(shape),
        maximum.reshape(shape),
    )


def _prepared_grid(
    xyz: np.ndarray,
    geometry: BEVGeometry,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
    valid_xyz, rows, cols, valid, height, width = metric_to_grid(
        xyz,
        geometry.x_range,
        geometry.y_range,
        geometry.resolution,
    )
    cells = rows.astype(np.int64) * width + cols.astype(np.int64)
    return valid_xyz, cells, (height, width), valid


def radar_analysis_channels(
    raw_radar_points: np.ndarray,
    aligned_radar_points: np.ndarray,
    geometry: BEVGeometry = BEVGeometry(),
) -> dict[str, np.ndarray]:
    """Rasterize the official seven-column five-frame VoD radar representation."""

    raw = np.asarray(raw_radar_points, dtype=np.float32)
    aligned = np.asarray(aligned_radar_points, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 7 or aligned.shape != raw.shape:
        raise ValueError("raw and aligned radar points must both have shape [N,7]")
    aligned_xyz, cells, shape, valid = _prepared_grid(aligned[:, :3], geometry)
    raw = raw[valid]
    aligned = aligned[valid]
    count = np.bincount(cells, minlength=shape[0] * shape[1]).astype(np.float32)
    count = count.reshape(shape)
    occupancy = (count > 0).astype(np.float32)

    rcs_mean, rcs_variance, _rcs_std, rcs_max = _moment_grids(
        cells, raw[:, 3], shape
    )
    raw_velocity_mean, _raw_velocity_var, _raw_velocity_std, _ = _moment_grids(
        cells, raw[:, 4], shape
    )
    compensated_mean, _compensated_var, _compensated_std, _ = _moment_grids(
        cells, raw[:, 5], shape
    )
    compensated_speed_mean, _, _, _ = _moment_grids(
        cells, np.abs(raw[:, 5]), shape
    )

    native_range = np.linalg.norm(raw[:, :3], axis=1)
    native_planar_range = np.linalg.norm(raw[:, :2], axis=1)
    native_azimuth = np.arctan2(raw[:, 1], raw[:, 0])
    native_elevation = np.arctan2(raw[:, 2], native_planar_range)
    range_mean, _, _, _ = _moment_grids(cells, native_range, shape)
    azimuth_mean, _, _, _ = _moment_grids(cells, native_azimuth, shape)
    elevation_mean, _, _, _ = _moment_grids(cells, native_elevation, shape)
    time_index_mean, _, _, time_index_max = _moment_grids(cells, raw[:, 6], shape)
    time_index_min = _quantile_grid(cells, raw[:, 6], shape, 0.0)

    aligned_planar_range = np.maximum(
        np.linalg.norm(aligned_xyz[:, :2], axis=1), 1.0e-6
    )
    aligned_range_mean, _, _, _ = _moment_grids(
        cells, aligned_planar_range, shape
    )
    radial_vx = raw[:, 5] * aligned_xyz[:, 0] / aligned_planar_range
    radial_vy = raw[:, 5] * aligned_xyz[:, 1] / aligned_planar_range
    radial_vx_mean, _, _, _ = _moment_grids(cells, radial_vx, shape)
    radial_vy_mean, _, _, _ = _moment_grids(cells, radial_vy, shape)

    height_p10 = _quantile_grid(cells, aligned_xyz[:, 2], shape, 0.10)
    height_p90 = _quantile_grid(cells, aligned_xyz[:, 2], shape, 0.90)
    doppler_p10 = _quantile_grid(cells, raw[:, 4], shape, 0.10)
    doppler_p90 = _quantile_grid(cells, raw[:, 4], shape, 0.90)

    return {
        "occupancy": occupancy,
        "point_count": count,
        "log_point_count": np.log1p(count).astype(np.float32),
        "rcs_mean_db": rcs_mean,
        "rcs_max_db": rcs_max,
        "rcs_p90_db": _quantile_grid(cells, raw[:, 3], shape, 0.90),
        "rcs_variance_db2": rcs_variance,
        "raw_radial_velocity_mean_mps": raw_velocity_mean,
        "compensated_radial_velocity_mean_mps": compensated_mean,
        "compensated_radial_speed_mean_mps": compensated_speed_mean,
        "radial_projection_vx_mean_mps": radial_vx_mean,
        "radial_projection_vy_mean_mps": radial_vy_mean,
        "doppler_spread_p90_p10_mps": doppler_p90 - doppler_p10,
        "robust_upper_height_p90_m": height_p90,
        "height_spread_p90_p10_m": height_p90 - height_p10,
        "native_range_mean_m": range_mean,
        "aligned_range_mean_m": aligned_range_mean,
        "native_azimuth_mean_rad": azimuth_mean,
        "native_elevation_mean_rad": elevation_mean,
        "scan_time_index_mean": time_index_mean,
        "scan_time_index_span": time_index_max - time_index_min,
    }


def lidar_analysis_channels(
    lidar_points: np.ndarray,
    geometry: BEVGeometry = BEVGeometry(),
) -> dict[str, np.ndarray]:
    """Rasterize all supported VoD LiDAR analysis channels."""

    points = np.asarray(lidar_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("VoD LiDAR points must have shape [N,4]")
    xyz, cells, shape, valid = _prepared_grid(points[:, :3], geometry)
    points = points[valid]
    count = np.bincount(cells, minlength=shape[0] * shape[1]).astype(np.float32)
    count = count.reshape(shape)
    occupancy = (count > 0).astype(np.float32)

    height_mean, height_variance, height_std, _ = _moment_grids(
        cells, xyz[:, 2], shape
    )
    reflectivity_mean, _, _, reflectivity_max = _moment_grids(
        cells, points[:, 3], shape
    )
    height_p10 = _quantile_grid(cells, xyz[:, 2], shape, 0.10)
    height_median = _quantile_grid(cells, xyz[:, 2], shape, 0.50)
    height_p90 = _quantile_grid(cells, xyz[:, 2], shape, 0.90)
    range_mean, _, _, _ = _moment_grids(
        cells, np.linalg.norm(xyz[:, :2], axis=1), shape
    )
    azimuth_mean, _, _, _ = _moment_grids(
        cells, np.arctan2(xyz[:, 1], xyz[:, 0]), shape
    )

    return {
        "occupancy": occupancy,
        "point_count": count,
        "log_point_count": np.log1p(count).astype(np.float32),
        "robust_upper_height_p90_m": height_p90,
        "height_spread_p90_p10_m": height_p90 - height_p10,
        "lower_height_p10_m": height_p10,
        "mean_height_m": height_mean,
        "median_height_m": height_median,
        "height_variance_m2": height_variance,
        "height_std_m": height_std,
        "reflectivity_mean": reflectivity_mean,
        "reflectivity_max": reflectivity_max,
        "reflectivity_p90": _quantile_grid(cells, points[:, 3], shape, 0.90),
        "range_mean_m": range_mean,
        "azimuth_mean_rad": azimuth_mean,
    }


def _normalize_supported(
    values: np.ndarray,
    occupancy: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    supported = occupancy > 0.0
    output[supported] = np.clip(
        (values[supported] - minimum) / (maximum - minimum),
        0.0,
        1.0,
    )
    return output


def lidar_model_channels(
    lidar_points: np.ndarray,
    geometry: BEVGeometry = BEVGeometry(),
) -> np.ndarray:
    """Return the six normalized handcrafted LiDAR input channels."""

    channels = lidar_analysis_channels(lidar_points, geometry)
    occupancy = channels["occupancy"]
    return np.stack(
        (
            occupancy,
            np.clip(channels["log_point_count"] / _LOG_COUNT_MAX, 0.0, 1.0),
            np.clip(
                channels["height_spread_p90_p10_m"]
                / (_HEIGHT_MAX_M - _HEIGHT_MIN_M),
                0.0,
                1.0,
            ),
            _normalize_supported(
                channels["robust_upper_height_p90_m"],
                occupancy,
                _HEIGHT_MIN_M,
                _HEIGHT_MAX_M,
            ),
            np.clip(channels["range_mean_m"] / _MAX_RANGE_M, 0.0, 1.0),
            np.clip(
                channels["reflectivity_p90"] / _LIDAR_REFLECTIVITY_MAX,
                0.0,
                1.0,
            ),
        ),
        axis=0,
    ).astype(np.float32, copy=False)


def radar_model_channels(
    raw_radar_points: np.ndarray,
    aligned_radar_points: np.ndarray,
    geometry: BEVGeometry = BEVGeometry(),
) -> np.ndarray:
    """Return the seven normalized handcrafted Radar input channels."""

    channels = radar_analysis_channels(
        raw_radar_points,
        aligned_radar_points,
        geometry,
    )
    occupancy = channels["occupancy"]
    return np.stack(
        (
            occupancy,
            np.clip(channels["log_point_count"] / _LOG_COUNT_MAX, 0.0, 1.0),
            np.clip(
                channels["height_spread_p90_p10_m"]
                / (_HEIGHT_MAX_M - _HEIGHT_MIN_M),
                0.0,
                1.0,
            ),
            _normalize_supported(
                channels["robust_upper_height_p90_m"],
                occupancy,
                _HEIGHT_MIN_M,
                _HEIGHT_MAX_M,
            ),
            np.clip(
                channels["aligned_range_mean_m"] / _MAX_RANGE_M,
                0.0,
                1.0,
            ),
            _normalize_supported(
                channels["rcs_p90_db"],
                occupancy,
                _RADAR_RCS_MIN_DB,
                _RADAR_RCS_MAX_DB,
            ),
            _normalize_supported(
                channels["compensated_radial_velocity_mean_mps"],
                occupancy,
                -_RADAR_VELOCITY_LIMIT_MPS,
                _RADAR_VELOCITY_LIMIT_MPS,
            ),
        ),
        axis=0,
    ).astype(np.float32, copy=False)


RADAR_CHANNEL_NOTES = {
    "rcs": "VoD supplies RCS in dB, not a separate raw-power field.",
    "scan_time_index": (
        "The accumulated release supplies scan indices -4..0, not time in seconds."
    ),
    "radial_projection_vx_vy": (
        "Derived by projecting compensated radial velocity onto the aligned XY "
        "line of sight; these are not full Cartesian object velocities."
    ),
}

LIDAR_UNAVAILABLE_CHANNELS = {
    "relative_timestamp": (
        "VoD LiDAR .bin rows contain only x, y, z, and reflectivity; no per-point "
        "relative timestamp is available."
    )
}
