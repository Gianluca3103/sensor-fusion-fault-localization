"""Calibrated angular projection and exact original-point bookkeeping.

Beam centres and azimuth sampling are supplied by sensor configuration. This
module deliberately has no dataset-specific default beam count or elevation
range: the existing four-column point files contain no ring identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RangeGeometry:
    beam_elevations_rad: tuple[float, ...]
    azimuth_bins: int
    min_range_m: float
    max_range_m: float
    azimuth_span_rad: float
    azimuth_offset_rad: float = 0.0
    max_beam_error_rad: float | None = None

    def __post_init__(self) -> None:
        beams = np.asarray(self.beam_elevations_rad, dtype=np.float64)
        if len(beams) < 1 or not np.isfinite(beams).all() or np.any(np.diff(beams) <= 0):
            raise ValueError("beam_elevations_rad must be finite, nonempty and strictly increasing")
        if np.any(np.abs(beams) >= np.pi / 2):
            raise ValueError("beam elevations must lie strictly between -pi/2 and pi/2")
        if self.azimuth_bins < 2 or not (0 < self.min_range_m < self.max_range_m):
            raise ValueError("azimuth_bins and physical range bounds are invalid")
        if not 0 < self.azimuth_span_rad <= 2 * np.pi:
            raise ValueError("azimuth_span_rad must lie in (0,2pi]")
        if len(beams) == 1 and self.max_beam_error_rad is None:
            raise ValueError("single-beam geometry needs max_beam_error_rad")
        if not np.isfinite(self.azimuth_offset_rad):
            raise ValueError("azimuth_offset_rad must be finite")
        if self.max_beam_error_rad is not None and self.max_beam_error_rad <= 0:
            raise ValueError("max_beam_error_rad must be positive when set")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.beam_elevations_rad), self.azimuth_bins

    @classmethod
    def from_json(cls, path: str | Path) -> "RangeGeometry":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "beam_elevations_rad" not in payload:
            raise ValueError("Sensor geometry needs calibrated beam_elevations_rad; no beam table was found in point files")
        return cls(
            beam_elevations_rad=tuple(float(v) for v in payload["beam_elevations_rad"]),
            azimuth_bins=int(payload["azimuth_bins"]),
            min_range_m=float(payload["min_range_m"]),
            max_range_m=float(payload["max_range_m"]),
            azimuth_span_rad=float(payload["azimuth_span_rad"]),
            azimuth_offset_rad=float(payload.get("azimuth_offset_rad", 0.0)),
            max_beam_error_rad=(None if payload.get("max_beam_error_rad") is None
                                else float(payload["max_beam_error_rad"])),
        )

    def ray_directions(self) -> np.ndarray:
        azimuth = self.azimuth_offset_rad + (
            np.arange(self.azimuth_bins, dtype=np.float64) + 0.5
        ) * (self.azimuth_span_rad / self.azimuth_bins)
        elevation = np.asarray(self.beam_elevations_rad, dtype=np.float64)
        cos_elevation = np.cos(elevation)[:, None]
        return np.stack(np.broadcast_arrays(
            cos_elevation * np.cos(azimuth)[None, :],
            cos_elevation * np.sin(azimuth)[None, :],
            np.sin(elevation)[:, None],
        ), axis=-1).astype(np.float32)


@dataclass(frozen=True)
class RangeProjection:
    range_m: np.ndarray
    valid: np.ndarray
    reflectivity: np.ndarray
    nearest_original_index: np.ndarray
    collision_count: np.ndarray
    point_row: np.ndarray
    point_col: np.ndarray
    point_valid: np.ndarray


def angular_indices(points_xyz: np.ndarray, geometry: RangeGeometry, *,
                    require_beam_match: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(points_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("points_xyz must have shape [N, >=3]")
    radius = np.linalg.norm(xyz[:, :3], axis=1)
    valid = np.isfinite(xyz[:, :3]).all(axis=1) & (radius >= geometry.min_range_m) & (radius <= geometry.max_range_m)
    safe_radius = np.maximum(radius, 1e-12)
    elevation = np.arcsin(np.clip(xyz[:, 2] / safe_radius, -1, 1))
    beams = np.asarray(geometry.beam_elevations_rad)
    right = np.searchsorted(beams, elevation).clip(0, len(beams) - 1)
    left = (right - 1).clip(0, len(beams) - 1)
    row = np.where(np.abs(elevation - beams[left]) <= np.abs(elevation - beams[right]), left, right).astype(np.int32)
    if geometry.max_beam_error_rad is not None:
        beam_error = geometry.max_beam_error_rad
    elif len(beams) > 1:
        beam_error = float(np.max(np.diff(beams))) / 2
    else:
        beam_error = 0.0
    if require_beam_match and geometry.max_beam_error_rad is not None:
        valid &= np.abs(elevation - beams[row]) <= geometry.max_beam_error_rad
    elif require_beam_match:
        spacings = np.diff(beams)
        local_tolerance = np.empty(len(beams), dtype=np.float64)
        local_tolerance[0] = spacings[0] / 2
        local_tolerance[-1] = spacings[-1] / 2
        if len(beams) > 2:
            local_tolerance[1:-1] = np.maximum(spacings[:-1], spacings[1:]) / 2
        valid &= np.abs(elevation - beams[row]) <= local_tolerance[row]
    else:
        # Radar need not lie exactly on a LiDAR beam. Retain it within the
        # calibrated vertical field and aggregate into the nearest beam row.
        valid &= (elevation >= beams[0] - beam_error) & (elevation <= beams[-1] + beam_error)
    azimuth = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]) - geometry.azimuth_offset_rad, 2 * np.pi)
    if geometry.azimuth_span_rad < 2 * np.pi - 1e-8:
        valid &= azimuth < geometry.azimuth_span_rad
    col = np.floor(azimuth * geometry.azimuth_bins / geometry.azimuth_span_rad).astype(np.int64)
    col = np.clip(col, 0, geometry.azimuth_bins - 1).astype(np.int32)
    return row, col, radius.astype(np.float32), valid


def project_lidar(points: np.ndarray, geometry: RangeGeometry) -> RangeProjection:
    """Nearest visible return wins per ray; every original retains its own index."""
    points = np.asarray(points, dtype=np.float32)
    row, col, radius, valid = angular_indices(points, geometry)
    height, width = geometry.shape
    image_range = np.zeros((height, width), dtype=np.float32)
    image_valid = np.zeros((height, width), dtype=bool)
    reflectivity = np.zeros((height, width), dtype=np.float32)
    nearest = np.full((height, width), -1, dtype=np.int64)
    collision_count = np.zeros((height, width), dtype=np.int32)
    indices = np.flatnonzero(valid)
    if len(indices):
        flat = row[indices].astype(np.int64) * width + col[indices]
        collision_count[:] = np.bincount(flat, minlength=height * width).reshape(height, width)
        order = np.lexsort((radius[indices], flat))
        ordered_indices = indices[order]
        ordered_flat = flat[order]
        first = np.r_[True, ordered_flat[1:] != ordered_flat[:-1]]
        winners = ordered_indices[first]
        rr, cc = row[winners], col[winners]
        image_range[rr, cc] = radius[winners]
        image_valid[rr, cc] = True
        nearest[rr, cc] = winners
        if points.shape[1] > 3:
            reflectivity[rr, cc] = points[winners, 3]
    return RangeProjection(image_range, image_valid, reflectivity, nearest,
                           collision_count, row, col, valid)


def backproject(rows: np.ndarray, cols: np.ndarray, ranges_m: np.ndarray, geometry: RangeGeometry) -> np.ndarray:
    rows, cols, ranges = np.broadcast_arrays(rows, cols, ranges_m)
    if np.any((rows < 0) | (rows >= geometry.shape[0]) | (cols < 0) | (cols >= geometry.shape[1])):
        raise ValueError("ray indices outside geometry")
    if not np.isfinite(ranges).all() or np.any((ranges < geometry.min_range_m) | (ranges > geometry.max_range_m)):
        raise ValueError("predicted ranges outside physical sensor bounds")
    return (geometry.ray_directions()[rows, cols] * ranges[..., None]).astype(np.float32)


def transform_points(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    """Apply a calibrated rigid transform while retaining non-XYZ attributes.

    The model and merge remain in the LiDAR frame. Call this on the final
    ``MergeResult.points`` only when a vehicle/world-frame output is needed.
    """
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(target_from_source, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or transform.shape != (4, 4):
        raise ValueError("expected points [N,>=3] and calibrated 4x4 transform")
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("invalid homogeneous transform")
    output = points.copy()
    output[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    return output


def transform_radar_to_lidar(points: np.ndarray, lidar_from_radar: np.ndarray) -> np.ndarray:
    return transform_points(points, lidar_from_radar)
