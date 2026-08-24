"""Deterministic conversion of reconstructed BEV cells to repaired LiDAR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Fault_Localization_Model.bev_utils import HEIGHT_RANGE_M, metric_to_grid


@dataclass(frozen=True)
class ReconstructionPointCloudConfig:
    occupancy_threshold: float = 0.5
    reflectivity_policy: str = "nearest_preserved"

    def validate(self) -> None:
        if not 0.0 <= self.occupancy_threshold <= 1.0:
            raise ValueError("occupancy_threshold must be in [0, 1]")
        if self.reflectivity_policy not in {"nearest_preserved", "zero"}:
            raise ValueError(
                "reflectivity_policy must be nearest_preserved or zero"
            )


def _as_chw(bev: np.ndarray, geometry: BEVGridGeometry) -> np.ndarray:
    array = np.asarray(bev, dtype=np.float32)
    if array.shape == (geometry.height, geometry.width, 3):
        array = array.transpose(2, 0, 1)
    if array.shape != (3, geometry.height, geometry.width):
        raise ValueError(
            "reconstructed_bev must be [3,H,W] or [H,W,3], got "
            f"{array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("reconstructed_bev contains NaN or Inf")
    return array


def _as_mask(mask: np.ndarray, geometry: BEVGridGeometry) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape == (1, geometry.height, geometry.width):
        array = array[0]
    if array.shape != (geometry.height, geometry.width):
        raise ValueError(f"reconstruction_mask must be [H,W], got {array.shape}")
    return array > 0.5


def points_inside_bev_mask(
    points: np.ndarray,
    mask: np.ndarray,
    geometry,
) -> np.ndarray:
    """Return a boolean vector identifying points whose XY cell is masked."""

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {points.shape}")
    repair = _as_mask(mask, geometry)
    inside = np.zeros(len(points), dtype=bool)
    if not len(points):
        return inside
    _, rows, cols, valid, height, width = metric_to_grid(
        points[:, :3],
        (geometry.x_min, geometry.x_max),
        (geometry.y_min, geometry.y_max),
        geometry.pillar_size_x,
    )
    if (height, width) != repair.shape:
        raise ValueError("BEV geometry does not match reconstruction_mask")
    inside[np.flatnonzero(valid)] = repair[rows, cols]
    return inside


def _nearest_reflectivity(xy: np.ndarray, preserved: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(preserved) == 0 or preserved.shape[1] < 4:
        return np.zeros(len(xy), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(preserved[:, :2]).query(xy, k=1, workers=-1)
    except TypeError:  # Older SciPy without the workers argument.
        _, indices = cKDTree(preserved[:, :2]).query(xy, k=1)
    except ImportError:
        # Dependency-light deterministic fallback for export environments.
        indices = np.empty(len(xy), dtype=np.int64)
        for start in range(0, len(xy), 4096):
            chunk = xy[start : start + 4096]
            squared = np.sum(
                (chunk[:, None, :] - preserved[None, :, :2]) ** 2,
                axis=2,
            )
            indices[start : start + len(chunk)] = np.argmin(squared, axis=1)
    return preserved[np.asarray(indices), 3].astype(np.float32, copy=False)


def repair_point_cloud(
    faulty_points: np.ndarray,
    reconstructed_bev: np.ndarray,
    reconstruction_mask: np.ndarray,
    geometry,
    config: ReconstructionPointCloudConfig | None = None,
) -> np.ndarray:
    """Preserve faulty points outside repair and replace only masked cells.

    Each predicted occupied cell creates one point at its metric cell centre
    and decoded robust P90 height.  BEV has no reflectivity channel, so the
    default copies reflectivity from the nearest preserved measured point.
    This function is shared unchanged by coarse and fine evaluation.
    """

    geometry.validate()
    config = config or ReconstructionPointCloudConfig()
    config.validate()
    faulty = np.asarray(faulty_points, dtype=np.float32)
    if faulty.ndim != 2 or faulty.shape[1] != 4:
        raise ValueError(
            f"faulty_points must be [N,4] x/y/z/reflectivity, got {faulty.shape}"
        )
    if not np.isfinite(faulty).all():
        raise ValueError("faulty_points contains NaN or Inf")
    bev = _as_chw(reconstructed_bev, geometry)
    repair = _as_mask(reconstruction_mask, geometry)

    removed = points_inside_bev_mask(faulty, repair, geometry)
    preserved = faulty[~removed].copy()
    occupied = repair & (bev[0] >= config.occupancy_threshold)
    rows, cols = np.nonzero(occupied)
    if not len(rows):
        return preserved

    x = geometry.x_min + (geometry.height - rows - 0.5) * geometry.pillar_size_x
    y = geometry.y_min + (cols + 0.5) * geometry.pillar_size_y
    z_min, z_max = HEIGHT_RANGE_M
    z = z_min + np.clip(bev[2, rows, cols], 0.0, 1.0) * (z_max - z_min)
    xy = np.column_stack((x, y)).astype(np.float32)
    if config.reflectivity_policy == "nearest_preserved":
        reflectivity = _nearest_reflectivity(xy, preserved)
    else:
        reflectivity = np.zeros(len(xy), dtype=np.float32)
    generated = np.column_stack((x, y, z, reflectivity)).astype(np.float32)
    return np.concatenate((preserved, generated), axis=0)


def repair_point_cloud_with_clean_points(
    faulty_points: np.ndarray,
    clean_points: np.ndarray,
    reconstruction_mask: np.ndarray,
    geometry,
) -> np.ndarray:
    """Create an oracle repair using measured clean points inside the mask.

    Faulty measurements outside the reconstruction mask remain byte-for-byte
    unchanged.  Faulty measurements inside it are replaced by the matching
    clean LiDAR measurements.  This is a diagnostic upper bound, not a model
    input or a deployable reconstruction method.
    """

    geometry.validate()
    faulty = np.asarray(faulty_points, dtype=np.float32)
    clean = np.asarray(clean_points, dtype=np.float32)
    for name, points in (("faulty_points", faulty), ("clean_points", clean)):
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError(
                f"{name} must be [N,4] x/y/z/reflectivity, got {points.shape}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"{name} contains NaN or Inf")

    repair = _as_mask(reconstruction_mask, geometry)
    faulty_inside = points_inside_bev_mask(faulty, repair, geometry)
    clean_inside = points_inside_bev_mask(clean, repair, geometry)
    return np.concatenate(
        (faulty[~faulty_inside].copy(), clean[clean_inside].copy()),
        axis=0,
    )
