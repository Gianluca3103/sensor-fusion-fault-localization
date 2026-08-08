from pathlib import Path
from typing import Dict, Tuple

import numpy as np


LIDAR_CHANNELS = (
    "occupancy",
    "log_density",
    "robust_upper_height",
)
UPPER_HEIGHT_QUANTILE = 0.90
HEIGHT_RANGE_M = (-3.0, 5.0)


def _validate_geometry(x_range, y_range, resolution):
    if len(x_range) != 2 or len(y_range) != 2:
        raise ValueError("x_range and y_range must each contain exactly two values")
    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    resolution = float(resolution)
    if not np.isfinite([x_min, x_max, y_min, y_max, resolution]).all():
        raise ValueError("BEV ranges and resolution must be finite")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("BEV range maxima must be greater than their minima")
    if resolution <= 0.0:
        raise ValueError("BEV resolution must be positive")
    return x_min, x_max, y_min, y_max, resolution


def metric_to_grid(
    xyz: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: float,
):
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must have shape [N,>=3], got {xyz.shape}")
    if not np.isfinite(xyz[:, :3]).all():
        raise ValueError("xyz contains non-finite coordinates")

    x_min, x_max, y_min, y_max, resolution = _validate_geometry(
        x_range, y_range, resolution
    )
    height = int(np.ceil((x_max - x_min) / resolution))
    width = int(np.ceil((y_max - y_min) / resolution))

    valid = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] < x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] < y_max)
    )
    xyz_valid = xyz[valid]
    cols = np.floor((xyz_valid[:, 1] - y_min) / resolution).astype(np.int32)
    rows_from_bottom = np.floor((xyz_valid[:, 0] - x_min) / resolution).astype(np.int32)
    # Float32 arithmetic can round a coordinate just below an exclusive range
    # maximum onto width/height exactly (for example, column 320 in a
    # 320-column grid). The metric validity check above already established
    # that these points belong to the BEV, so keep them in the final edge cell.
    cols = np.clip(cols, 0, width - 1)
    rows_from_bottom = np.clip(rows_from_bottom, 0, height - 1)
    rows = height - 1 - rows_from_bottom
    return xyz_valid, rows, cols, valid, height, width


def normalize_by_max(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.size == 0:
        return np.zeros_like(grid, dtype=np.float32)
    if not np.isfinite(grid).all():
        raise ValueError("Cannot normalize a grid containing non-finite values")
    max_value = float(np.max(grid))
    if max_value <= 0.0:
        return np.zeros_like(grid, dtype=np.float32)
    return (grid / max_value).astype(np.float32)


def normalize_occupied(grid: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    occupied = np.asarray(occupied, dtype=bool)
    if grid.shape != occupied.shape:
        raise ValueError(
            f"grid and occupied must have the same shape, got {grid.shape} and {occupied.shape}"
        )
    output = np.zeros_like(grid, dtype=np.float32)
    if not np.any(occupied):
        return output

    values = grid[occupied]
    if not np.isfinite(values).all():
        raise ValueError("Cannot normalize occupied cells containing non-finite values")
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value == min_value:
        output[occupied] = 1.0
    else:
        output[occupied] = (values - min_value) / (max_value - min_value)
    return output


def robust_upper_height_grid(
    flat_cells: np.ndarray,
    heights_m: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return the radar-matched per-cell upper-height representation."""

    output = np.zeros(shape[0] * shape[1], dtype=np.float32)
    if flat_cells.size == 0:
        return output.reshape(shape)

    z_min, z_max = HEIGHT_RANGE_M
    plausible = (heights_m >= z_min) & (heights_m <= z_max)
    flat_cells = flat_cells[plausible]
    heights_m = heights_m[plausible]
    if flat_cells.size == 0:
        return output.reshape(shape)

    order = np.lexsort((heights_m, flat_cells))
    sorted_cells = flat_cells[order]
    sorted_heights = heights_m[order]
    starts = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1]])
    ends = np.r_[starts[1:], len(sorted_cells)]
    counts = ends - starts
    offsets = np.ceil(UPPER_HEIGHT_QUANTILE * counts).astype(np.int64) - 1
    upper_heights = sorted_heights[starts + offsets]

    normalized = (upper_heights - z_min) / (z_max - z_min)
    output[sorted_cells[starts]] = normalized.astype(np.float32)
    return output.reshape(shape)


def project_lidar_bev(
    points: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: float,
) -> Dict[str, np.ndarray]:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape [N,>=3], got {points.shape}")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError("points contains non-finite XYZ values")
    x_min, x_max, y_min, y_max, resolution = _validate_geometry(
        x_range, y_range, resolution
    )

    if points.size == 0:
        height = int(np.ceil((x_max - x_min) / resolution))
        width = int(np.ceil((y_max - y_min) / resolution))
        zeros = np.zeros((height, width), dtype=np.float32)
        return {
            "occupancy": zeros.copy(),
            "density": zeros.copy(),
            "robust_upper_height": zeros.copy(),
            "raw_density": zeros.copy(),
        }

    xyz, rows, cols, valid, height, width = metric_to_grid(
        points[:, :3],
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        resolution=resolution,
    )
    density = np.zeros((height, width), dtype=np.float32)

    if len(xyz) > 0:
        np.add.at(density, (rows, cols), 1.0)

    occupancy = (density > 0).astype(np.float32)
    density_normalized = normalize_by_max(np.log1p(density))
    flat_cells = rows.astype(np.int64) * width + cols
    robust_upper_height = robust_upper_height_grid(
        flat_cells,
        xyz[:, 2],
        (height, width),
    )

    return {
        "occupancy": occupancy,
        "density": density_normalized,
        "robust_upper_height": robust_upper_height,
        "raw_density": density,
    }


def make_rgb_preview(bev_layers: Dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.stack(
        [
            bev_layers["occupancy"],
            bev_layers["density"],
            bev_layers["robust_upper_height"],
        ],
        axis=-1,
    )
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def write_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(path)
