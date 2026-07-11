from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def metric_to_grid(
    xyz: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: float,
):
    x_min, x_max = x_range
    y_min, y_max = y_range
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
    rows = height - 1 - rows_from_bottom
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    return xyz_valid, rows, cols, valid, height, width


def normalize_by_max(grid: np.ndarray) -> np.ndarray:
    max_value = float(np.max(grid))
    if max_value <= 0.0:
        return np.zeros_like(grid, dtype=np.float32)
    return (grid / max_value).astype(np.float32)


def normalize_occupied(grid: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    output = np.zeros_like(grid, dtype=np.float32)
    if not np.any(occupied):
        return output

    values = grid[occupied]
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value == min_value:
        output[occupied] = 1.0
    else:
        output[occupied] = (values - min_value) / (max_value - min_value)
    return output


def project_lidar_bev(
    points: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    resolution: float,
) -> Dict[str, np.ndarray]:
    if points.size == 0:
        x_min, x_max = x_range
        y_min, y_max = y_range
        height = int(np.ceil((x_max - x_min) / resolution))
        width = int(np.ceil((y_max - y_min) / resolution))
        zeros = np.zeros((height, width), dtype=np.float32)
        return {
            "density": zeros.copy(),
            "height": zeros.copy(),
            "intensity": zeros.copy(),
            "height_spread": zeros.copy(),
            "raw_density": zeros.copy(),
        }

    xyz, rows, cols, valid, height, width = metric_to_grid(
        points[:, :3],
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )
    intensity = points[valid, 3]

    density = np.zeros((height, width), dtype=np.float32)
    max_height = np.full((height, width), -np.inf, dtype=np.float32)
    min_height = np.full((height, width), np.inf, dtype=np.float32)
    intensity_sum = np.zeros((height, width), dtype=np.float32)

    if len(xyz) > 0:
        np.add.at(density, (rows, cols), 1.0)
        np.maximum.at(max_height, (rows, cols), xyz[:, 2])
        np.minimum.at(min_height, (rows, cols), xyz[:, 2])
        np.add.at(intensity_sum, (rows, cols), intensity)

    occupied = density > 0
    mean_intensity = np.zeros_like(intensity_sum)
    np.divide(intensity_sum, density, out=mean_intensity, where=occupied)

    height_normalized = normalize_occupied(max_height, occupied)
    density_normalized = normalize_by_max(np.log1p(density))
    intensity_normalized = normalize_occupied(mean_intensity, occupied)

    height_spread = np.zeros((height, width), dtype=np.float32)
    height_spread[occupied] = max_height[occupied] - min_height[occupied]
    height_spread = normalize_by_max(height_spread)

    return {
        "density": density_normalized,
        "height": height_normalized,
        "intensity": intensity_normalized,
        "height_spread": height_spread,
        "raw_density": density,
    }


def make_rgb_preview(bev_layers: Dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.stack(
        [bev_layers["height"], bev_layers["intensity"], bev_layers["density"]],
        axis=-1,
    )
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def write_image(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(path)
