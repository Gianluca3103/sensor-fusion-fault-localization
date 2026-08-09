"""Deterministic clean-LiDAR observability from exact XY grid traversal.

Physical interpretation:

1. A LiDAR return is direct occupied evidence.
2. The ray before that return is evidence of visible/free 3D space.
3. Space behind the return remains unknown because it is occluded.
4. A pillar without a return is not automatically free.
5. A ray crossing one height does not observe the pillar's full height.
6. Confidence measures support for treating a clean-empty BEV cell as
   observed free space; it does not replace the clean occupancy target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from numba import njit
except ImportError:  # Correct reference implementation remains available.
    njit = None

from Fault_Localization_Model.bev_utils import HEIGHT_RANGE_M, metric_to_grid


LIDAR_SENSOR_ORIGIN = (0.0, 0.0, 0.0)
DEFAULT_NUM_Z_BINS = 16
DEFAULT_RAY_SUPPORT_TAU = 4.0


if njit is not None:

    @njit(cache=True, nogil=True)
    def _compute_ray_observations_compiled(
        points,
        origin,
        x_min,
        x_max,
        y_min,
        y_max,
        resolution,
        height,
        width,
        z_min,
        z_max,
        num_z_bins,
    ):
        """Compiled equivalent of the reference Amanatides-Woo loop."""

        ray_count = np.zeros((height, width), dtype=np.uint32)
        observed = np.zeros((height, width, num_z_bins), dtype=np.bool_)
        hit_mask = np.zeros((height, width), dtype=np.bool_)
        ox, oy, oz = origin[0], origin[1], origin[2]
        bin_size = (z_max - z_min) / num_z_bins

        for point_index in range(points.shape[0]):
            px = points[point_index, 0]
            py = points[point_index, 1]
            pz = points[point_index, 2]
            dx, dy, dz = px - ox, py - oy, pz - oz

            hit_inside = x_min <= px < x_max and y_min <= py < y_max
            if hit_inside:
                hit_y = int(np.floor((py - y_min) / resolution))
                hit_x = int(np.floor((px - x_min) / resolution))
                hit_y = min(max(hit_y, 0), width - 1)
                hit_x = min(max(hit_x, 0), height - 1)
                hit_mask[height - 1 - hit_x, hit_y] = True
            else:
                hit_x = -1
                hit_y = -1

            if dx == 0.0 and dy == 0.0:
                continue

            t_enter = 0.0
            t_final = 1.0
            valid_segment = True
            if dx == 0.0:
                if ox < x_min or ox >= x_max:
                    valid_segment = False
            else:
                first = (x_min - ox) / dx
                second = (x_max - ox) / dx
                near = min(first, second)
                far = max(first, second)
                t_enter = max(t_enter, near)
                t_final = min(t_final, far)
            if dy == 0.0:
                if oy < y_min or oy >= y_max:
                    valid_segment = False
            else:
                first = (y_min - oy) / dy
                second = (y_max - oy) / dy
                near = min(first, second)
                far = max(first, second)
                t_enter = max(t_enter, near)
                t_final = min(t_final, far)
            if not valid_segment or t_final <= t_enter:
                continue
            t_enter = max(0.0, t_enter)
            t_final = min(1.0, t_final)
            if t_final <= t_enter:
                continue

            start_x = ox + t_enter * dx
            start_y = oy + t_enter * dy
            scaled_x = (start_x - x_min) / resolution
            scaled_y = (start_y - y_min) / resolution
            rounded_x = round(scaled_x)
            rounded_y = round(scaled_y)
            if dx < 0.0 and abs(scaled_x - rounded_x) <= 1.0e-10:
                x_index = int(rounded_x) - 1
            else:
                x_index = int(np.floor(scaled_x))
            if dy < 0.0 and abs(scaled_y - rounded_y) <= 1.0e-10:
                y_index = int(rounded_y) - 1
            else:
                y_index = int(np.floor(scaled_y))
            x_index = min(max(x_index, 0), height - 1)
            y_index = min(max(y_index, 0), width - 1)

            if dx > 0.0:
                step_x = 1
            elif dx < 0.0:
                step_x = -1
            else:
                step_x = 0
            if dy > 0.0:
                step_y = 1
            elif dy < 0.0:
                step_y = -1
            else:
                step_y = 0

            if step_x != 0:
                boundary_x = x_min + (
                    x_index + (1 if step_x > 0 else 0)
                ) * resolution
                t_max_x = (boundary_x - ox) / dx
                t_delta_x = resolution / abs(dx)
            else:
                t_max_x = np.inf
                t_delta_x = np.inf
            if step_y != 0:
                boundary_y = y_min + (
                    y_index + (1 if step_y > 0 else 0)
                ) * resolution
                t_max_y = (boundary_y - oy) / dy
                t_delta_y = resolution / abs(dy)
            else:
                t_max_y = np.inf
                t_delta_y = np.inf

            t_current = t_enter
            for _step in range(height + width + 2):
                if not (0 <= x_index < height and 0 <= y_index < width):
                    break
                if hit_inside and x_index == hit_x and y_index == hit_y:
                    break
                cell_exit = min(t_max_x, t_max_y, t_final)
                if cell_exit > t_current:
                    row = height - 1 - x_index
                    ray_count[row, y_index] += 1
                    z_start = oz + t_current * dz
                    z_end = oz + cell_exit * dz
                    lower = max(min(z_start, z_end), z_min)
                    upper = min(max(z_start, z_end), z_max)
                    if upper >= lower and lower < z_max and upper >= z_min:
                        first_bin = int(np.floor((lower - z_min) / bin_size))
                        if upper > lower:
                            upper_index = np.nextafter(upper, -np.inf)
                        else:
                            upper_index = upper
                        last_bin = int(
                            np.floor((upper_index - z_min) / bin_size)
                        )
                        first_bin = min(max(first_bin, 0), num_z_bins - 1)
                        last_bin = min(max(last_bin, 0), num_z_bins - 1)
                        if last_bin >= first_bin:
                            for z_index in range(first_bin, last_bin + 1):
                                observed[row, y_index, z_index] = True
                if cell_exit >= t_final:
                    break
                tolerance = 1.0e-12 * max(1.0, abs(cell_exit))
                cross_x = t_max_x <= cell_exit + tolerance
                cross_y = t_max_y <= cell_exit + tolerance
                if cross_x:
                    x_index += step_x
                    t_max_x += t_delta_x
                if cross_y:
                    y_index += step_y
                    t_max_y += t_delta_y
                if not cross_x and not cross_y:
                    break
                t_current = cell_exit

        return ray_count, observed, hit_mask

else:
    _compute_ray_observations_compiled = None


def _validated_geometry(x_range, y_range, resolution):
    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    resolution = float(resolution)
    values = np.asarray((x_min, x_max, y_min, y_max, resolution))
    if not np.isfinite(values).all():
        raise ValueError("BEV geometry must contain finite values")
    if x_max <= x_min or y_max <= y_min or resolution <= 0.0:
        raise ValueError("BEV ranges must increase and resolution must be positive")
    height = int(np.ceil((x_max - x_min) / resolution))
    width = int(np.ceil((y_max - y_min) / resolution))
    return x_min, x_max, y_min, y_max, resolution, height, width


def _clip_segment_to_xy_bounds(origin, endpoint, bounds):
    """Return the parametric interval where a segment lies in the BEV."""

    x_min, x_max, y_min, y_max = bounds
    t_enter = 0.0
    t_exit = 1.0
    for start, delta, lower, upper in (
        (origin[0], endpoint[0] - origin[0], x_min, x_max),
        (origin[1], endpoint[1] - origin[1], y_min, y_max),
    ):
        if delta == 0.0:
            if start < lower or start >= upper:
                return None
            continue
        first = (lower - start) / delta
        second = (upper - start) / delta
        near, far = sorted((first, second))
        t_enter = max(t_enter, near)
        t_exit = min(t_exit, far)
        if t_exit <= t_enter:
            return None
    return max(0.0, t_enter), min(1.0, t_exit)


def _cell_index(coordinate, lower, resolution, direction, count):
    scaled = (coordinate - lower) / resolution
    nearest = round(scaled)
    if direction < 0.0 and np.isclose(scaled, nearest, rtol=0.0, atol=1.0e-10):
        index = int(nearest) - 1
    else:
        index = int(np.floor(scaled))
    return min(max(index, 0), count - 1)


def _traversed_xy_cells(origin, endpoint, x_range, y_range, resolution):
    """Yield ``(x_index, y_index, t_enter, t_exit)`` before the hit.

    This is a two-dimensional Amanatides-Woo traversal. Cells touched only at
    a corner are not double-counted. If the return is inside the BEV, its final
    hit cell is deliberately excluded from free-space traversal.
    """

    x_min, x_max, y_min, y_max, resolution, height, width = _validated_geometry(
        x_range, y_range, resolution
    )
    origin = np.asarray(origin, dtype=np.float64)
    endpoint = np.asarray(endpoint, dtype=np.float64)
    delta = endpoint - origin
    clipped = _clip_segment_to_xy_bounds(
        origin, endpoint, (x_min, x_max, y_min, y_max)
    )
    if clipped is None or np.all(delta[:2] == 0.0):
        return
    t_current, t_final = clipped
    if t_final <= t_current:
        return

    start_x = origin[0] + t_current * delta[0]
    start_y = origin[1] + t_current * delta[1]
    x_index = _cell_index(start_x, x_min, resolution, delta[0], height)
    y_index = _cell_index(start_y, y_min, resolution, delta[1], width)

    hit_inside = (
        x_min <= endpoint[0] < x_max and y_min <= endpoint[1] < y_max
    )
    if hit_inside:
        _, hit_rows, hit_cols, _valid, _height, _width = metric_to_grid(
            endpoint[None, :3], x_range, y_range, resolution
        )
        hit_x_index = height - 1 - int(hit_rows[0])
        hit_y_index = int(hit_cols[0])
    else:
        hit_x_index = hit_y_index = -1

    step_x = 1 if delta[0] > 0.0 else -1 if delta[0] < 0.0 else 0
    step_y = 1 if delta[1] > 0.0 else -1 if delta[1] < 0.0 else 0
    if step_x:
        boundary_x = x_min + (x_index + (step_x > 0)) * resolution
        t_max_x = (boundary_x - origin[0]) / delta[0]
        t_delta_x = resolution / abs(delta[0])
    else:
        t_max_x = t_delta_x = np.inf
    if step_y:
        boundary_y = y_min + (y_index + (step_y > 0)) * resolution
        t_max_y = (boundary_y - origin[1]) / delta[1]
        t_delta_y = resolution / abs(delta[1])
    else:
        t_max_y = t_delta_y = np.inf

    maximum_steps = height + width + 2
    for _ in range(maximum_steps):
        if not (0 <= x_index < height and 0 <= y_index < width):
            break
        if hit_inside and x_index == hit_x_index and y_index == hit_y_index:
            break
        cell_exit = min(t_max_x, t_max_y, t_final)
        if cell_exit > t_current:
            yield x_index, y_index, t_current, cell_exit
        if cell_exit >= t_final:
            break

        tolerance = 1.0e-12 * max(1.0, abs(cell_exit))
        cross_x = t_max_x <= cell_exit + tolerance
        cross_y = t_max_y <= cell_exit + tolerance
        if cross_x:
            x_index += step_x
            t_max_x += t_delta_x
        if cross_y:
            y_index += step_y
            t_max_y += t_delta_y
        if not cross_x and not cross_y:
            raise RuntimeError("LiDAR grid traversal made no progress")
        t_current = cell_exit
    else:
        raise RuntimeError("LiDAR grid traversal exceeded its finite step bound")


def _mark_vertical_interval(observed_bins, z_start, z_end, z_range):
    z_min, z_max = z_range
    lower = max(min(z_start, z_end), z_min)
    upper = min(max(z_start, z_end), z_max)
    if upper < z_min or lower >= z_max or upper < lower:
        return
    bin_size = (z_max - z_min) / len(observed_bins)
    first = int(np.floor((lower - z_min) / bin_size))
    if upper > lower:
        upper_for_index = np.nextafter(upper, -np.inf)
    else:
        upper_for_index = upper
    last = int(np.floor((upper_for_index - z_min) / bin_size))
    first = min(max(first, 0), len(observed_bins) - 1)
    last = min(max(last, 0), len(observed_bins) - 1)
    if last >= first:
        observed_bins[first : last + 1] = True


def _pillar_range_map(shape, x_range, y_range, resolution, sensor_origin):
    x_min, x_max = (float(value) for value in x_range)
    y_min, y_max = (float(value) for value in y_range)
    height, width = shape
    x_lower = x_min + np.arange(height, dtype=np.float64) * resolution
    y_lower = y_min + np.arange(width, dtype=np.float64) * resolution
    x_centers_increasing = (x_lower + np.minimum(x_lower + resolution, x_max)) / 2.0
    y_centers = (y_lower + np.minimum(y_lower + resolution, y_max)) / 2.0
    x_centers = x_centers_increasing[::-1]
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    return np.hypot(
        grid_x - float(sensor_origin[0]),
        grid_y - float(sensor_origin[1]),
    ).astype(np.float32)


def compute_ray_observations(
    clean_points: np.ndarray,
    sensor_origin: Sequence[float],
    x_range,
    y_range,
    resolution: float,
    *,
    z_range=HEIGHT_RANGE_M,
    num_z_bins: int = DEFAULT_NUM_Z_BINS,
    use_compiled: bool = True,
) -> dict[str, np.ndarray]:
    """Trace clean return segments and return raw geometric observations."""

    points = np.asarray(clean_points)
    origin = np.asarray(sensor_origin, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"clean_points must have shape [N,>=3], got {points.shape}")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("sensor_origin must contain three finite coordinates")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError("clean_points contains non-finite XYZ coordinates")
    if int(num_z_bins) != num_z_bins or num_z_bins < 1:
        raise ValueError("num_z_bins must be a positive integer")
    z_min, z_max = (float(value) for value in z_range)
    if not np.isfinite((z_min, z_max)).all() or z_max <= z_min:
        raise ValueError("z_range must contain increasing finite values")
    x_min, x_max, y_min, y_max, resolution, height, width = _validated_geometry(
        x_range, y_range, resolution
    )

    if use_compiled and _compute_ray_observations_compiled is not None:
        ray_count, observed_z_bins, hit_mask = (
            _compute_ray_observations_compiled(
                np.ascontiguousarray(points[:, :3], dtype=np.float64),
                origin,
                x_min,
                x_max,
                y_min,
                y_max,
                resolution,
                height,
                width,
                z_min,
                z_max,
                int(num_z_bins),
            )
        )
    else:
        ray_count = np.zeros((height, width), dtype=np.uint32)
        observed_z_bins = np.zeros((height, width, num_z_bins), dtype=bool)
        hit_mask = np.zeros((height, width), dtype=bool)
        if len(points):
            _xyz, rows, cols, _valid, _height, _width = metric_to_grid(
                points[:, :3], (x_min, x_max), (y_min, y_max), resolution
            )
            hit_mask[rows, cols] = True
            for endpoint in points[:, :3].astype(np.float64, copy=False):
                delta_z = endpoint[2] - origin[2]
                for x_index, y_index, t_enter, t_exit in _traversed_xy_cells(
                    origin,
                    endpoint,
                    (x_min, x_max),
                    (y_min, y_max),
                    resolution,
                ):
                    row = height - 1 - x_index
                    ray_count[row, y_index] += 1
                    _mark_vertical_interval(
                        observed_z_bins[row, y_index],
                        origin[2] + t_enter * delta_z,
                        origin[2] + t_exit * delta_z,
                        (z_min, z_max),
                    )

    vertical_coverage = observed_z_bins.mean(axis=2, dtype=np.float32)
    range_map = _pillar_range_map(
        (height, width),
        (x_min, x_max),
        (y_min, y_max),
        resolution,
        origin,
    )
    return {
        "ray_count": ray_count,
        "observed_z_bins": observed_z_bins,
        "vertical_coverage": vertical_coverage,
        "range_map": range_map,
        "hit_mask": hit_mask,
    }


def warm_observability_backend() -> bool:
    """Compile/load the exact DDA kernel once before worker processes spawn."""

    if _compute_ray_observations_compiled is None:
        return False
    _compute_ray_observations_compiled(
        np.empty((0, 3), dtype=np.float64),
        np.asarray(LIDAR_SENSOR_ORIGIN, dtype=np.float64),
        0.0,
        1.0,
        -0.5,
        0.5,
        1.0,
        1,
        1,
        -3.0,
        5.0,
        DEFAULT_NUM_Z_BINS,
    )
    return True


def compute_observability_confidence(
    ray_count: np.ndarray,
    vertical_coverage: np.ndarray,
    *,
    ray_support_tau: float = DEFAULT_RAY_SUPPORT_TAU,
) -> dict[str, np.ndarray]:
    """Combine raw observations using the version-one confidence formula."""

    ray_count = np.asarray(ray_count)
    vertical_coverage = np.asarray(vertical_coverage, dtype=np.float32)
    if ray_count.shape != vertical_coverage.shape:
        raise ValueError("ray_count and vertical_coverage shapes must match")
    if not np.isfinite(vertical_coverage).all():
        raise ValueError("vertical_coverage must be finite")
    if not np.isfinite(ray_support_tau) or ray_support_tau <= 0.0:
        raise ValueError("ray_support_tau must be positive and finite")
    ray_support = 1.0 - np.exp(
        -ray_count.astype(np.float32) / float(ray_support_tau)
    )
    confidence = np.clip(vertical_coverage * ray_support, 0.0, 1.0)
    return {
        "ray_support": ray_support.astype(np.float32),
        "observability_confidence": confidence.astype(np.float32),
    }


def create_observability_map(
    clean_points: np.ndarray,
    sensor_origin: Sequence[float],
    x_range,
    y_range,
    resolution: float,
    *,
    z_range=HEIGHT_RANGE_M,
    num_z_bins: int = DEFAULT_NUM_Z_BINS,
    ray_support_tau: float = DEFAULT_RAY_SUPPORT_TAU,
) -> dict[str, np.ndarray]:
    observations = compute_ray_observations(
        clean_points,
        sensor_origin,
        x_range,
        y_range,
        resolution,
        z_range=z_range,
        num_z_bins=num_z_bins,
    )
    observations.update(
        compute_observability_confidence(
            observations["ray_count"],
            observations["vertical_coverage"],
            ray_support_tau=ray_support_tau,
        )
    )
    return observations


def save_observability_debug_figure(
    output_path: str | Path,
    clean_occupancy: np.ndarray,
    observations: dict[str, np.ndarray],
) -> None:
    """Save the five requested alignment/debug panels for one clean frame."""

    import matplotlib.pyplot as plt

    panels = (
        (clean_occupancy, "Clean occupancy", "gray"),
        (observations["ray_count"], "Ray count", "magma"),
        (observations["vertical_coverage"], "Vertical coverage", "viridis"),
        (observations["ray_support"], "Ray support", "plasma"),
        (observations["observability_confidence"], "Observability confidence", "turbo"),
    )
    figure, axes = plt.subplots(1, 5, figsize=(22, 5))
    for axis, (values, title, cmap) in zip(axes, panels):
        image = axis.imshow(values, cmap=cmap, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
