from __future__ import annotations

import numpy as np


def compensate_doppler(
    points: np.ndarray,
    sensor_velocity: np.ndarray,
    doppler_sign: str = "auto",
    sign_inference_min_speed_mps: float = 0.5,
    doppler_period_mps: float | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Subtract expected ego Doppler, including K-Radar velocity wrapping."""

    points = np.asarray(points, dtype=np.float64)
    sensor_velocity = np.asarray(sensor_velocity, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"points must have shape [N,>=4], got {points.shape}")
    if sensor_velocity.shape != (3,) or not np.isfinite(sensor_velocity).all():
        raise ValueError("sensor_velocity must be a finite 3-vector")
    if doppler_sign not in {"auto", "1", "-1"}:
        raise ValueError("doppler_sign must be one of: auto, 1, -1")
    if doppler_period_mps is not None and doppler_period_mps <= 0.0:
        raise ValueError("doppler_period_mps must be positive when provided")
    ranges = np.linalg.norm(points[:, :3], axis=1)
    valid = ranges > 1e-6
    line_of_sight = np.zeros((len(points), 3), dtype=np.float64)
    line_of_sight[valid] = points[valid, :3] / ranges[valid, None]
    expected_static = -(line_of_sight @ sensor_velocity)
    measured = points[:, 3]

    def residual_for_sign(sign: int) -> np.ndarray:
        residual = sign * measured - expected_static
        if doppler_period_mps is not None:
            half_period = 0.5 * doppler_period_mps
            residual = (residual + half_period) % doppler_period_mps - half_period
        return residual

    if doppler_sign == "auto":
        if np.linalg.norm(sensor_velocity) < sign_inference_min_speed_mps or not np.any(valid):
            sign = 1
        else:
            positive_score = float(np.median(np.abs(residual_for_sign(1)[valid])))
            negative_score = float(np.median(np.abs(residual_for_sign(-1)[valid])))
            sign = 1 if positive_score <= negative_score else -1
    else:
        sign = int(doppler_sign)
    residual = residual_for_sign(sign)
    residual[~valid] = 0.0
    return residual.astype(np.float32), sign, expected_static.astype(np.float32)
