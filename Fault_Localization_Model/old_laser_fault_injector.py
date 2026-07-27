import numpy as np


def apply_old_laser_degradation(points, severity, rng_seed=0, return_mask=False):
    """Apply range-dependent attenuation/dropout and optionally return kept rows."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"points must have shape [N,>=4], got {points.shape}")
    if not np.isfinite(points[:, :4]).all():
        raise ValueError("points contains non-finite XYZ/intensity values")
    severity = int(severity)
    if not 0 <= severity <= 5:
        raise ValueError(f"severity must lie in [0,5], got {severity}")
    if not isinstance(rng_seed, (int, np.integer)) or int(rng_seed) < 0:
        raise ValueError(f"rng_seed must be non-negative, got {rng_seed}")
    rng_seed = int(rng_seed)
    if len(points) == 0:
        output = points.copy()
        mask = np.zeros(0, dtype=bool)
        return (output, mask) if return_mask else output

    severity_name = {
        0: "very_mild",
        1: "mild",
        2: "moderate",
        3: "severe",
        4: "extreme",
        5: "extreme",
    }[severity]

    xyz = points[:, :3]
    intensity = points[:, 3]
    rng = np.random.default_rng(rng_seed)
    ranges = np.linalg.norm(xyz, axis=1)

    if severity_name == "very_mild":
        alpha = 0.9
        p_max = 0.35
        gamma = 2.0
        q_cap = 0.93
    elif severity_name == "mild":
        alpha = 0.8
        p_max = 0.6
        gamma = 2.0
        q_cap = 0.85
    elif severity_name == "severe":
        alpha = 0.3
        p_max = 1.0
        gamma = 3.0
        q_cap = 0.50
    elif severity_name == "extreme":
        output = np.empty((0, points.shape[1]), dtype=np.float32)
        mask = np.zeros(len(points), dtype=bool)
        return (output, mask) if return_mask else output
    else:
        alpha = 0.6
        p_max = 0.8
        gamma = 2.5
        q_cap = 0.70

    r0 = np.quantile(ranges, 0.30)
    r1 = np.quantile(ranges, q_cap)
    mask = ranges <= r1

    attenuated_intensity = alpha * intensity
    threshold = np.quantile(intensity, 0.10)
    mask &= attenuated_intensity >= threshold

    denom = max(r1 - r0, 1e-6)
    normalized_range = np.clip((ranges - r0) / denom, 0.0, 1.0)
    drop_probability = p_max * (normalized_range ** gamma)
    mask &= rng.random(len(xyz)) > drop_probability

    output = points[mask].copy()
    output[:, 3] = attenuated_intensity[mask]
    output = output.astype(np.float32)
    return (output, mask) if return_mask else output
