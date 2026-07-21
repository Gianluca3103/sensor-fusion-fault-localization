import numpy as np


def apply_old_laser_degradation(points, severity, rng_seed=0, return_mask=False):
    """Apply range-dependent attenuation/dropout and optionally return kept rows."""
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
    }.get(int(severity), "mild")

    xyz = points[:, :3]
    intens = points[:, 3] if points.shape[1] > 3 else None
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

    if intens is not None:
        intens_att = alpha * intens
        threshold = np.quantile(intens, 0.10)
        mask &= intens_att >= threshold
    else:
        intens_att = None

    denom = max(r1 - r0, 1e-6)
    normalized_range = np.clip((ranges - r0) / denom, 0.0, 1.0)
    drop_probability = p_max * (normalized_range ** gamma)
    mask &= rng.random(len(xyz)) > drop_probability

    output = points[mask].copy()
    if intens_att is not None and output.shape[1] > 3:
        output[:, 3] = intens_att[mask]
    output = output.astype(np.float32)
    return (output, mask) if return_mask else output
