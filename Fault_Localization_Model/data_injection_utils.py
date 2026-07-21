from pathlib import Path
from typing import Dict
import copy
import importlib.util
import json
import sys
import types

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HERCULES_ROOT = REPO_ROOT / "data" / "HerculesFiles" / "Data"
DEFAULT_INJECTOR_ROOT = REPO_ROOT / "Weather_Injector" / "3D_Corruptions_AD"
DEFAULT_FOG_ROOT = REPO_ROOT / "Weather_Injector" / "LiDAR_fog_sim"

AEVA_RECORD_BYTES = 29
FOG_ALPHA_BY_SEVERITY = [0.005, 0.01, 0.02, 0.03, 0.06]


def patch_compatibility_modules() -> None:
    for name in ["open3d", "h5py", "distortion"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    try:
        import multiprocessing as mp
        import multiprocessing.pool as mp_pool
        if not hasattr(mp, "pool"):
            mp.pool = mp_pool
    except ImportError:
        pass

    try:
        import scipy.integrate
    except ImportError:
        return

    if not hasattr(scipy.integrate, "trapz"):
        if hasattr(np, "trapezoid"):
            scipy.integrate.trapz = np.trapezoid
        elif hasattr(np, "trapz"):
            scipy.integrate.trapz = np.trapz
        else:
            def _trapz(y, x=None, dx=1.0, axis=-1):
                y = np.asarray(y)
                if x is None:
                    d = dx
                else:
                    d = np.diff(np.asarray(x), axis=axis)
                upper = np.take(y, range(1, y.shape[axis]), axis=axis)
                lower = np.take(y, range(0, y.shape[axis] - 1), axis=axis)
                return np.sum((upper + lower) * 0.5 * d, axis=axis)

            scipy.integrate.trapz = _trapz


def import_lidar_corruptions(injector_root: Path):
    patch_compatibility_modules()
    if str(injector_root) not in sys.path:
        sys.path.insert(0, str(injector_root))

    module_path = injector_root / "LiDAR_corruptions.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find LiDAR_corruptions.py at {module_path}")

    spec = importlib.util.spec_from_file_location("thu_lidar_corruptions", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_pointcloud(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected a point cloud with at least 4 columns, got {points.shape}")
    finite = np.isfinite(points[:, :4]).all(axis=1)
    return points[finite]


def apply_rain_wrapper(injector_root: Path, points: np.ndarray, severity: int) -> np.ndarray:
    patch_compatibility_modules()
    if str(injector_root) not in sys.path:
        sys.path.insert(0, str(injector_root))
    from utils import lisa

    rain_model = lisa.LISA(show_progressbar=False)
    rain_rate = [0.20, 0.73, 1.5625, 3.125, 7.29][severity - 1]
    return rain_model.augment(points, rain_rate)


def apply_snow_wrapper(injector_root: Path, points: np.ndarray, severity: int) -> np.ndarray:
    patch_compatibility_modules()
    if str(injector_root) not in sys.path:
        sys.path.insert(0, str(injector_root))
    from utils import lisa

    snow_model = lisa.LISA(mode="gunn", show_progressbar=False)
    snowfall_rate = [0.20, 0.73, 1.5625, 3.125, 7.29][severity - 1]
    return snow_model.augment(points, snowfall_rate)


def apply_fault(module, injector_root: Path, fault: str, points: np.ndarray, severity: int) -> np.ndarray:
    if fault == "rain_sim":
        return safe_pointcloud(apply_rain_wrapper(injector_root, points.copy(), severity))
    if fault == "snow_sim":
        return safe_pointcloud(apply_snow_wrapper(injector_root, points.copy(), severity))

    func = getattr(module, fault)
    np.random.seed(1000 + severity)
    return safe_pointcloud(func(points.copy(), severity))


def filter_pointcloud(
    points: np.ndarray,
    min_range: float,
    max_range: float,
    return_mask: bool = False,
):
    """Remove invalid/out-of-range returns and optionally expose the row mask."""
    original_length = len(points)
    points = safe_pointcloud(points)
    if len(points) != original_length and return_mask:
        raise ValueError(
            "Cannot return a source-aligned range mask after safe_pointcloud removed non-finite rows. "
            "Validate finite rows before provenance-aware filtering."
        )
    xyz = points[:, :3]
    distances = np.linalg.norm(xyz, axis=1)
    valid = distances >= min_range
    valid &= distances <= max_range
    valid &= ~np.all(np.isclose(xyz, 0.0), axis=1)
    filtered = points[valid]
    return (filtered, valid) if return_mask else filtered


def lisa_label_counts(points: np.ndarray) -> Dict[str, int]:
    counts = {
        "lisa_label0_lost_points": 0,
        "lisa_label1_non_scattered_points": 0,
        "lisa_label2_scattered_points": 0,
    }
    if points.shape[1] < 5:
        return counts
    labels = points[:, 4].astype(np.int32)
    counts["lisa_label0_lost_points"] = int(np.sum(labels == 0))
    counts["lisa_label1_non_scattered_points"] = int(np.sum(labels == 1))
    counts["lisa_label2_scattered_points"] = int(np.sum(labels == 2))
    return counts


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not np.any(mask):
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                y0 = radius + dy
                x0 = radius + dx
                output |= padded[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    return output


def find_aeva_dir(data_root: Path, day: str, session: str) -> Path:
    candidates = [
        data_root / day / session / "LiDAR" / "LiDAR" / "Aeva",
        data_root / day / "LiDAR" / "LiDAR" / "Aeva",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Hercules Aeva LiDAR folder. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def list_aeva_bins(aeva_dir: Path):
    bins = sorted(aeva_dir.glob("*.bin"), key=lambda path: path.stem)
    if not bins:
        raise FileNotFoundError(f"No Hercules Aeva .bin files found in {aeva_dir}")
    return bins


def read_hercules_aeva_bin(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n_points = len(raw) // AEVA_RECORD_BYTES
    if n_points == 0:
        return np.empty((0, 4), dtype=np.float32)

    usable = raw[: n_points * AEVA_RECORD_BYTES]
    records = np.frombuffer(usable, dtype=np.uint8).reshape(n_points, AEVA_RECORD_BYTES)
    xyzi = records[:, :16].copy().view("<f4").reshape(n_points, 4)
    finite = np.isfinite(xyzi).all(axis=1)
    xyzi = xyzi[finite]
    plausible = (
        (np.abs(xyzi[:, 0]) < 250.0)
        & (np.abs(xyzi[:, 1]) < 250.0)
        & (np.abs(xyzi[:, 2]) < 80.0)
    )
    return xyzi[plausible].astype(np.float32, copy=False)


def apply_fog_simulator(fog_root: Path, points: np.ndarray, severity: int, noise: int):
    if str(fog_root) not in sys.path:
        sys.path.insert(0, str(fog_root))

    from fog_simulation import ParameterSet, P_R_fog_hard, P_R_fog_soft

    alpha = FOG_ALPHA_BY_SEVERITY[severity - 1]
    parameter_set = ParameterSet(alpha=alpha, gamma=0.000001)
    original_intensity = copy.deepcopy(points[:, 3])
    hard_pc = P_R_fog_hard(parameter_set, copy.deepcopy(points))
    augmented_pc, simulated_fog_pc, info = P_R_fog_soft(
        parameter_set,
        copy.deepcopy(hard_pc),
        original_intensity,
        noise=noise,
        gain=False,
        noise_variant="v1",
    )
    soft_mask = np.linalg.norm(augmented_pc[:, :3] - hard_pc[:, :3], axis=1) > 1e-4
    labels = np.ones((augmented_pc.shape[0], 1), dtype=np.float32)
    labels[soft_mask, 0] = 2.0
    return np.hstack([augmented_pc[:, :4], labels]), {
        "fog_alpha": alpha,
        "fog_soft_response_points": int(np.sum(soft_mask)),
        "fog_info_json": json.dumps(info or {}, sort_keys=True),
    }
