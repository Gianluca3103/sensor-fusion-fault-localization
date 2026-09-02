from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path
import importlib.util
import json
import random
import sys
import types
import numpy as np
from Fault_Localization_Model.config.defaults import DEFAULT_FOG_ROOT

_FOG_SIMULATION_PATH = DEFAULT_FOG_ROOT / "fog_simulation.py"
_FOG_SIMULATION_SPEC = importlib.util.spec_from_file_location(
    "weather_injector_fog_simulation",
    _FOG_SIMULATION_PATH,
)
if _FOG_SIMULATION_SPEC is None or _FOG_SIMULATION_SPEC.loader is None:
    raise ImportError(f"Could not load fog simulation module from {_FOG_SIMULATION_PATH}")
_FOG_SIMULATION_MODULE = importlib.util.module_from_spec(_FOG_SIMULATION_SPEC)
_FOG_SIMULATION_SPEC.loader.exec_module(_FOG_SIMULATION_MODULE)
ParameterSet = _FOG_SIMULATION_MODULE.ParameterSet
P_R_fog_hard = _FOG_SIMULATION_MODULE.P_R_fog_hard
P_R_fog_soft = _FOG_SIMULATION_MODULE.P_R_fog_soft

FOG_ALPHA_BY_SEVERITY = [0.005, 0.01, 0.02, 0.03, 0.06]
DEFAULT_FOG_SIMULATOR_NOISE = 10
DEFAULT_WEATHER_THREADS = 1
ROW_ALIGNED_CORRUPTIONS = {
    "scene_glare_noise",
    "lidar_crosstalk_noise",
    "gaussian_noise",
    "uniform_noise",
    "impulse_noise",
}
SPECIAL_CORRUPTIONS = {
    "fog_sim",
    "rain_sim",
    "snow_sim",
    "fov_filter",
    "old_laser_degradation",
    "laser_device_failure",
    "total_loss",
}
SUPPORTED_CORRUPTIONS = ROW_ALIGNED_CORRUPTIONS | SPECIAL_CORRUPTIONS


def validate_fault_spec(fault: str, severity: int) -> tuple[str, int]:
    fault = str(fault).strip()
    if fault not in SUPPORTED_CORRUPTIONS:
        raise ValueError(
            f"Unsupported provenance-aware fault {fault!r}. Supported faults: "
            + ", ".join(sorted(SUPPORTED_CORRUPTIONS))
        )
    severity = int(severity)
    minimum = 0 if fault in {"old_laser_degradation", "laser_device_failure"} else 1
    if not minimum <= severity <= 5:
        raise ValueError(
            f"Severity for {fault!r} must be between {minimum} and 5, got {severity}"
        )
    if fault == "total_loss" and severity != 1:
        raise ValueError("total_loss has one deterministic severity: 1")
    return fault, severity


@contextmanager
def temporary_random_seed(seed):
    """Seed legacy NumPy/Python RNG users without leaking state to later samples."""
    if seed is None:
        yield
        return
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def patch_compatibility_modules() -> None:
    for name in ["open3d", "h5py", "distortion"]:
        if name not in sys.modules and importlib.util.find_spec(name) is None:
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
    if str(injector_root) not in sys.path:
        sys.path.insert(0, str(injector_root))
    patch_compatibility_modules()

    module_path = injector_root / "LiDAR_corruptions.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find LiDAR_corruptions.py at {module_path}")

    spec = importlib.util.spec_from_file_location("thu_lidar_corruptions", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def safe_pointcloud(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Expected a point cloud with at least 4 columns, got {points.shape}")
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


@lru_cache(maxsize=8)
def _load_lisa_model(module_path_text: str, mode: str):
    module_path = Path(module_path_text)
    digest = hashlib.sha256(str(module_path.resolve()).encode("utf-8")).hexdigest()[:16]
    module_name = f"weather_lisa_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LISA module spec from {module_path}")
    lisa = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = lisa
    try:
        spec.loader.exec_module(lisa)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    model = (
        lisa.LISA(mode="gunn", show_progressbar=False)
        if mode == "snow"
        else lisa.LISA(show_progressbar=False)
    )
    return lisa, model


def _apply_lisa_weather(
    injector_root: Path,
    points: np.ndarray,
    severity: int,
    mode: str,
    rng_seed=None,
    weather_threads: int = DEFAULT_WEATHER_THREADS,
) -> np.ndarray:
    validate_fault_spec(f"{mode}_sim", severity)
    if int(weather_threads) < 1:
        raise ValueError("weather_threads must be at least 1")
    if str(injector_root) not in sys.path:
        sys.path.insert(0, str(injector_root))
    patch_compatibility_modules()
    module_path = (injector_root / "utils" / "lisa.py").resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find LISA weather model at {module_path}")
    lisa, model = _load_lisa_model(
        str(module_path),
        mode,
    )
    rate = [0.20, 0.73, 1.5625, 3.125, 7.29][severity - 1]
    original_cpu_count = lisa.mp.cpu_count
    lisa.mp.cpu_count = lambda: int(weather_threads)
    try:
        with temporary_random_seed(rng_seed):
            return model.augment(points, rate)
    finally:
        lisa.mp.cpu_count = original_cpu_count


def apply_fault(
    module,
    injector_root: Path,
    fault: str,
    points: np.ndarray,
    severity: int,
    rng_seed=None,
    weather_threads: int = DEFAULT_WEATHER_THREADS,
) -> np.ndarray:
    fault, severity = validate_fault_spec(fault, severity)
    if fault == "rain_sim":
        return safe_pointcloud(
            _apply_lisa_weather(
                injector_root,
                points.copy(),
                severity,
                "rain",
                rng_seed=rng_seed,
                weather_threads=weather_threads,
            )
        )
    if fault == "snow_sim":
        return safe_pointcloud(
            _apply_lisa_weather(
                injector_root,
                points.copy(),
                severity,
                "snow",
                rng_seed=rng_seed,
                weather_threads=weather_threads,
            )
        )
    if fault not in ROW_ALIGNED_CORRUPTIONS:
        raise ValueError(f"Fault {fault!r} requires a dedicated provenance wrapper")

    func = getattr(module, fault, None)
    if not callable(func):
        raise AttributeError(f"The 3D corruptions module does not define {fault!r}")
    with temporary_random_seed(rng_seed):
        return safe_pointcloud(func(points.copy(), severity))


def filter_pointcloud(
    points: np.ndarray,
    min_range: float,
    max_range: float,
    return_mask: bool = False,
):
    """Remove invalid/out-of-range returns and optionally expose the row mask."""
    if not np.isfinite([min_range, max_range]).all() or max_range <= min_range:
        raise ValueError("Point range must be finite and max_range must exceed min_range")
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


def apply_fog_simulator(
    fog_root: Path,
    points: np.ndarray,
    severity: int,
    noise: int = DEFAULT_FOG_SIMULATOR_NOISE,
    rng_seed=None,
):
    validate_fault_spec("fog_sim", severity)
    if int(noise) < 0:
        raise ValueError(f"Fog noise must be non-negative, got {noise}")
    if str(fog_root) not in sys.path:
        sys.path.insert(0, str(fog_root))


    alpha = FOG_ALPHA_BY_SEVERITY[severity - 1]
    parameter_set = ParameterSet(alpha=alpha, gamma=0.000001)
    original_intensity = points[:, 3].copy()
    with temporary_random_seed(rng_seed):
        hard_pc = P_R_fog_hard(parameter_set, points.copy())
        augmented_pc, _, info = P_R_fog_soft(
            parameter_set,
            hard_pc.copy(),
            original_intensity,
            noise=noise,
            gain=False,
            noise_variant="v1",
        )
    soft_mask = np.linalg.norm(augmented_pc[:, :3] - hard_pc[:, :3], axis=1) > 1e-4
    labels = np.ones((augmented_pc.shape[0], 1), dtype=np.float32)
    labels[soft_mask, 0] = 2.0
    return np.hstack([augmented_pc[:, :4], labels]), {
        "fog_noise": int(noise),
        "fog_alpha": alpha,
        "fog_soft_response_points": int(np.sum(soft_mask)),
        "fog_info_json": json.dumps(info or {}, sort_keys=True),
    }
