import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Fault_Localization_Model.data_injection_utils import (
    DEFAULT_FOG_SIMULATOR_NOISE,
    DEFAULT_WEATHER_THREADS,
    apply_fault,
    apply_fog_simulator,
    import_lidar_corruptions,
    validate_fault_spec,
)
from Fault_Localization_Model.old_laser_fault_injector import apply_old_laser_degradation


EMPTY_FOG_METADATA = {
    "fog_alpha": "",
    "fog_soft_response_points": 0,
    "fog_info_json": "",
    "fov_center_deg": "",
    "fov_retained_width_deg": "",
    "fov_missing_center_deg": "",
    "fov_missing_width_deg": "",
    "fov_source": "",
}
NO_SOURCE_ID = -1
INJECTION_VERSION = 4


@dataclass(frozen=True)
class FaultInjectionResult:
    """Corrupted returns and their exact relationship to the clean input rows."""

    points: np.ndarray
    point_ids: np.ndarray
    source_ids: np.ndarray
    injector_labels: np.ndarray


def _validate_clean_ids(clean_points, clean_point_ids):
    clean_point_ids = np.asarray(clean_point_ids, dtype=np.int64)
    if clean_point_ids.shape != (len(clean_points),):
        raise ValueError(
            f"Expected one clean point ID per row, got {clean_point_ids.shape} for {len(clean_points)} points."
        )
    if len(np.unique(clean_point_ids)) != len(clean_point_ids):
        raise ValueError("Clean point IDs must be unique within a frame.")
    if np.any(clean_point_ids < 0):
        raise ValueError("Clean point IDs must be non-negative; negative IDs are reserved for missing provenance.")
    return clean_point_ids


def _row_aligned_result(points, clean_point_ids, fault):
    """Create provenance for injectors whose output row i derives from input row i."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Injector {fault!r} returned an invalid point-cloud shape: {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"Injector {fault!r} returned non-finite point values")
    if len(points) != len(clean_point_ids):
        raise ValueError(
            f"Injector {fault!r} returned {len(points)} rows for {len(clean_point_ids)} inputs. "
            "Exact ID tracking requires an injector-specific row mapping."
        )

    labels = np.ones(len(points), dtype=np.int8)
    if points.shape[1] > 4:
        raw_labels = points[:, 4]
        if not np.isfinite(raw_labels).all():
            raise ValueError(f"Injector {fault!r} returned non-finite provenance labels")
        if not np.all(np.equal(raw_labels, np.round(raw_labels))):
            raise ValueError(f"Injector {fault!r} returned non-integral provenance labels")
        labels = raw_labels.astype(np.int8)
        unexpected = set(np.unique(labels)) - {0, 1, 2}
        if unexpected:
            raise ValueError(
                f"Injector {fault!r} returned unsupported provenance labels: "
                f"{sorted(unexpected)}"
            )

    source_ids = clean_point_ids.copy()
    point_ids = clean_point_ids.copy()
    synthetic = labels == 2
    if np.any(synthetic):
        next_id = int(clean_point_ids.max()) + 1 if len(clean_point_ids) else 0
        point_ids[synthetic] = next_id + np.arange(np.sum(synthetic), dtype=np.int64)
        source_ids[synthetic] = NO_SOURCE_ID

    return FaultInjectionResult(points, point_ids, source_ids, labels)


def _subset_result(points, clean_point_ids, keep_mask):
    """Create provenance for injectors that return an ordered subset of input rows."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape != (len(clean_point_ids),):
        raise ValueError(f"Injector keep mask has shape {keep_mask.shape}, expected {(len(clean_point_ids),)}")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"Subset injector returned an invalid point-cloud shape: {points.shape}")
    if len(points) != int(keep_mask.sum()):
        raise ValueError(
            f"Subset injector returned {len(points)} rows but its keep mask selects "
            f"{int(keep_mask.sum())}"
        )
    if not np.isfinite(points).all():
        raise ValueError("Subset injector returned non-finite point values")
    kept_ids = clean_point_ids[keep_mask]
    labels = np.ones(len(kept_ids), dtype=np.int8)
    return FaultInjectionResult(points, kept_ids.copy(), kept_ids, labels)


def _ordered_subset_keep_mask(clean_points, subset_points):
    """Recover the keep mask for an injector that returns rows from the input in order."""

    clean_points = np.asarray(clean_points)
    subset_points = np.asarray(subset_points)
    keep_mask = np.zeros(len(clean_points), dtype=bool)
    subset_index = 0
    for clean_index, clean_row in enumerate(clean_points):
        if subset_index >= len(subset_points):
            break
        if np.array_equal(clean_row, subset_points[subset_index]):
            keep_mask[clean_index] = True
            subset_index += 1
    if subset_index != len(subset_points):
        raise ValueError(
            "Could not recover the ordered subset mask for fov_filter. "
            "The injector must return an ordered subset of input rows."
        )
    return keep_mask


def _apply_literature_fov_filter(lidar_corruptions, clean_points, severity):
    func = getattr(lidar_corruptions, "fov_filter", None)
    if not callable(func):
        raise AttributeError("The 3D corruptions module does not define 'fov_filter'")
    faulty_raw = np.asarray(func(clean_points.copy(), severity), dtype=np.float32)
    if faulty_raw.ndim != 2 or faulty_raw.shape[1] < 4:
        raise ValueError(f"fov_filter returned an invalid point-cloud shape: {faulty_raw.shape}")
    if not np.isfinite(faulty_raw).all():
        raise ValueError("fov_filter returned non-finite point values")
    keep_mask = _ordered_subset_keep_mask(clean_points, faulty_raw)
    return faulty_raw, keep_mask


def parse_fault_plan(items):
    plan = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Fault plan item must look like fault:severity, got {item!r}")
        fault, severity_text = item.split(":", 1)
        fault = fault.strip()
        if not fault:
            raise ValueError(f"Missing fault name in plan item {item!r}")
        try:
            severity = int(severity_text)
        except ValueError as exc:
            raise ValueError(f"Severity must be an integer in plan item {item!r}") from exc
        plan.append(validate_fault_spec(fault, severity))
    if not plan:
        raise ValueError("--fault-plan was provided but no valid items were parsed.")
    return plan


def build_fault_plan(fault_plan_items, faults, severities, default_fault_plan):
    if fault_plan_items:
        return parse_fault_plan(fault_plan_items)

    selected_faults = faults if faults else [fault for fault, _ in default_fault_plan]
    default_by_fault = dict(default_fault_plan)
    plan = []
    for fault in selected_faults:
        selected_severities = severities
        if not selected_severities:
            selected_severities = [default_by_fault.get(fault, 5 if fault in {"rain_sim", "snow_sim", "fog_sim"} else 1)]
        for severity in selected_severities:
            plan.append(validate_fault_spec(fault, severity))
    if not plan:
        raise ValueError("At least one fault/severity pair is required.")
    return plan


def choose_samples(bins, num_samples, seed, plan, shuffle=True):
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if not bins:
        raise ValueError("At least one LiDAR frame is required.")
    if not plan:
        raise ValueError("At least one fault-plan item is required.")
    rng = random.Random(seed)
    bin_order = list(bins)
    if len(set(bin_order)) != len(bin_order):
        raise ValueError("LiDAR candidate paths must be unique.")
    if int(num_samples) > len(bin_order):
        raise ValueError(
            f"Requested {num_samples} samples from only {len(bin_order)} "
            "unique LiDAR frames. Source frames are never repeated."
        )
    if shuffle:
        rng.shuffle(bin_order)

    samples = []
    for index in range(num_samples):
        bin_path = bin_order[index]
        fault, severity = plan[index % len(plan)]
        samples.append((bin_path, fault, severity))
    if shuffle:
        rng.shuffle(samples)
    return samples


def load_fault_injector(injector_root: Path):
    return import_lidar_corruptions(injector_root)


def inject_fault(
    fault,
    clean_points,
    clean_point_ids,
    severity,
    injector_root,
    fog_root,
    fog_noise=DEFAULT_FOG_SIMULATOR_NOISE,
    lidar_corruptions=None,
    rng_seed=None,
    weather_threads=DEFAULT_WEATHER_THREADS,
):
    """Inject one fault while preserving exact clean-to-faulty point provenance."""
    fault, severity = validate_fault_spec(fault, severity)
    clean_point_ids = _validate_clean_ids(clean_points, clean_point_ids)
    metadata = {
        **EMPTY_FOG_METADATA,
        "injection_version": INJECTION_VERSION,
        "injection_seed": "" if rng_seed is None else int(rng_seed),
        "fog_noise": int(fog_noise),
        "weather_threads": int(weather_threads),
    }

    if fault == "fog_sim":
        faulty_raw, fog_metadata = apply_fog_simulator(
            fog_root,
            clean_points,
            severity,
            noise=fog_noise,
            rng_seed=rng_seed,
        )
        metadata.update(fog_metadata)
        return _row_aligned_result(faulty_raw, clean_point_ids, fault), metadata

    if fault in {"old_laser_degradation", "laser_device_failure"}:
        faulty_raw, keep_mask = apply_old_laser_degradation(
            clean_points,
            severity,
            rng_seed=0 if rng_seed is None else rng_seed,
            return_mask=True,
        )
        return _subset_result(faulty_raw, clean_point_ids, keep_mask), metadata

    if fault == "fov_filter":
        faulty_raw, keep_mask = _apply_literature_fov_filter(
            lidar_corruptions,
            clean_points,
            severity,
        )
        metadata.update(
            {
                "fov_source": "Weather_Injector/3D_Corruptions_AD/LiDAR_corruptions.py:fov_filter",
            }
        )
        return _subset_result(faulty_raw, clean_point_ids, keep_mask), metadata

    faulty_raw = apply_fault(
        lidar_corruptions,
        injector_root,
        fault,
        clean_points,
        severity,
        rng_seed=rng_seed,
        weather_threads=weather_threads,
    )
    return _row_aligned_result(faulty_raw, clean_point_ids, fault), metadata
