import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_injection_utils import apply_fault, apply_fog_simulator, import_lidar_corruptions
from old_laser_fault_injector import apply_old_laser_degradation


EMPTY_FOG_METADATA = {"fog_alpha": "", "fog_soft_response_points": 0, "fog_info_json": ""}
NO_SOURCE_ID = -1


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
    if len(points) != len(clean_point_ids):
        raise ValueError(
            f"Injector {fault!r} returned {len(points)} rows for {len(clean_point_ids)} inputs. "
            "Exact ID tracking requires an injector-specific row mapping."
        )

    labels = np.ones(len(points), dtype=np.int8)
    if points.shape[1] > 4:
        labels = points[:, 4].astype(np.int8)

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
    kept_ids = clean_point_ids[keep_mask]
    labels = np.ones(len(kept_ids), dtype=np.int8)
    return FaultInjectionResult(np.asarray(points, dtype=np.float32), kept_ids.copy(), kept_ids, labels)


def _fov_keep_mask(points, severity):
    angle1 = [-105, -90, -75, -60, -45][int(severity) - 1]
    angle2 = [105, 90, 75, 60, 45][int(severity) - 1]
    angles = np.degrees(np.arctan2(points[:, 0], points[:, 1]))
    return (angles >= angle1) & (angles <= angle2)


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
        plan.append((fault, severity))
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
            plan.append((fault, severity))
    return plan


def choose_samples(bins, num_samples, seed, plan, shuffle=True):
    rng = random.Random(seed)
    bin_order = list(bins)
    if shuffle:
        rng.shuffle(bin_order)

    samples = []
    for index in range(num_samples):
        if shuffle and index > 0 and index % len(bin_order) == 0:
            rng.shuffle(bin_order)
        bin_path = bin_order[index % len(bin_order)]
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
    fog_noise,
    lidar_corruptions,
):
    """Inject one fault while preserving exact clean-to-faulty point provenance."""
    clean_point_ids = _validate_clean_ids(clean_points, clean_point_ids)

    if fault == "fog_sim":
        faulty_raw, metadata = apply_fog_simulator(fog_root, clean_points, severity, noise=fog_noise)
        return _row_aligned_result(faulty_raw, clean_point_ids, fault), metadata

    if fault in {"old_laser_degradation", "laser_device_failure"}:
        faulty_raw, keep_mask = apply_old_laser_degradation(
            clean_points,
            severity,
            rng_seed=1000 + int(severity),
            return_mask=True,
        )
        return _subset_result(faulty_raw, clean_point_ids, keep_mask), EMPTY_FOG_METADATA.copy()

    if fault == "fov_filter":
        keep_mask = _fov_keep_mask(clean_points, severity)
        faulty_raw = clean_points[keep_mask].copy()
        return _subset_result(faulty_raw, clean_point_ids, keep_mask), EMPTY_FOG_METADATA.copy()

    faulty_raw = apply_fault(lidar_corruptions, injector_root, fault, clean_points, severity)
    return _row_aligned_result(faulty_raw, clean_point_ids, fault), EMPTY_FOG_METADATA.copy()
