import random
from pathlib import Path

from data_injection_utils import apply_fault, apply_fog_simulator, import_lidar_corruptions
from old_laser_fault_injector import apply_old_laser_degradation


EMPTY_FOG_METADATA = {"fog_alpha": "", "fog_soft_response_points": 0, "fog_info_json": ""}


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
    samples = []
    for index in range(num_samples):
        bin_path = bins[index % len(bins)]
        fault, severity = plan[index % len(plan)]
        samples.append((bin_path, fault, severity))
    if shuffle:
        rng.shuffle(samples)
    return samples


def load_fault_injector(injector_root: Path):
    return import_lidar_corruptions(injector_root)


def inject_fault(fault, clean_points, severity, injector_root, fog_root, fog_noise, lidar_corruptions):
    if fault == "fog_sim":
        return apply_fog_simulator(fog_root, clean_points, severity, noise=fog_noise)

    if fault in {"old_laser_degradation", "laser_device_failure"}:
        faulty_raw = apply_old_laser_degradation(clean_points, severity, rng_seed=1000 + int(severity))
        return faulty_raw, EMPTY_FOG_METADATA.copy()

    faulty_raw = apply_fault(lidar_corruptions, injector_root, fault, clean_points, severity)
    return faulty_raw, EMPTY_FOG_METADATA.copy()
