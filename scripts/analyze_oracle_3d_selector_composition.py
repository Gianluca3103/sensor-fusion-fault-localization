"""Audit clean and missing voxel composition inside oracle 3D selector masks.

The inspection generator saves matching artifacts under::

    <inspection-root>/ground_truth/<sample>/fault_ground_truth_3d.npz
    <inspection-root>/selector/<sample>/oracle_selector_3d.npz

This tool measures each selector mask against the exact provenance-derived
targets.  ``repair`` means a clean voxel is missing or spatially displaced;
``healthy_clean`` means a clean voxel is preserved without a repair/remove
operation.  It reports both the tight operation mask and its contextual halo,
so crop context is never confused with the reconstruction target itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.io_utils import atomic_write_json


MASK_NAMES = ("operation_mask", "context_mask", "context_halo")


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def mask_composition(
    selector_masks: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> dict[str, int | float | None]:
    """Measure every selector mask using the same exact target definition.

    ``missing_repair_voxels`` and ``healthy_clean_voxels`` are disjoint.  The
    latter deliberately uses ``preserve_mask`` instead of ``clean_occupancy``:
    a missing point is present in the clean reference and would otherwise be
    counted as both missing and clean.
    """
    repair = np.asarray(targets["repair_mask"], dtype=bool)
    remove = np.asarray(targets["remove_mask"], dtype=bool)
    preserve = np.asarray(targets["preserve_mask"], dtype=bool)
    clean_occupancy = np.asarray(targets["clean_occupancy"], dtype=bool)
    faulty_occupancy = np.asarray(targets["faulty_occupancy"], dtype=bool)
    expected_shape = repair.shape
    if not all(array.shape == expected_shape for array in (
        remove, preserve, clean_occupancy, faulty_occupancy,
    )):
        raise ValueError("Ground-truth target arrays must have identical shapes")

    result: dict[str, int | float | None] = {
        "total_missing_repair_voxels": int(repair.sum()),
        "total_healthy_clean_voxels": int(preserve.sum()),
    }
    for name in MASK_NAMES:
        mask = np.asarray(selector_masks[name], dtype=bool)
        if mask.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {mask.shape}; expected {expected_shape}"
            )
        missing = int((mask & repair).sum())
        healthy = int((mask & preserve).sum())
        remove_only = int((mask & remove & ~repair).sum())
        changed = int((mask & (repair | remove)).sum())
        clean = int((mask & clean_occupancy).sum())
        faulty = int((mask & faulty_occupancy).sum())
        mask_voxels = int(mask.sum())
        prefix = name.removesuffix("_mask")
        selected_clean_target = missing + healthy
        result.update({
            f"{prefix}_mask_voxels": mask_voxels,
            f"{prefix}_missing_repair_voxels": missing,
            f"{prefix}_healthy_clean_voxels": healthy,
            f"{prefix}_remove_only_voxels": remove_only,
            f"{prefix}_changed_voxels": changed,
            f"{prefix}_clean_occupied_voxels": clean,
            f"{prefix}_faulty_occupied_voxels": faulty,
            f"{prefix}_empty_voxels": int(mask_voxels - (clean_occupancy | faulty_occupancy)[mask].sum()),
            f"{prefix}_missing_recall": _ratio(missing, int(repair.sum())),
            f"{prefix}_healthy_clean_recall": _ratio(healthy, int(preserve.sum())),
            f"{prefix}_missing_share_of_selected_clean": _ratio(missing, selected_clean_target),
            f"{prefix}_healthy_clean_share_of_selected_clean": _ratio(healthy, selected_clean_target),
        })
    return result


def _load_npz(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in names if name not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing arrays: {', '.join(missing)}")
        return {name: np.asarray(archive[name]) for name in names}


def _aggregate(rows: list[dict[str, int | float | str | None]]) -> dict[str, int | float | None]:
    if not rows:
        return {}
    total_keys = (
        "total_missing_repair_voxels",
        "total_healthy_clean_voxels",
    ) + tuple(
        f"{name.removesuffix('_mask')}_{suffix}"
        for name in MASK_NAMES
        for suffix in (
            "mask_voxels",
            "missing_repair_voxels",
            "healthy_clean_voxels",
            "remove_only_voxels",
            "changed_voxels",
            "clean_occupied_voxels",
            "faulty_occupied_voxels",
            "empty_voxels",
        )
    )
    summary: dict[str, int | float | None] = {
        "samples": len(rows),
        **{key: int(sum(int(row[key]) for row in rows)) for key in total_keys},
    }
    for name in MASK_NAMES:
        prefix = name.removesuffix("_mask")
        missing = int(summary[f"{prefix}_missing_repair_voxels"])
        healthy = int(summary[f"{prefix}_healthy_clean_voxels"])
        summary[f"{prefix}_missing_recall"] = _ratio(
            missing, int(summary["total_missing_repair_voxels"])
        )
        summary[f"{prefix}_healthy_clean_recall"] = _ratio(
            healthy, int(summary["total_healthy_clean_voxels"])
        )
        summary[f"{prefix}_missing_share_of_selected_clean"] = _ratio(
            missing, missing + healthy
        )
        summary[f"{prefix}_healthy_clean_share_of_selected_clean"] = _ratio(
            healthy, missing + healthy
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <inspection-root>/mask_composition",
    )
    args = parser.parse_args()

    ground_truth_root = args.inspection_root / "ground_truth"
    selector_root = args.inspection_root / "selector"
    if not ground_truth_root.is_dir() or not selector_root.is_dir():
        raise FileNotFoundError(
            "Expected ground_truth/ and selector/ beneath --inspection-root"
        )
    output_root = args.output_root or args.inspection_root / "mask_composition"
    output_root.mkdir(parents=True, exist_ok=True)

    ground_truth_paths = sorted(ground_truth_root.glob("*/fault_ground_truth_3d.npz"))
    if not ground_truth_paths:
        raise FileNotFoundError(f"No 3D ground-truth archives found in {ground_truth_root}")

    rows: list[dict[str, int | float | str | None]] = []
    target_names = (
        "repair_mask", "remove_mask", "preserve_mask", "clean_occupancy", "faulty_occupancy",
    )
    for ground_truth_path in ground_truth_paths:
        sample_id = ground_truth_path.parent.name
        selector_path = selector_root / sample_id / "oracle_selector_3d.npz"
        if not selector_path.is_file():
            raise FileNotFoundError(f"Missing selector artifact for {sample_id}: {selector_path}")
        row: dict[str, int | float | str | None] = {
            "sample_id": sample_id,
            "ground_truth": str(ground_truth_path),
            "selector": str(selector_path),
        }
        row.update(mask_composition(
            _load_npz(selector_path, MASK_NAMES),
            _load_npz(ground_truth_path, target_names),
        ))
        rows.append(row)

    csv_path = output_root / "mask_voxel_composition.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _aggregate(rows)
    summary.update({
        "inspection_root": str(args.inspection_root),
        "per_sample_csv": str(csv_path),
    })
    summary_path = output_root / "mask_voxel_composition_summary.json"
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Per-sample composition: {csv_path}")
    print(f"Aggregate composition: {summary_path}")


if __name__ == "__main__":
    main()
