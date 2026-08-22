"""Evaluate reconstruction geometry inside fixed View-of-Delft GT boxes.

This is deliberately not an object-detector evaluation.  Annotations define
oracle regions only after frozen reconstruction inference has completed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import (
    _split_paths,
    resolve_device,
    seed_everything,
)
from models.two_stage_reconstruction_head import (
    CoarseReconstructionDataset,
    coarse_reconstruction_collate,
    load_frozen_coarse_model,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _load_selector_config,
    _move_batch,
)
from models.two_stage_reconstruction_head.gt_conditioned_reconstruction_metrics import (
    evaluate_bev_condition,
    rasterize_rotated_box,
)
from models.two_stage_reconstruction_head.object_detection.annotations import (
    DEFAULT_VOD_CLASSES,
    VODAnnotationLoader,
)


CONDITIONS = ("faulty", "coarse")
SCOPES = ("full_box", "repair_intersection")
TOLERANCES_M = (0.2, 0.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-checkpoint", required=True, type=Path)
    parser.add_argument("--selector-config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--recovery-f1-threshold",
        type=float,
        default=0.5,
        help="Object is geometrically recovered when repair-region F1@0.5m reaches this value.",
    )
    return parser.parse_args()


def _range_group(distance_m: float) -> str:
    for lower, upper in ((0, 15), (15, 30), (30, 45), (45, 60)):
        if lower <= distance_m < upper:
            return f"{lower}-{upper}m"
    return ">=60m"


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _condition_summary(rows: list[dict], condition: str) -> dict:
    prefix = f"{condition}_"
    usable = [row for row in rows if int(row["clean_occupied_cells"]) > 0]
    summary: dict[str, float | int | None] = {
        "objects": len(rows),
        "metric_objects": len(usable),
        "excluded_without_clean_support": len(rows) - len(usable),
    }
    tp = sum(int(row[prefix + "tp"]) for row in usable)
    fp = sum(int(row[prefix + "fp"]) for row in usable)
    fn = sum(int(row[prefix + "fn"]) for row in usable)
    tn = sum(int(row[prefix + "tn"]) for row in usable)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    summary.update(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "exact_precision": precision,
            "exact_recall": recall,
            "exact_f1": f1,
            "exact_iou": _ratio(tp, tp + fp + fn),
            "hallucination_fraction": _ratio(fp, tp + fp),
        }
    )
    predicted = sum(int(row[prefix + "predicted_occupied_cells"]) for row in usable)
    target = sum(int(row["clean_occupied_cells"]) for row in usable)
    summary["occupied_count_ratio"] = _ratio(predicted, target)
    for tolerance_m in TOLERANCES_M:
        name = str(tolerance_m).replace(".", "_") + "m"
        matched_predictions = sum(
            int(row[prefix + f"tolerant_{name}_matched_predictions"])
            for row in usable
        )
        matched_targets = sum(
            int(row[prefix + f"tolerant_{name}_matched_targets"])
            for row in usable
        )
        tolerant_precision = _ratio(matched_predictions, predicted)
        tolerant_recall = _ratio(matched_targets, target)
        tolerant_f1 = _ratio(
            2.0 * tolerant_precision * tolerant_recall,
            tolerant_precision + tolerant_recall,
        )
        summary.update(
            {
                f"tolerant_{name}_precision": tolerant_precision,
                f"tolerant_{name}_recall": tolerant_recall,
                f"tolerant_{name}_f1": tolerant_f1,
                f"tolerant_{name}_iou": _ratio(tolerant_f1, 2.0 - tolerant_f1),
            }
        )
    clean_support = sum(int(row[prefix + "clean_support_cells"]) for row in usable)
    matched_support = sum(int(row[prefix + "matched_support_cells"]) for row in usable)
    summary["density_mae_clean_support"] = _ratio(
        sum(float(row[prefix + "density_abs_error_sum"]) for row in usable),
        clean_support,
    )
    summary["height_mae_m_clean_support"] = _ratio(
        sum(float(row[prefix + "height_abs_error_m_sum"]) for row in usable),
        clean_support,
    )
    summary["height_mae_m_matched_support"] = _ratio(
        sum(
            float(row[prefix + "matched_height_abs_error_m_sum"])
            for row in usable
        ),
        matched_support,
    )
    chamfer = [
        float(row[prefix + "symmetric_chamfer_m"])
        for row in usable
        if row[prefix + "symmetric_chamfer_m"] is not None
    ]
    summary["macro_symmetric_chamfer_m"] = (
        float(np.mean(chamfer)) if chamfer else None
    )
    return summary


def _summarize(rows: list[dict], recovery_threshold: float) -> dict:
    summary = {
        condition: _condition_summary(rows, condition)
        for condition in CONDITIONS
    }
    for metric in (
        "exact_iou",
        "exact_f1",
        "tolerant_0_2m_iou",
        "tolerant_0_2m_f1",
        "tolerant_0_5m_iou",
        "tolerant_0_5m_f1",
    ):
        summary[f"coarse_minus_faulty_{metric}"] = (
            summary["coarse"][metric] - summary["faulty"][metric]
        )
    affected = [
        row
        for row in rows
        if row["scope"] == "repair_intersection"
        and int(row["clean_occupied_cells"]) > 0
    ]
    lost = [
        row
        for row in affected
        if float(row["faulty_tolerant_0_5m_f1"]) < recovery_threshold
    ]
    recovered = [
        row
        for row in lost
        if float(row["coarse_tolerant_0_5m_f1"]) >= recovery_threshold
    ]
    summary["object_recovery"] = {
        "criterion": f"tolerant_0_5m_f1 >= {recovery_threshold:g}",
        "affected_objects": len(affected),
        "objects_lost_in_faulty": len(lost),
        "objects_recovered_by_coarse": len(recovered),
        "recovery_rate": _ratio(len(recovered), len(lost)),
    }
    return summary


def _flatten_summary(group_type: str, group_name: str, scope: str, summary: dict) -> dict:
    row = {"group_type": group_type, "group": group_name, "scope": scope}
    for condition in CONDITIONS:
        for key, value in summary[condition].items():
            row[f"{condition}_{key}"] = value
    for key, value in summary.items():
        if key.startswith("coarse_minus_faulty_"):
            row[key] = value
    row.update(summary["object_recovery"])
    return row


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if not 0.0 <= args.recovery_f1_threshold <= 1.0:
        raise ValueError("recovery F1 threshold must be in [0,1]")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    coarse_model, checkpoint = load_frozen_coarse_model(
        args.coarse_checkpoint,
        device,
        allow_pointpillars=True,
    )
    selector = _load_selector_config(args.selector_config)
    paths = _split_paths(args.data_root, args.split, args.limit_samples, args.seed)
    metadata = {str(path): load_sample_metadata(path) for path in paths}
    dataset = CoarseReconstructionDataset(
        paths,
        args.radar_root,
        data_root=args.data_root,
        selector_config=selector,
        use_pointpillars=coarse_model.config.pointpillars_enabled,
    )
    geometry = dataset.grid_geometry
    checkpoint_geometry = checkpoint.get("grid_geometry")
    if checkpoint_geometry is not None and checkpoint_geometry != geometry.to_dict():
        raise ValueError("Coarse checkpoint and evaluation dataset use different BEV geometry")
    if not np.isclose(geometry.pillar_size_x, geometry.pillar_size_y):
        raise ValueError("GT-conditioned metric tolerances require square BEV cells")
    annotations = VODAnnotationLoader(
        args.vod_root,
        geometry,
        label_root=args.label_root,
        classes=DEFAULT_VOD_CLASSES,
    )
    frame_ids = [str(metadata[str(path)]["frame_id"]) for path in paths]
    annotations.validate_split(frame_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=coarse_reconstruction_collate,
    )
    radar_enabled = bool(
        checkpoint.get("radar_enabled", not checkpoint.get("radar_disabled", False))
    )
    use_amp = device.type == "cuda" and not args.no_amp
    rows: list[dict] = []
    completed = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = _move_batch(batch, device)
            if not radar_enabled:
                inputs["radar_bev"] = torch.zeros_like(inputs["radar_bev"])
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                output = coarse_model(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                    radar_enabled=radar_enabled,
                )
            clean_batch = inputs["clean_bev"].detach().float().cpu().numpy()
            faulty_batch = inputs["faulty_bev"].detach().float().cpu().numpy()
            coarse_batch = output["coarse_lidar_bev"].detach().float().cpu().numpy()
            repair_batch = inputs["reconstruction_mask"].detach().cpu().numpy()[:, 0] > 0.5
            for index, sample_path in enumerate(batch["sample_path"]):
                sample_metadata = metadata[str(sample_path)]
                frame_id = str(sample_metadata["frame_id"])
                fault = str(sample_metadata.get("fault", "unknown"))
                severity = str(sample_metadata.get("severity", ""))
                fault_group = f"{fault}_s{severity}" if severity else fault
                for object_index, box in enumerate(annotations.load(frame_id)):
                    box_mask = rasterize_rotated_box(box, geometry)
                    distance_m = math.hypot(box.x, box.y)
                    for scope_name, scope_mask in (
                        ("full_box", box_mask),
                        ("repair_intersection", box_mask & repair_batch[index]),
                    ):
                        clean_support = int(
                            ((clean_batch[index, 0] >= 0.5) & scope_mask).sum()
                        )
                        row = {
                            "sample_path": str(sample_path),
                            "frame_id": f"{int(frame_id):05d}",
                            "object_index": object_index,
                            "class_name": box.class_name,
                            "distance_m": distance_m,
                            "range_group": _range_group(distance_m),
                            "fault": fault,
                            "severity": severity,
                            "fault_group": fault_group,
                            "scope": scope_name,
                            "box_x": box.x,
                            "box_y": box.y,
                            "box_z": box.z,
                            "box_length": box.length,
                            "box_width": box.width,
                            "box_height": box.height,
                            "box_yaw": box.yaw,
                            "scope_cells": int(scope_mask.sum()),
                            "clean_occupied_cells": clean_support,
                        }
                        for condition, condition_bev in (
                            ("faulty", faulty_batch[index]),
                            ("coarse", coarse_batch[index]),
                        ):
                            metrics = evaluate_bev_condition(
                                clean_batch[index],
                                condition_bev,
                                scope_mask,
                                resolution_m=geometry.pillar_size_x,
                                tolerances_m=TOLERANCES_M,
                            )
                            row.update(
                                {f"{condition}_{key}": value for key, value in metrics.items()}
                            )
                        row["coarse_minus_faulty_exact_iou"] = (
                            row["coarse_exact_iou"] - row["faulty_exact_iou"]
                        )
                        row["coarse_minus_faulty_tolerant_0_5m_iou"] = (
                            row["coarse_tolerant_0_5m_iou"]
                            - row["faulty_tolerant_0_5m_iou"]
                        )
                        row["lost_in_faulty"] = bool(
                            scope_name == "repair_intersection"
                            and clean_support > 0
                            and row["faulty_tolerant_0_5m_f1"]
                            < args.recovery_f1_threshold
                        )
                        row["recovered_by_coarse"] = bool(
                            row["lost_in_faulty"]
                            and row["coarse_tolerant_0_5m_f1"]
                            >= args.recovery_f1_threshold
                        )
                        rows.append(row)
            completed += len(batch["sample_path"])
            if completed % 100 < len(batch["sample_path"]) or completed == len(dataset):
                print(f"Evaluated {completed}/{len(dataset)} frames", flush=True)

    grouped: dict[str, dict[str, dict[str, dict]]] = {}
    summary_rows = []
    grouping_functions = {
        "overall": lambda row: "all",
        "class": lambda row: row["class_name"],
        "range": lambda row: row["range_group"],
        "fault": lambda row: row["fault_group"],
    }
    for group_type, key_function in grouping_functions.items():
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[str(key_function(row))].append(row)
        grouped[group_type] = {}
        for group_name, group_rows in sorted(buckets.items()):
            grouped[group_type][group_name] = {}
            for scope in SCOPES:
                scope_rows = [row for row in group_rows if row["scope"] == scope]
                summary = _summarize(scope_rows, args.recovery_f1_threshold)
                grouped[group_type][group_name][scope] = summary
                summary_rows.append(
                    _flatten_summary(group_type, group_name, scope, summary)
                )

    if not rows:
        raise RuntimeError("No supported annotated VoD objects were found")

    output = {
        "evaluation": "detector-independent GT-conditioned reconstruction geometry",
        "split": args.split,
        "frames": len(paths),
        "object_scope_rows": len(rows),
        "coarse_checkpoint": str(args.coarse_checkpoint.resolve()),
        "coarse_epoch": int(checkpoint.get("epoch", -1)),
        "annotations_are_oracle_regions_only": True,
        "ground_truth_used_by_reconstruction_model": False,
        "occupancy_threshold": 0.5,
        "tolerances_m": list(TOLERANCES_M),
        "scopes": {
            "full_box": "complete rotated GT box footprint",
            "repair_intersection": "GT box intersected with reconstruction mask",
        },
        "groups": grouped,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "summary.json", output)
    atomic_write_json(args.output_root / "per_object_metrics.json", rows)
    write_csv_rows(args.output_root / "per_object_metrics.csv", rows)
    write_csv_rows(args.output_root / "summary.csv", summary_rows)

    repair_summary = grouped["overall"]["all"]["repair_intersection"]
    print("\nGT-CONDITIONED RECONSTRUCTION ACCURACY (repair intersection)")
    print(f"Objects with clean support: {repair_summary['coarse']['metric_objects']}")
    print(
        "Exact IoU: "
        f"faulty {repair_summary['faulty']['exact_iou']:.2%} -> "
        f"coarse {repair_summary['coarse']['exact_iou']:.2%} "
        f"({repair_summary['coarse_minus_faulty_exact_iou']:+.2%})"
    )
    print(
        "F1@0.5m:  "
        f"faulty {repair_summary['faulty']['tolerant_0_5m_f1']:.2%} -> "
        f"coarse {repair_summary['coarse']['tolerant_0_5m_f1']:.2%} "
        f"({repair_summary['coarse_minus_faulty_tolerant_0_5m_f1']:+.2%})"
    )
    print(json.dumps(repair_summary["object_recovery"], indent=2))
    print(f"Saved results to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
