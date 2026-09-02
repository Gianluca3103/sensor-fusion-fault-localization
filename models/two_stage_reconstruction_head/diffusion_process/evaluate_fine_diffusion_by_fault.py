"""Evaluate fine diffusion with the same metrics used by coarse reconstruction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head import (
    BEVChannelNormalization,
    CoarseReconstructionDataset,
    FineDiffusionConfig,
    FineDiffusionRefiner,
    FrozenCoarseFineDiffusionPipeline,
    ResidualChannelNormalization,
    coarse_reconstruction_collate,
    coarse_reconstruction_metrics,
    coarse_reconstruction_range_metrics,
    load_frozen_coarse_model,
    validate_fine_diffusion_checkpoint_compatibility,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _load_selector_config,
    _move_batch,
    _occupancy_counts,
    _passes_selector_for_metrics,
    _safe_ratio,
    _target_occupied_cells,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.coarse_loss import (
    BEV_RESOLUTION_M,
    _dilate_with_metric_disk,
)
from models.two_stage_reconstruction_head.diffusion_process.diffusion_metrics import (
    reconstruction_mask_boundary_bands,
    tolerant_metrics_from_counts,
    tolerant_occupancy_counts,
)
from models.two_stage_reconstruction_head.reconstruction_visualization import (
    save_three_panel_reconstruction,
)


REFERENCE_OCCUPANCY_THRESHOLD = 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--coarse-checkpoint",
        type=Path,
        help="Override the frozen coarse checkpoint recorded inside the fine checkpoint.",
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/coarse_reconstruction_vod.json"),
        help="Coarse configuration used to validate cached Fault Selector masks.",
    )
    parser.add_argument(
        "--fine-config",
        type=Path,
        help="Optional fine diffusion config. Used only for channel normalization metadata.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=0.5,
        help=(
            "Physical tolerance in metres used for tolerant occupancy IoU/F1 "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.5,
        help=(
            "Single Fine Diffusion occupancy threshold (default: 0.5). "
            "Superseded by --fine-occupancy-thresholds when that option is used."
        ),
    )
    parser.add_argument(
        "--fine-occupancy-thresholds",
        type=float,
        nargs="+",
        help=(
            "Evaluate several Fine Diffusion occupancy thresholds in one inference "
            "pass. Clean targets and the coarse baseline remain fixed at 0.5."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--visualize-samples-per-fault",
        type=int,
        default=5,
        help="Comparison PNGs saved for each fault group; 0 disables.",
    )
    parser.add_argument(
        "--boundary-diagnostics",
        action="store_true",
        help=(
            "Measure reconstruction quality in inward repair-boundary bands "
            "and save visualization variants at thresholds 0.3, 0.4, and 0.5."
        ),
    )
    return parser.parse_args()


def _normalizer_from_fine_config(
    path: Path | None,
    diffusion_config: FineDiffusionConfig,
) -> BEVChannelNormalization:
    if path is None:
        return BEVChannelNormalization(
            means=(0.0,) * diffusion_config.lidar_channels,
            stds=(1.0,) * diffusion_config.lidar_channels,
            source="identity_from_checkpoint",
        )
    payload = torch.load(path, map_location="cpu", weights_only=False) if path.suffix == ".pt" else None
    if payload is not None:
        section = dict(payload.get("fine_diffusion", {}))
    else:
        import json

        with path.open("r", encoding="utf-8") as handle:
            section = dict(json.load(handle).get("fine_diffusion", {}))
    normalization = dict(section.get("normalization", {}))
    return BEVChannelNormalization(
        means=normalization.get("channel_means", (0.0,) * diffusion_config.lidar_channels),
        stds=normalization.get("channel_stds", (1.0,) * diffusion_config.lidar_channels),
        epsilon=float(normalization.get("epsilon", 1.0e-6)),
        source=normalization.get("source", f"configured:{path}"),
    )


def _diffusion_config_from_checkpoint(
    payload: dict, architecture: dict | None = None
) -> FineDiffusionConfig:
    valid = {item.name for item in fields(FineDiffusionConfig)}
    config_payload = {
        key: value for key, value in dict(payload).items() if key in valid
    }
    if (
        isinstance(architecture, dict)
        and int(architecture.get("version", -1)) == 10
        and "transformer_spatial_input_mode" not in config_payload
    ):
        config_payload["transformer_spatial_input_mode"] = "current_lidar"
    unknown = sorted(set(payload) - valid)
    if unknown:
        raise ValueError(
            "Unknown fine diffusion checkpoint settings: " + ", ".join(unknown)
        )
    config = FineDiffusionConfig(**config_payload)
    config.validate()
    return config


def _stage_outputs(
    bev: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    *,
    occupancy_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if occupancy_logits is None:
        occupancy = bev[:, 0:1].clamp(1.0e-6, 1.0 - 1.0e-6)
        occupancy_logits = torch.logit(occupancy)
    return {
        "reconstruction_mask": reconstruction_mask,
        "occupancy_logits": occupancy_logits,
        "coarse_lidar_bev": bev,
    }


TRANSITION_KEYS = (
    "beneficial_additions",
    "harmful_additions",
    "beneficial_removals",
    "harmful_removals",
)


def _occupancy_transition_counts(
    clean: torch.Tensor,
    coarse: torch.Tensor,
    fine: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    threshold: float,
) -> dict[str, int]:
    """Classify changes while varying only the Fine Diffusion threshold."""

    valid = reconstruction_mask > 0.5
    clean_occupied = clean[:, 0:1] >= REFERENCE_OCCUPANCY_THRESHOLD
    coarse_occupied = coarse[:, 0:1] >= REFERENCE_OCCUPANCY_THRESHOLD
    fine_occupied = fine[:, 0:1] >= threshold
    coarse_empty = ~coarse_occupied
    fine_empty = ~fine_occupied
    return {
        "beneficial_additions": int(
            (valid & clean_occupied & coarse_empty & fine_occupied).sum()
        ),
        "harmful_additions": int(
            (valid & ~clean_occupied & coarse_empty & fine_occupied).sum()
        ),
        "beneficial_removals": int(
            (valid & ~clean_occupied & coarse_occupied & fine_empty).sum()
        ),
        "harmful_removals": int(
            (valid & clean_occupied & coarse_occupied & fine_empty).sum()
        ),
    }


def _threshold_occupancy_counts(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    prediction_threshold: float,
    tolerance_m: float,
    resolution_m: float = BEV_RESOLUTION_M,
) -> dict[str, int]:
    """Return additive exact and tolerant counts for one decision threshold."""

    valid = mask > 0.5
    predicted = (probability >= prediction_threshold) & valid
    occupied = (target >= REFERENCE_OCCUPANCY_THRESHOLD) & valid
    target_neighborhood = _dilate_with_metric_disk(
        occupied, tolerance_m, resolution_m
    )
    prediction_neighborhood = _dilate_with_metric_disk(
        predicted, tolerance_m, resolution_m
    )
    return {
        "tp": int((predicted & occupied).sum()),
        "fp": int((predicted & ~occupied & valid).sum()),
        "fn": int((~predicted & occupied & valid).sum()),
        "tn": int((~predicted & ~occupied & valid).sum()),
        "tolerant_matched_predictions": int(
            (predicted & target_neighborhood).sum()
        ),
        "tolerant_matched_targets": int(
            (occupied & prediction_neighborhood).sum()
        ),
        "tolerant_prediction_count": int(predicted.sum()),
        "tolerant_target_count": int(occupied.sum()),
    }


def _fixed_physical_tolerance_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    geometry,
) -> dict[str, int | float]:
    """Return fixed 0.2 m and 0.5 m metrics using dataset BEV geometry."""

    valid = mask > 0.5
    predicted = probability >= REFERENCE_OCCUPANCY_THRESHOLD
    occupied = target >= REFERENCE_OCCUPANCY_THRESHOLD
    output: dict[str, int | float] = {}
    for tolerance_m in (0.2, 0.5):
        label = str(tolerance_m).replace(".", "_") + "m"
        counts = tolerant_occupancy_counts(
            predicted,
            occupied,
            valid,
            tolerance_m=tolerance_m,
            meters_per_cell_x=geometry.pillar_size_x,
            meters_per_cell_y=geometry.pillar_size_y,
        )
        metrics = tolerant_metrics_from_counts(counts)
        output.update(
            {
                f"tolerant_{label}_{name}": value
                for name, value in (*counts.items(), *metrics.items())
            }
        )
    return output


def _boundary_diagnostic_records(
    *,
    clean: torch.Tensor,
    faulty: torch.Tensor,
    coarse: torch.Tensor,
    fine: torch.Tensor,
    radar: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    geometry,
    sample_path: str,
    fault_group: str,
) -> list[dict[str, int | float | str]]:
    """Collect sufficient statistics for inward repair-boundary bands."""

    clean_occupied = clean[:, 0:1] >= REFERENCE_OCCUPANCY_THRESHOLD
    radar_support = radar.abs().amax(dim=1, keepdim=True) > 0.0
    probabilities = {
        "faulty": faulty[:, 0:1],
        "coarse": coarse[:, 0:1],
        "fine": fine[:, 0:1],
    }
    rows = []
    for band, band_mask in reconstruction_mask_boundary_bands(
        reconstruction_mask
    ).items():
        selected = band_mask.bool()
        target_selected = clean_occupied & selected
        radar_selected = radar_support & selected
        row: dict[str, int | float | str] = {
            "sample_path": sample_path,
            "fault_group": fault_group,
            "band": band,
            "band_cells": int(selected.sum()),
            "clean_occupied_cells": int(target_selected.sum()),
            "radar_support_cells": int(radar_selected.sum()),
            "clean_radar_overlap_cells": int(
                (target_selected & radar_support).sum()
            ),
            "fine_probability_on_clean_sum": float(
                (fine[:, 0:1] * target_selected).sum()
            ),
        }
        for prefix, probability in probabilities.items():
            exact = _occupancy_counts(probability, clean[:, 0:1], selected)
            row.update(
                {f"{prefix}_{name}": int(value) for name, value in exact.items()}
            )
            tolerant = tolerant_occupancy_counts(
                probability >= REFERENCE_OCCUPANCY_THRESHOLD,
                clean_occupied,
                selected,
                tolerance_m=0.2,
                meters_per_cell_x=geometry.pillar_size_x,
                meters_per_cell_y=geometry.pillar_size_y,
            )
            row.update(
                {
                    f"{prefix}_0p2m_{name}": float(value)
                    for name, value in tolerant.items()
                }
            )
        rows.append(row)
    return rows


BOUNDARY_BAND_ORDER = ("0-1 cells", "2-3 cells", "4-7 cells", "8+ cells")


def _summarize_boundary_diagnostics(
    records: list[dict[str, int | float | str]],
) -> list[dict[str, int | float | str]]:
    """Aggregate exact and 0.2 m metrics from boundary sufficient statistics."""

    rows = []
    for band in BOUNDARY_BAND_ORDER:
        selected = [record for record in records if record["band"] == band]
        if not selected:
            continue

        def total(name: str) -> float:
            return sum(float(record[name]) for record in selected)

        target_count = total("clean_occupied_cells")
        radar_count = total("radar_support_cells")
        radar_overlap = total("clean_radar_overlap_cells")
        row: dict[str, int | float | str] = {
            "band": band,
            "samples": sum(int(record["band_cells"]) > 0 for record in selected),
            "band_cells": int(total("band_cells")),
            "clean_occupied_cells": int(target_count),
            "radar_support_cells": int(radar_count),
            "radar_clean_recall": _safe_ratio(radar_overlap, target_count),
            "radar_clean_precision": _safe_ratio(radar_overlap, radar_count),
            "fine_mean_probability_on_clean": _safe_ratio(
                total("fine_probability_on_clean_sum"), target_count
            ),
        }
        for prefix in ("faulty", "coarse", "fine"):
            tp = total(f"{prefix}_tp")
            fp = total(f"{prefix}_fp")
            fn = total(f"{prefix}_fn")
            precision = _safe_ratio(tp, tp + fp)
            recall = _safe_ratio(tp, tp + fn)
            f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
            row.update(
                {
                    f"{prefix}_tp": int(tp),
                    f"{prefix}_fp": int(fp),
                    f"{prefix}_fn": int(fn),
                    f"{prefix}_precision": precision,
                    f"{prefix}_recall": recall,
                    f"{prefix}_f1": f1,
                    f"{prefix}_iou": _safe_ratio(tp, tp + fp + fn),
                }
            )
            tolerant_counts = {
                name: total(f"{prefix}_0p2m_{name}")
                for name in (
                    "matched_predictions",
                    "matched_targets",
                    "prediction_count",
                    "target_count",
                )
            }
            for name, value in tolerant_metrics_from_counts(
                tolerant_counts
            ).items():
                row[f"{prefix}_0p2m_{name}"] = value
        rows.append(row)
    return rows


def _print_boundary_diagnostics(rows: list[dict[str, int | float | str]]) -> None:
    print("\nREPAIR-MASK BOUNDARY DIAGNOSTICS")
    header = (
        f"{'Inward band':<12} {'Cells':>10} {'Clean occ.':>11} "
        f"{'Faulty IoU':>11} {'Base IoU':>10} {'Fine IoU':>10} "
        f"{'Fine R':>9} {'Fine@0.2m':>11} {'Fine FN':>9} "
        f"{'Mean p(y=1)':>12} {'Radar recall':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['band']:<12} {int(row['band_cells']):10d} "
            f"{int(row['clean_occupied_cells']):11d} "
            f"{float(row['faulty_iou']):10.2%} "
            f"{float(row['coarse_iou']):9.2%} "
            f"{float(row['fine_iou']):9.2%} "
            f"{float(row['fine_recall']):8.2%} "
            f"{float(row['fine_0p2m_iou']):10.2%} "
            f"{int(row['fine_fn']):9d} "
            f"{float(row['fine_mean_probability_on_clean']):11.3f} "
            f"{float(row['radar_clean_recall']):12.2%}"
        )


def _threshold_sweep_record(
    clean: torch.Tensor,
    coarse: torch.Tensor,
    fine: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    *,
    fine_threshold: float,
    tolerance_m: float,
) -> dict[str, int | float]:
    """Build the sufficient statistics for one sample and fine threshold."""

    coarse_counts = _threshold_occupancy_counts(
        coarse[:, 0:1],
        clean[:, 0:1],
        reconstruction_mask,
        prediction_threshold=REFERENCE_OCCUPANCY_THRESHOLD,
        tolerance_m=tolerance_m,
    )
    fine_counts = _threshold_occupancy_counts(
        fine[:, 0:1],
        clean[:, 0:1],
        reconstruction_mask,
        prediction_threshold=fine_threshold,
        tolerance_m=tolerance_m,
    )
    valid = reconstruction_mask > 0.5
    clean_occupied = clean[:, 0:1] >= REFERENCE_OCCUPANCY_THRESHOLD
    coarse_occupied = coarse[:, 0:1] >= REFERENCE_OCCUPANCY_THRESHOLD
    record: dict[str, int | float] = {
        "fine_threshold": fine_threshold,
        "repair_cells": int(valid.sum()),
        "target_occupied_cells": coarse_counts["tp"] + coarse_counts["fn"],
        "missing_clean_cells_from_coarse": int(
            (valid & clean_occupied & ~coarse_occupied).sum()
        ),
        "coarse_false_positive_cells": int(
            (valid & ~clean_occupied & coarse_occupied).sum()
        ),
    }
    for prefix, counts in (("coarse", coarse_counts), ("fine", fine_counts)):
        record.update({f"{prefix}_{key}": value for key, value in counts.items()})
    record.update(
        _occupancy_transition_counts(
            clean,
            coarse,
            fine,
            reconstruction_mask,
            fine_threshold,
        )
    )
    return record


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _summarize_threshold_records(
    records: list[dict[str, int | float]],
) -> dict[str, int | float]:
    """Aggregate one threshold using global (micro) occupancy counts."""

    evaluable = [
        record
        for record in records
        if int(record["repair_cells"]) > 0
        and int(record["target_occupied_cells"]) > 0
    ]
    summary: dict[str, int | float] = {
        "fine_threshold": float(records[0]["fine_threshold"]),
        "samples": len(records),
        "metric_samples": len(evaluable),
        "excluded_samples": len(records) - len(evaluable),
        "repair_cells": sum(int(record["repair_cells"]) for record in evaluable),
    }
    for prefix in ("coarse", "fine"):
        for key in (
            "tp",
            "fp",
            "fn",
            "tn",
            "tolerant_matched_predictions",
            "tolerant_matched_targets",
            "tolerant_prediction_count",
            "tolerant_target_count",
        ):
            summary[f"{prefix}_{key}"] = sum(
                int(record[f"{prefix}_{key}"]) for record in evaluable
            )
        tp = int(summary[f"{prefix}_tp"])
        fp = int(summary[f"{prefix}_fp"])
        fn = int(summary[f"{prefix}_fn"])
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2.0 * precision * recall, precision + recall)
        tolerant_precision = _ratio(
            int(summary[f"{prefix}_tolerant_matched_predictions"]),
            int(summary[f"{prefix}_tolerant_prediction_count"]),
        )
        tolerant_recall = _ratio(
            int(summary[f"{prefix}_tolerant_matched_targets"]),
            int(summary[f"{prefix}_tolerant_target_count"]),
        )
        tolerant_f1 = _ratio(
            2.0 * tolerant_precision * tolerant_recall,
            tolerant_precision + tolerant_recall,
        )
        summary.update(
            {
                f"{prefix}_exact_precision": precision,
                f"{prefix}_exact_recall": recall,
                f"{prefix}_exact_f1": f1,
                f"{prefix}_exact_iou": _ratio(tp, tp + fp + fn),
                f"{prefix}_tolerant_precision": tolerant_precision,
                f"{prefix}_tolerant_recall": tolerant_recall,
                f"{prefix}_tolerant_f1": tolerant_f1,
                f"{prefix}_tolerant_iou": _ratio(
                    tolerant_f1, 2.0 - tolerant_f1
                ),
            }
        )
    for key in (
        *TRANSITION_KEYS,
        "missing_clean_cells_from_coarse",
        "coarse_false_positive_cells",
    ):
        summary[key] = sum(int(record[key]) for record in evaluable)
    changed = sum(int(summary[key]) for key in TRANSITION_KEYS)
    summary["changed_cells"] = changed
    summary["addition_precision"] = _ratio(
        int(summary["beneficial_additions"]),
        int(summary["beneficial_additions"])
        + int(summary["harmful_additions"]),
    )
    summary["missing_cell_recovery_rate"] = _ratio(
        int(summary["beneficial_additions"]),
        int(summary["missing_clean_cells_from_coarse"]),
    )
    summary["removal_precision"] = _ratio(
        int(summary["beneficial_removals"]),
        int(summary["beneficial_removals"])
        + int(summary["harmful_removals"]),
    )
    summary["coarse_false_positive_removal_rate"] = _ratio(
        int(summary["beneficial_removals"]),
        int(summary["coarse_false_positive_cells"]),
    )
    summary["fine_minus_coarse_exact_iou"] = (
        float(summary["fine_exact_iou"])
        - float(summary["coarse_exact_iou"])
    )
    summary["fine_minus_coarse_tolerant_iou"] = (
        float(summary["fine_tolerant_iou"])
        - float(summary["coarse_tolerant_iou"])
    )
    return summary


def _print_threshold_sweep(rows: list[dict[str, int | float]], tolerance_m: float) -> None:
    tolerance_label = f"{tolerance_m:g}m"
    print()
    print("FINE OCCUPANCY THRESHOLD SWEEP (coarse and clean fixed at 0.500)")
    header = (
        f"{'Thr':>5} {'Exact IoU':>10} {'Delta':>9} {'F1':>8} {'P':>8} {'R':>8} "
        f"{f'IoU@{tolerance_label}':>10} {f'F1@{tolerance_label}':>10} "
        f"{f'Delta@{tolerance_label}':>12} "
        f"{'Benefit+':>10} {'Harm+':>8} {'Recover':>9} "
        f"{'Benefit-':>10} {'Harm-':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{float(row['fine_threshold']):5.2f} "
            f"{float(row['fine_exact_iou']):9.2%} "
            f"{float(row['fine_minus_coarse_exact_iou']):+8.2%} "
            f"{float(row['fine_exact_f1']):7.2%} "
            f"{float(row['fine_exact_precision']):7.2%} "
            f"{float(row['fine_exact_recall']):7.2%} "
            f"{float(row['fine_tolerant_iou']):9.2%} "
            f"{float(row['fine_tolerant_f1']):9.2%} "
            f"{float(row['fine_minus_coarse_tolerant_iou']):+11.2%} "
            f"{int(row['beneficial_additions']):10d} "
            f"{int(row['harmful_additions']):8d} "
            f"{float(row['missing_cell_recovery_rate']):8.2%} "
            f"{int(row['beneficial_removals']):10d} "
            f"{int(row['harmful_removals']):8d}"
        )


def _rename_coarse_metrics(metrics: dict[str, torch.Tensor], target_prefix: str) -> dict[str, torch.Tensor]:
    renamed = {}
    for key, value in metrics.items():
        if key.startswith("coarse_"):
            renamed[f"{target_prefix}_{key[len('coarse_'):]}"] = value
        elif key.startswith("range_") and "/occupancy" in key:
            renamed[f"{target_prefix}_{key}"] = value
        elif key.startswith("range_") and key.endswith("/repair_cells"):
            renamed[key] = value
        elif key == "outside_mask_max_change":
            renamed[f"{target_prefix}_{key}"] = value
    return renamed


def _stage_metric_bundle(
    prefix: str,
    bev: torch.Tensor,
    clean: torch.Tensor,
    faulty: torch.Tensor,
    mask: torch.Tensor,
    geometry,
    epsilon: float,
    *,
    occupancy_logits: torch.Tensor | None = None,
    observability: torch.Tensor | None = None,
    tolerance_m: float = 0.5,
) -> dict[str, torch.Tensor]:
    outputs = _stage_outputs(bev, mask, occupancy_logits=occupancy_logits)
    metrics = coarse_reconstruction_metrics(
        outputs,
        faulty,
        clean,
        epsilon,
        observability,
        include_tolerant=True,
        tolerance_m=tolerance_m,
    )
    renamed = _rename_coarse_metrics(metrics, prefix)
    ranges = coarse_reconstruction_range_metrics(
        outputs,
        clean,
        x_range=(geometry.x_min, geometry.x_max),
        y_range=(geometry.y_min, geometry.y_max),
        epsilon=epsilon,
        include_tolerant=True,
    )
    renamed.update(_rename_coarse_metrics(ranges, prefix))
    return renamed


def summarize_records(records: list[dict]) -> dict:
    """Macro means plus exact-cell micro metrics for faulty/coarse/fine."""

    if not records:
        return {"samples": 0}
    ignored = {
        "sample_path",
        "fault",
        "severity",
        "fault_group",
        "sequence_id",
        "frame_id",
        "target_occupied_cells",
        "occupancy_threshold",
        "bypassed_coarse_reconstruction",
        *TRANSITION_KEYS,
    }
    metric_keys = [
        key
        for key, value in records[0].items()
        if key not in ignored
        and not key.endswith(("_tp", "_fp", "_fn", "_tn"))
        and not key.endswith(
            (
                "_matched_predictions",
                "_matched_targets",
                "_prediction_count",
                "_target_count",
            )
        )
        and isinstance(value, (int, float))
    ]
    summary = {
        "samples": len(records),
        "total_repair_cells": sum(record["repair_cells"] for record in records),
    }
    evaluable = [record for record in records if _passes_selector_for_metrics(record)]
    selector_rejected = [
        record for record in records if int(record.get("repair_cells", 0)) <= 0
    ]
    empty_target = [
        record
        for record in records
        if int(record.get("repair_cells", 0)) > 0
        and _target_occupied_cells(record) <= 0
    ]
    summary["metric_samples"] = len(evaluable)
    summary["occupancy_metric_samples"] = len(evaluable)
    summary["excluded_selector_rejected_samples"] = len(selector_rejected)
    summary["excluded_empty_target_samples"] = len(empty_target)
    summary["excluded_metric_samples"] = len(records) - len(evaluable)
    summary["repair_cells"] = sum(record["repair_cells"] for record in evaluable)
    transition_total = 0
    for key in TRANSITION_KEYS:
        count = sum(int(record.get(key, 0)) for record in evaluable)
        summary[f"transitions/{key}"] = count
        transition_total += count
    summary["transitions/total_changed_cells"] = transition_total
    for key in TRANSITION_KEYS:
        count = summary[f"transitions/{key}"]
        summary[f"transitions/{key}_fraction_of_changes"] = _safe_ratio(
            count, transition_total
        )
        summary[f"transitions/{key}_rate_per_repair_cell"] = _safe_ratio(
            count, summary["repair_cells"]
        )
    for key in metric_keys:
        if key == "repair_cells":
            continue
        summary[f"macro/{key}"] = (
            sum(record[key] for record in evaluable) / len(evaluable)
            if evaluable
            else 0.0
        )
    for prefix in ("faulty", "coarse", "fine"):
        tp = sum(record[f"{prefix}_tp"] for record in evaluable)
        fp = sum(record[f"{prefix}_fp"] for record in evaluable)
        fn = sum(record[f"{prefix}_fn"] for record in evaluable)
        tn = sum(record[f"{prefix}_tn"] for record in evaluable)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
        summary.update(
            {
                f"micro/{prefix}_tp": tp,
                f"micro/{prefix}_fp": fp,
                f"micro/{prefix}_fn": fn,
                f"micro/{prefix}_tn": tn,
                f"micro/{prefix}_precision": precision,
                f"micro/{prefix}_recall": recall,
                f"micro/{prefix}_f1": f1,
                f"micro/{prefix}_iou": _safe_ratio(tp, tp + fp + fn),
            }
        )
        for tolerance_m in (0.2, 0.5):
            label = str(tolerance_m).replace(".", "_") + "m"
            count_names = (
                "matched_predictions",
                "matched_targets",
                "prediction_count",
                "target_count",
            )
            counts = {
                name: sum(
                    float(
                        record[
                            f"{prefix}_tolerant_{label}_{name}"
                        ]
                    )
                    for record in evaluable
                )
                for name in count_names
            }
            metrics = tolerant_metrics_from_counts(counts)
            for name, value in metrics.items():
                summary[f"micro/{prefix}_tolerant_{label}_{name}"] = value
    summary["micro/coarse_iou_improvement"] = (
        summary["micro/coarse_iou"] - summary["micro/faulty_iou"]
    )
    summary["micro/fine_iou_improvement"] = (
        summary["micro/fine_iou"] - summary["micro/faulty_iou"]
    )
    summary["micro/fine_minus_coarse_iou"] = (
        summary["micro/fine_iou"] - summary["micro/coarse_iou"]
    )
    summary["micro/fine_f1_improvement"] = (
        summary["micro/fine_f1"] - summary["micro/faulty_f1"]
    )
    summary["micro/fine_minus_coarse_f1"] = (
        summary["micro/fine_f1"] - summary["micro/coarse_f1"]
    )
    return summary


def _group_summaries(records: list[dict], key) -> dict[str, dict]:
    groups = defaultdict(list)
    for record in records:
        groups[str(key(record))].append(record)
    return {
        group: summarize_records(group_records)
        for group, group_records in sorted(groups.items())
    }


def _group_raw_records(records: list[dict], key) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for record in records:
        groups[str(key(record))].append(record)
    return dict(groups)


def _print_table(
    groups: dict[str, dict], *, baseline_label: str = "Coarse", tolerance_m: float = 0.5
) -> None:
    tolerance_label = f"{tolerance_m:g}m"
    header = (
        f"{'Fault':<28} {'N':>6} {'Used':>6} {'Faulty IoU':>11} "
        f"{f'{baseline_label} IoU':>15} {'Fine IoU':>10} {'Fine-Faulty':>12} "
        f"{f'Fine-{baseline_label}':>16} {f'Faulty@{tolerance_label}':>13} "
        f"{f'{baseline_label}@{tolerance_label}':>17} "
        f"{f'Fine@{tolerance_label}':>11} "
        f"{f'Fine-Faulty@{tolerance_label}':>18} "
        f"{f'Fine F1@{tolerance_label}':>14} "
        f"{'Fine Halluc.':>13}"
    )
    print(header)
    print("-" * len(header))
    for name, summary in groups.items():
        print(
            f"{name:<28} {summary['samples']:6d} "
            f"{summary.get('metric_samples', summary['samples']):6d} "
            f"{summary['micro/faulty_iou']:10.2%} "
            f"{summary['micro/coarse_iou']:14.2%} "
            f"{summary['micro/fine_iou']:9.2%} "
            f"{summary['micro/fine_iou_improvement']:+11.2%} "
            f"{summary['micro/fine_minus_coarse_iou']:+15.2%} "
            f"{summary['macro/faulty_occupancy_tolerant_iou']:12.2%} "
            f"{summary['macro/coarse_occupancy_tolerant_iou']:16.2%} "
            f"{summary['macro/fine_occupancy_tolerant_iou']:10.2%} "
            f"{summary['macro/fine_tolerant_iou_improvement']:+17.2%} "
            f"{summary['macro/fine_occupancy_tolerant_f1']:13.2%} "
            f"{summary['macro/fine_occupancy_hallucination_rate']:12.2%}"
        )


def _print_transition_table(groups: dict[str, dict], threshold: float) -> None:
    print()
    print(f"COARSE -> FINE OCCUPANCY TRANSITIONS (threshold={threshold:.3f})")
    header = (
        f"{'Fault':<28} {'Used':>6} {'Beneficial +':>14} {'Harmful +':>11} "
        f"{'Beneficial -':>14} {'Harmful -':>11} {'Changed':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, summary in groups.items():
        print(
            f"{name:<28} {summary.get('metric_samples', 0):6d} "
            f"{summary['transitions/beneficial_additions']:14d} "
            f"{summary['transitions/harmful_additions']:11d} "
            f"{summary['transitions/beneficial_removals']:14d} "
            f"{summary['transitions/harmful_removals']:11d} "
            f"{summary['transitions/total_changed_cells']:10d}"
        )


def _print_fixed_metric_summary(summary: dict, baseline_label: str) -> None:
    """Print exact, 0.2 m, and 0.5 m micro metrics together."""

    print("\nOVERALL FIXED OCCUPANCY METRICS")
    print(
        f"  exact IoU/F1     | {baseline_label} "
        f"{summary['micro/coarse_iou']:.2%} / "
        f"{summary['micro/coarse_f1']:.2%} -> fine "
        f"{summary['micro/fine_iou']:.2%} / "
        f"{summary['micro/fine_f1']:.2%}"
    )
    print(
        "  exact improvement| IoU "
        f"{summary['micro/fine_minus_coarse_iou']:+.2%} | F1 "
        f"{summary['micro/fine_minus_coarse_f1']:+.2%}"
    )
    for tolerance_m in (0.2, 0.5):
        label = str(tolerance_m).replace(".", "_") + "m"
        coarse_iou = summary[f"micro/coarse_tolerant_{label}_iou"]
        coarse_f1 = summary[f"micro/coarse_tolerant_{label}_f1"]
        fine_iou = summary[f"micro/fine_tolerant_{label}_iou"]
        fine_f1 = summary[f"micro/fine_tolerant_{label}_f1"]
        print(
            f"  {tolerance_m:g}m IoU/F1      | {baseline_label} "
            f"{coarse_iou:.2%} / {coarse_f1:.2%} -> fine "
            f"{fine_iou:.2%} / {fine_f1:.2%}"
        )
        print(
            f"  {tolerance_m:g}m improvement | IoU "
            f"{fine_iou - coarse_iou:+.2%} | F1 "
            f"{fine_f1 - coarse_f1:+.2%}"
        )


def _save_comparison(
    destination: Path,
    clean_bev: torch.Tensor,
    faulty_bev: torch.Tensor,
    fine_bev: torch.Tensor,
    radar_bev: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    record: dict,
    occupancy_threshold: float = REFERENCE_OCCUPANCY_THRESHOLD,
) -> None:
    baseline_name = (
        "erased faulty"
        if record.get("bypassed_coarse_reconstruction", False)
        else "coarse"
    )
    save_three_panel_reconstruction(
        destination,
        clean_bev=clean_bev,
        faulty_bev=faulty_bev,
        reconstructed_bev=fine_bev,
        radar_bev=radar_bev,
        reconstruction_mask=reconstruction_mask,
        reconstruction_title="Fine reconstruction",
        figure_title=(
            f"{record['fault_group']} | frame {record['frame_id']} | "
            f"faulty {record['faulty_occupancy_exact_iou']:.2%}, "
            f"{baseline_name} {record['coarse_occupancy_exact_iou']:.2%}, "
            f"fine {record['fine_occupancy_exact_iou']:.2%}"
        ),
        occupancy_threshold=occupancy_threshold,
    )


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.visualize_samples_per_fault < 0:
        raise ValueError("visualization count cannot be negative")
    requested_thresholds = (
        args.fine_occupancy_thresholds
        if args.fine_occupancy_thresholds is not None
        else [args.occupancy_threshold]
    )
    if any(not 0.0 < threshold < 1.0 for threshold in requested_thresholds):
        raise ValueError("occupancy thresholds must be strictly between 0 and 1")
    fine_thresholds = tuple(sorted(set(float(value) for value in requested_thresholds)))
    if args.tolerance_m < 0.0:
        raise ValueError("tolerance must be non-negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "diffusion_state_dict" not in checkpoint or "diffusion_config" not in checkpoint:
        raise ValueError("Checkpoint does not contain a fine diffusion model")

    diffusion_config = _diffusion_config_from_checkpoint(
        checkpoint["diffusion_config"],
        checkpoint.get("fine_diffusion_architecture"),
    )
    validate_fine_diffusion_checkpoint_compatibility(
        checkpoint, diffusion_config
    )
    bev_normalizer = _normalizer_from_fine_config(
        args.fine_config, diffusion_config
    )
    bev_metadata = checkpoint.get("bev_normalization")
    if bev_metadata is not None:
        bev_normalizer = BEVChannelNormalization(
            means=bev_metadata["means"],
            stds=bev_metadata["stds"],
            epsilon=float(bev_metadata["epsilon"]),
            source=bev_metadata.get("source", "fine_diffusion_checkpoint"),
        )
    residual_metadata = checkpoint.get("residual_normalization")
    if residual_metadata is None:
        raise ValueError(
            "Fine Diffusion checkpoint lacks train-split residual "
            "normalization. Start a fresh Fine Diffusion run."
        )
    residual_normalizer = ResidualChannelNormalization(
        residual_metadata.get(
            "raw_channel_stds", residual_metadata.get("channel_stds")
        ),
        minimum_std=float(
            residual_metadata.get(
                "minimum_std", diffusion_config.minimum_residual_std
            )
        ),
        source=residual_metadata.get("source", "fine_diffusion_checkpoint"),
    )
    if (
        diffusion_config.bypass_coarse_reconstruction
        and not diffusion_config.use_pointpillars_conditioning
    ):
        coarse_checkpoint_path = None
        coarse = None
        coarse_checkpoint = {}
        use_pointpillars = False
    else:
        recorded_coarse = checkpoint.get("coarse_checkpoint")
        if args.coarse_checkpoint is None and not recorded_coarse:
            raise ValueError("Fine checkpoint does not identify its coarse model")
        coarse_checkpoint_path = args.coarse_checkpoint or Path(recorded_coarse)
        coarse, coarse_checkpoint = load_frozen_coarse_model(
            coarse_checkpoint_path,
            device,
            allow_pointpillars=True,
        )
        use_pointpillars = coarse.config.pointpillars_enabled
    selector_config = _load_selector_config(args.config)
    sample_paths = _split_paths(
        args.data_root, args.split, args.limit_samples, args.seed
    )
    metadata = {str(path): load_sample_metadata(path) for path in sample_paths}
    dataset = CoarseReconstructionDataset(
        sample_paths,
        args.radar_root,
        data_root=args.data_root,
        selector_config=selector_config,
        use_pointpillars=use_pointpillars,
    )
    grid_geometry = (
        coarse.grid_geometry
        if coarse is not None and getattr(coarse, "grid_geometry", None) is not None
        else dataset.grid_geometry
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=coarse_reconstruction_collate,
    )
    diffusion = FineDiffusionRefiner(
        diffusion_config, bev_normalizer, residual_normalizer
    ).to(device)
    diffusion.load_state_dict(checkpoint["diffusion_state_dict"], strict=True)
    pipeline = FrozenCoarseFineDiffusionPipeline(coarse, diffusion).to(device)
    pipeline.eval()

    epsilon = float(coarse_checkpoint.get("loss_config", {}).get("epsilon", 1.0e-8))
    use_amp = device.type == "cuda" and not args.no_amp
    sampling_steps = args.sampling_steps or diffusion_config.sampling_steps
    records = []
    boundary_records: list[dict[str, int | float | str]] = []
    threshold_records: dict[float, list[dict[str, int | float]]] = {
        threshold: [] for threshold in fine_thresholds
    }
    visualized = defaultdict(int)
    completed = 0

    with torch.inference_mode():
        for batch in loader:
            inputs = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                coarse_bev, coarse_outputs = pipeline.coarse_forward(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                )
                sampled = pipeline.sample(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    coarse_lidar_bev=coarse_bev,
                    coarse_output=coarse_outputs,
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                    sampling_steps=sampling_steps,
                )
            fine_bev = sampled["final_lidar_bev"]
            batch_size = inputs["faulty_bev"].shape[0]
            for index in range(batch_size):
                sample_mask = inputs["reconstruction_mask"][index : index + 1]
                clean = inputs["clean_bev"][index : index + 1]
                faulty = inputs["faulty_bev"][index : index + 1]
                coarse_sample = coarse_bev[index : index + 1]
                fine_sample = fine_bev[index : index + 1]
                observability = (
                    inputs["observability_confidence"][index : index + 1]
                    if "observability_confidence" in inputs
                    else None
                )
                coarse_metrics = coarse_reconstruction_metrics(
                    {
                        "reconstruction_mask": sample_mask,
                        "occupancy_logits": coarse_outputs["occupancy_logits"][index : index + 1],
                        "coarse_lidar_bev": coarse_sample,
                    },
                    faulty,
                    clean,
                    epsilon,
                    observability,
                    include_tolerant=True,
                    tolerance_m=args.tolerance_m,
                )
                coarse_metrics.update(
                    coarse_reconstruction_range_metrics(
                        {
                            "reconstruction_mask": sample_mask,
                            "occupancy_logits": coarse_outputs["occupancy_logits"][index : index + 1],
                            "coarse_lidar_bev": coarse_sample,
                        },
                        clean,
                        x_range=(grid_geometry.x_min, grid_geometry.x_max),
                        y_range=(grid_geometry.y_min, grid_geometry.y_max),
                        epsilon=epsilon,
                        include_tolerant=True,
                    )
                )
                fine_metrics = _stage_metric_bundle(
                    "fine",
                    fine_sample,
                    clean,
                    faulty,
                    sample_mask,
                    grid_geometry,
                    epsilon,
                    observability=observability,
                    tolerance_m=args.tolerance_m,
                )
                sample_path = str(batch["sample_path"][index])
                sample_metadata = metadata[sample_path]
                severity = sample_metadata.get("severity", "")
                fault = str(sample_metadata.get("fault", "unknown"))
                record = {
                    "sample_path": sample_path,
                    "sequence_id": str(sample_metadata.get("sequence", "")),
                    "frame_id": str(sample_metadata.get("lidar_index", "")),
                    "fault": fault,
                    "severity": severity,
                    "fault_group": f"{fault}_s{severity}",
                    "bypassed_coarse_reconstruction": (
                        diffusion_config.bypass_coarse_reconstruction
                    ),
                    "repair_cells": int(sample_mask.sum()),
                    **{key: float(value) for key, value in coarse_metrics.items()},
                    **{key: float(value) for key, value in fine_metrics.items()},
                }
                for prefix, probability in (
                    ("faulty", faulty[:, 0:1]),
                    ("coarse", torch.sigmoid(coarse_outputs["occupancy_logits"][index : index + 1])),
                    ("fine", fine_sample[:, 0:1]),
                ):
                    counts = _occupancy_counts(probability, clean[:, 0:1], sample_mask)
                    record.update({f"{prefix}_{key}": value for key, value in counts.items()})
                    physical_tolerance = _fixed_physical_tolerance_metrics(
                        probability,
                        clean[:, 0:1],
                        sample_mask,
                        grid_geometry,
                    )
                    record.update(
                        {
                            f"{prefix}_{key}": value
                            for key, value in physical_tolerance.items()
                        }
                    )
                record["target_occupied_cells"] = record["fine_tp"] + record["fine_fn"]
                record["occupancy_threshold"] = REFERENCE_OCCUPANCY_THRESHOLD
                record.update(
                    _occupancy_transition_counts(
                        clean,
                        coarse_sample,
                        fine_sample,
                        sample_mask,
                        REFERENCE_OCCUPANCY_THRESHOLD,
                    )
                )
                for fine_threshold in fine_thresholds:
                    threshold_records[fine_threshold].append(
                        _threshold_sweep_record(
                            clean,
                            coarse_sample,
                            fine_sample,
                            sample_mask,
                            fine_threshold=fine_threshold,
                            tolerance_m=args.tolerance_m,
                        )
                    )
                record["coarse_exact_iou_improvement"] = (
                    record["coarse_occupancy_exact_iou"]
                    - record["faulty_occupancy_exact_iou"]
                )
                record["fine_exact_iou_improvement"] = (
                    record["fine_occupancy_exact_iou"]
                    - record["faulty_occupancy_exact_iou"]
                )
                record["fine_minus_coarse_exact_iou"] = (
                    record["fine_occupancy_exact_iou"]
                    - record["coarse_occupancy_exact_iou"]
                )
                record["coarse_tolerant_iou_improvement"] = (
                    record["coarse_occupancy_tolerant_iou"]
                    - record["faulty_occupancy_tolerant_iou"]
                )
                record["fine_tolerant_iou_improvement"] = (
                    record["fine_occupancy_tolerant_iou"]
                    - record["faulty_occupancy_tolerant_iou"]
                )
                record["fine_minus_coarse_tolerant_iou"] = (
                    record["fine_occupancy_tolerant_iou"]
                    - record["coarse_occupancy_tolerant_iou"]
                )
                if abs(args.tolerance_m - 0.5) < 1.0e-9:
                    record["coarse_tolerant_0_5m_iou_improvement"] = (
                        record["coarse_occupancy_tolerant_0_5m_iou"]
                        - record["faulty_occupancy_tolerant_0_5m_iou"]
                    )
                    record["fine_tolerant_0_5m_iou_improvement"] = (
                        record["fine_occupancy_tolerant_0_5m_iou"]
                        - record["faulty_occupancy_tolerant_0_5m_iou"]
                    )
                    record["fine_minus_coarse_tolerant_0_5m_iou"] = (
                        record["fine_occupancy_tolerant_0_5m_iou"]
                        - record["coarse_occupancy_tolerant_0_5m_iou"]
                    )
                if args.boundary_diagnostics:
                    boundary_records.extend(
                        _boundary_diagnostic_records(
                            clean=clean,
                            faulty=faulty,
                            coarse=coarse_sample,
                            fine=fine_sample,
                            radar=inputs["radar_bev"][index : index + 1],
                            reconstruction_mask=sample_mask,
                            geometry=grid_geometry,
                            sample_path=sample_path,
                            fault_group=record["fault_group"],
                        )
                    )
                records.append(record)
                if (
                    visualized[record["fault_group"]]
                    < args.visualize_samples_per_fault
                ):
                    visual_index = visualized[record["fault_group"]]
                    destination = (
                        args.output_root
                        / "visualizations"
                        / record["fault_group"]
                        / f"{visual_index:03d}_{Path(sample_path).stem}.png"
                    )
                    visualization_thresholds = (
                        (0.3, 0.4, 0.5)
                        if args.boundary_diagnostics
                        else (REFERENCE_OCCUPANCY_THRESHOLD,)
                    )
                    for visualization_threshold in visualization_thresholds:
                        threshold_destination = (
                            destination.with_name(
                                destination.stem
                                + f"_threshold_{visualization_threshold:.1f}"
                                + destination.suffix
                            )
                            if args.boundary_diagnostics
                            else destination
                        )
                        _save_comparison(
                            threshold_destination,
                            clean[0],
                            faulty[0],
                            fine_sample[0],
                            inputs["radar_bev"][index],
                            sample_mask[0],
                            record,
                            occupancy_threshold=visualization_threshold,
                        )
                    visualized[record["fault_group"]] += 1
            completed += batch_size
            if completed % 500 < batch_size or completed == len(dataset):
                print(f"Evaluated {completed}/{len(dataset)} samples", flush=True)

    by_fault = _group_summaries(records, lambda record: record["fault"])
    by_fault_severity = _group_summaries(records, lambda record: record["fault_group"])
    threshold_sweep = [
        _summarize_threshold_records(threshold_records[threshold])
        for threshold in fine_thresholds
    ]
    summary = {
        "checkpoint": str(args.checkpoint),
        "coarse_checkpoint": (
            str(coarse_checkpoint_path)
            if coarse_checkpoint_path is not None
            else None
        ),
        "bypass_coarse_reconstruction": (
            diffusion_config.bypass_coarse_reconstruction
        ),
        "fine_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "sampling_steps": sampling_steps,
        "tolerance_m": args.tolerance_m,
        "standard_metrics_occupancy_threshold": REFERENCE_OCCUPANCY_THRESHOLD,
        "fine_occupancy_thresholds": list(fine_thresholds),
        "threshold_sweep": threshold_sweep,
        "overall": summarize_records(records),
        "by_fault": by_fault,
        "by_fault_severity": by_fault_severity,
    }
    boundary_summary = None
    if args.boundary_diagnostics:
        overall_boundary = _summarize_boundary_diagnostics(boundary_records)
        boundary_by_fault = {
            fault_group: _summarize_boundary_diagnostics(group_records)
            for fault_group, group_records in sorted(
                _group_raw_records(
                    boundary_records, lambda record: record["fault_group"]
                ).items()
            )
        }
        boundary_summary = {
            "distance_definition": (
                "inward Chebyshev cell distance from the reconstruction-mask "
                "boundary; image edges count as outside"
            ),
            "meters_per_cell_x": grid_geometry.pillar_size_x,
            "meters_per_cell_y": grid_geometry.pillar_size_y,
            "occupancy_threshold": REFERENCE_OCCUPANCY_THRESHOLD,
            "tolerant_metric_m": 0.2,
            "overall": overall_boundary,
            "by_fault_severity": boundary_by_fault,
        }
        summary["boundary_diagnostics"] = boundary_summary
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_root / "per_sample_metrics.csv", records)
    if args.boundary_diagnostics:
        write_csv_rows(
            args.output_root / "boundary_diagnostics_per_sample.csv",
            boundary_records,
        )
        boundary_summary_rows = [
            {"group": "overall", **row}
            for row in boundary_summary["overall"]
        ]
        for fault_group, rows in boundary_summary["by_fault_severity"].items():
            boundary_summary_rows.extend(
                {"group": fault_group, **row} for row in rows
            )
        write_csv_rows(
            args.output_root / "boundary_diagnostics_summary.csv",
            boundary_summary_rows,
        )
        atomic_write_json(
            args.output_root / "boundary_diagnostics.json", boundary_summary
        )
    summary_rows = []
    for group_type, groups in (
        ("fault", by_fault),
        ("fault_severity", by_fault_severity),
    ):
        for group, values in groups.items():
            summary_rows.append({"group_type": group_type, "group": group, **values})
    write_csv_rows(args.output_root / "by_fault_metrics.csv", summary_rows)
    write_csv_rows(
        args.output_root / "occupancy_threshold_sweep.csv", threshold_sweep
    )
    atomic_write_json(
        args.output_root / "occupancy_threshold_sweep.json",
        {
            "coarse_and_target_threshold": REFERENCE_OCCUPANCY_THRESHOLD,
            "tolerance_m": args.tolerance_m,
            "rows": threshold_sweep,
        },
    )
    atomic_write_json(args.output_root / "summary.json", summary)
    print()
    print(f"PER-FAULT FINE-DIFFUSION {args.split.upper()} RESULTS")
    _print_table(
        by_fault_severity,
        baseline_label=(
            "Erased faulty"
            if diffusion_config.bypass_coarse_reconstruction
            else "Coarse"
        ),
        tolerance_m=args.tolerance_m,
    )
    _print_transition_table(
        by_fault_severity, REFERENCE_OCCUPANCY_THRESHOLD
    )
    _print_threshold_sweep(threshold_sweep, args.tolerance_m)
    overall = summary["overall"]
    _print_fixed_metric_summary(
        overall,
        "Erased faulty"
        if diffusion_config.bypass_coarse_reconstruction
        else "Coarse",
    )
    if args.boundary_diagnostics:
        _print_boundary_diagnostics(boundary_summary["overall"])
    print(
        "\nOverall transitions: "
        f"beneficial additions={overall['transitions/beneficial_additions']}, "
        f"harmful additions={overall['transitions/harmful_additions']}, "
        f"beneficial removals={overall['transitions/beneficial_removals']}, "
        f"harmful removals={overall['transitions/harmful_removals']}"
    )
    print(f"\nSaved evaluation to {args.output_root}")


if __name__ == "__main__":
    main()
