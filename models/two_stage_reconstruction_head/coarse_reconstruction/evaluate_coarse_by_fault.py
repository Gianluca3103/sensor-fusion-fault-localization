"""Evaluate a frozen coarse checkpoint and summarize metrics by fault."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from Fault_Localization_Model.sample_utils import load_sample_metadata
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head import (
    CoarseReconstructionConfig,
    CoarseReconstructionDataset,
    CoarseReconstructionModel,
    BEVGridGeometry,
    FaultSelectorConfig,
    build_selector_config,
    coarse_reconstruction_collate,
    coarse_reconstruction_metrics,
    coarse_reconstruction_range_metrics,
    load_config,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.coarse_loss import (
    BEV_RESOLUTION_M,
    OCCUPANCY_TOLERANCE_M,
    _dilate_with_metric_disk,
)
from models.two_stage_reconstruction_head.reconstruction_visualization import (
    save_three_panel_reconstruction,
)


def _load_selector_config(path: Path) -> FaultSelectorConfig:
    """Load either a source training config or a run's resolved config."""

    payload = load_config(path)
    if "selector" not in payload:
        return build_selector_config(payload)
    selector_payload = payload["selector"]
    if not isinstance(selector_payload, dict):
        raise ValueError("resolved selector configuration must be an object")
    config = FaultSelectorConfig(**selector_payload)
    config.validate()
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/coarse_reconstruction_vod.json"),
        help="Configuration used to validate the cached Fault Selector masks.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--occupancy-thresholds",
        type=float,
        nargs="+",
        help=(
            "Evaluate several coarse occupancy thresholds in one inference pass. "
            "Clean targets and the faulty baseline remain fixed at 0.5."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--visualize-samples-per-fault",
        type=int,
        default=5,
        help="Clean/reconstructed comparisons saved for each fault group; 0 disables.",
    )
    return parser.parse_args()


def _move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "faulty_bev",
        "radar_bev",
        "reconstruction_mask",
        "healthy_context_mask",
        "halo_mask",
        "clean_bev",
    )
    moved = {key: batch[key].to(device, non_blocking=True) for key in keys}
    for key in ("faulty_lidar_points", "radar_points"):
        if key in batch:
            moved[key] = tuple(
                points.to(device, non_blocking=True) for points in batch[key]
            )
    if "observability_confidence" in batch:
        moved["observability_confidence"] = batch[
            "observability_confidence"
        ].to(device, non_blocking=True)
    for key in ("reconstruction_mask", "healthy_context_mask", "halo_mask"):
        moved[key] = moved[key].to(dtype=moved["faulty_bev"].dtype)
    return moved


def _occupancy_counts(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    prediction_threshold: float = 0.5,
) -> dict[str, int]:
    predicted = probability >= prediction_threshold
    occupied = target >= 0.5
    valid = mask > 0
    return {
        "tp": int((predicted & occupied & valid).sum()),
        "fp": int((predicted & ~occupied & valid).sum()),
        "fn": int((~predicted & occupied & valid).sum()),
        "tn": int((~predicted & ~occupied & valid).sum()),
    }


def _tolerant_occupancy_counts(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    prediction_threshold: float = 0.5,
) -> dict[str, int]:
    valid = mask > 0
    predicted = (probability >= prediction_threshold) & valid
    occupied = (target >= 0.5) & valid
    target_neighborhood = _dilate_with_metric_disk(
        occupied,
        OCCUPANCY_TOLERANCE_M,
        BEV_RESOLUTION_M,
    )
    prediction_neighborhood = _dilate_with_metric_disk(
        predicted,
        OCCUPANCY_TOLERANCE_M,
        BEV_RESOLUTION_M,
    )
    return {
        "tolerant_matched_predictions": int(
            (predicted & target_neighborhood).sum()
        ),
        "tolerant_matched_targets": int(
            (occupied & prediction_neighborhood).sum()
        ),
        "tolerant_prediction_count": int(predicted.sum()),
        "tolerant_target_count": int(occupied.sum()),
    }


def _coarse_threshold_summary(
    records: list[dict], threshold: float
) -> dict[str, int | float]:
    """Convert additive per-sample counts into one global sweep row."""

    summary = summarize_records(records)
    row: dict[str, int | float] = {
        "coarse_threshold": threshold,
        "samples": summary["samples"],
        "metric_samples": summary["metric_samples"],
        "excluded_samples": summary["excluded_metric_samples"],
        "repair_cells": summary["repair_cells"],
    }
    for prefix in ("faulty", "coarse"):
        for metric in ("precision", "recall", "f1", "iou"):
            row[f"{prefix}_exact_{metric}"] = summary[
                f"micro/{prefix}_{metric}"
            ]
        for metric in ("precision", "recall", "f1", "iou"):
            row[f"{prefix}_tolerant_0_5m_{metric}"] = summary[
                f"micro/{prefix}_occupancy_tolerant_0_5m_{metric}"
            ]
        fp = int(summary[f"micro/{prefix}_fp"])
        tn = int(summary[f"micro/{prefix}_tn"])
        row[f"{prefix}_hallucination_rate"] = _safe_ratio(fp, fp + tn)
    row["coarse_minus_faulty_exact_iou"] = (
        float(row["coarse_exact_iou"]) - float(row["faulty_exact_iou"])
    )
    row["coarse_minus_faulty_tolerant_0_5m_iou"] = (
        float(row["coarse_tolerant_0_5m_iou"])
        - float(row["faulty_tolerant_0_5m_iou"])
    )
    return row


def _print_coarse_threshold_sweep(rows: list[dict[str, int | float]]) -> None:
    print()
    print("COARSE OCCUPANCY THRESHOLD SWEEP (clean and faulty fixed at 0.500)")
    header = (
        f"{'Thr':>5} {'Exact IoU':>10} {'Delta':>9} {'F1':>8} "
        f"{'P':>8} {'R':>8} {'IoU@0.5m':>10} {'F1@0.5m':>10} "
        f"{'Delta@0.5m':>12} {'Halluc.':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{float(row['coarse_threshold']):5.2f} "
            f"{float(row['coarse_exact_iou']):9.2%} "
            f"{float(row['coarse_minus_faulty_exact_iou']):+8.2%} "
            f"{float(row['coarse_exact_f1']):7.2%} "
            f"{float(row['coarse_exact_precision']):7.2%} "
            f"{float(row['coarse_exact_recall']):7.2%} "
            f"{float(row['coarse_tolerant_0_5m_iou']):9.2%} "
            f"{float(row['coarse_tolerant_0_5m_f1']):9.2%} "
            f"{float(row['coarse_minus_faulty_tolerant_0_5m_iou']):+11.2%} "
            f"{float(row['coarse_hallucination_rate']):8.2%}"
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _target_occupied_cells(record: dict) -> int:
    return int(
        record.get(
            "target_occupied_cells",
            record.get("coarse_tp", 0) + record.get("coarse_fn", 0),
        )
    )


def _passes_selector_for_metrics(record: dict) -> bool:
    return int(record.get("repair_cells", 0)) > 0 and _target_occupied_cells(record) > 0


def summarize_records(records: list[dict]) -> dict:
    """Return macro metric means and exact-cell micro occupancy metrics."""

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
    }
    count_suffixes = (
        "_tp",
        "_fp",
        "_fn",
        "_tn",
        "_tolerant_matched_predictions",
        "_tolerant_matched_targets",
        "_tolerant_prediction_count",
        "_tolerant_target_count",
    )
    metric_keys = [
        key
        for key, value in records[0].items()
        if key not in ignored
        and not key.endswith(count_suffixes)
        and isinstance(value, (int, float))
    ]
    summary = {
        "samples": len(records),
        "total_repair_cells": sum(record["repair_cells"] for record in records),
    }
    evaluable_records = [
        record for record in records if _passes_selector_for_metrics(record)
    ]
    selector_rejected = [
        record for record in records if int(record.get("repair_cells", 0)) <= 0
    ]
    empty_target = [
        record
        for record in records
        if int(record.get("repair_cells", 0)) > 0
        and _target_occupied_cells(record) <= 0
    ]
    summary["metric_samples"] = len(evaluable_records)
    summary["occupancy_metric_samples"] = len(evaluable_records)
    summary["excluded_selector_rejected_samples"] = len(selector_rejected)
    summary["excluded_empty_target_samples"] = len(empty_target)
    summary["excluded_metric_samples"] = len(records) - len(evaluable_records)
    summary["repair_cells"] = sum(
        record["repair_cells"] for record in evaluable_records
    )
    for key in metric_keys:
        if key == "repair_cells":
            continue
        summary[f"macro/{key}"] = (
            sum(record[key] for record in evaluable_records)
            / len(evaluable_records)
            if evaluable_records
            else 0.0
        )
    for prefix in ("coarse", "faulty"):
        tp = sum(record[f"{prefix}_tp"] for record in evaluable_records)
        fp = sum(record[f"{prefix}_fp"] for record in evaluable_records)
        fn = sum(record[f"{prefix}_fn"] for record in evaluable_records)
        tn = sum(record[f"{prefix}_tn"] for record in evaluable_records)
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
        tolerant_matched_predictions = sum(
            record.get(f"{prefix}_tolerant_matched_predictions", 0)
            for record in evaluable_records
        )
        tolerant_matched_targets = sum(
            record.get(f"{prefix}_tolerant_matched_targets", 0)
            for record in evaluable_records
        )
        tolerant_prediction_count = sum(
            record.get(f"{prefix}_tolerant_prediction_count", 0)
            for record in evaluable_records
        )
        tolerant_target_count = sum(
            record.get(f"{prefix}_tolerant_target_count", 0)
            for record in evaluable_records
        )
        tolerant_precision = _safe_ratio(
            tolerant_matched_predictions, tolerant_prediction_count
        )
        tolerant_recall = _safe_ratio(
            tolerant_matched_targets, tolerant_target_count
        )
        tolerant_f1 = _safe_ratio(
            2.0 * tolerant_precision * tolerant_recall,
            tolerant_precision + tolerant_recall,
        )
        summary.update(
            {
                f"micro/{prefix}_occupancy_tolerant_0_5m_precision": (
                    tolerant_precision
                ),
                f"micro/{prefix}_occupancy_tolerant_0_5m_recall": (
                    tolerant_recall
                ),
                f"micro/{prefix}_occupancy_tolerant_0_5m_f1": tolerant_f1,
                f"micro/{prefix}_occupancy_tolerant_0_5m_iou": _safe_ratio(
                    tolerant_f1, 2.0 - tolerant_f1
                ),
            }
        )
    summary["micro/iou_improvement"] = (
        summary["micro/coarse_iou"] - summary["micro/faulty_iou"]
    )
    summary["micro/f1_improvement"] = (
        summary["micro/coarse_f1"] - summary["micro/faulty_f1"]
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


def _print_table(groups: dict[str, dict]) -> None:
    header = (
        f"{'Fault':<28} {'N':>6} {'Used':>6} {'Faulty IoU':>11} {'Coarse IoU':>11} "
        f"{'Improvement':>12} {'Faulty@0.5m':>13} {'Coarse@0.5m':>13} "
        f"{'Improvement@0.5m':>18} {'F1@0.5m':>10} {'Halluc.':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, summary in groups.items():
        print(
            f"{name:<28} {summary['samples']:6d} "
            f"{summary.get('metric_samples', summary['samples']):6d} "
            f"{summary['micro/faulty_iou']:10.2%} "
            f"{summary['micro/coarse_iou']:10.2%} "
            f"{summary['micro/iou_improvement']:+11.2%} "
            f"{summary['macro/faulty_occupancy_tolerant_0_5m_iou']:12.2%} "
            f"{summary['macro/coarse_occupancy_tolerant_0_5m_iou']:12.2%} "
            f"{summary['macro/tolerant_0_5m_iou_improvement']:+17.2%} "
            f"{summary['macro/coarse_occupancy_tolerant_0_5m_f1']:9.2%} "
            f"{summary['macro/coarse_occupancy_hallucination_rate']:8.2%}"
        )


def _bev_rgb(bev: torch.Tensor) -> np.ndarray:
    return (
        bev.detach()
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _save_comparison(
    destination: Path,
    clean_bev: torch.Tensor,
    faulty_bev: torch.Tensor,
    radar_bev: torch.Tensor,
    coarse_bev: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    record: dict,
) -> None:
    save_three_panel_reconstruction(
        destination,
        clean_bev=clean_bev,
        faulty_bev=faulty_bev,
        reconstructed_bev=coarse_bev,
        radar_bev=radar_bev,
        reconstruction_mask=reconstruction_mask,
        reconstruction_title="Coarse reconstruction",
        figure_title=(
            f"{record['fault_group']} | sequence {record['sequence_id']} | "
            f"frame {record['frame_id']} | "
            f"faulty IoU {record['faulty_occupancy_exact_iou']:.2%} -> "
            f"coarse IoU {record['coarse_occupancy_exact_iou']:.2%}"
        ),
    )


def _save_hrnet_debug(
    destination: Path,
    clean_bev: torch.Tensor,
    faulty_bev: torch.Tensor,
    coarse_bev: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    occupancy_logits: torch.Tensor,
    activations: dict[str, torch.Tensor],
) -> None:
    """Save the required output and multiresolution HRNet diagnostics."""

    clean = _bev_rgb(clean_bev)
    faulty = _bev_rgb(faulty_bev)
    coarse = _bev_rgb(coarse_bev)
    mask = reconstruction_mask.detach().float().squeeze().cpu().numpy()
    occupancy = (
        torch.sigmoid(occupancy_logits)
        .detach()
        .float()
        .squeeze()
        .cpu()
        .numpy()
    )
    thresholded = occupancy >= 0.5
    absolute_error = np.abs(coarse - clean).mean(axis=-1)
    output_panels = (
        (clean, "Clean LiDAR BEV", None),
        (faulty, "Faulty LiDAR BEV", None),
        (mask, "Reconstruction mask", "gray"),
        (coarse, "HRNet coarse reconstruction", None),
        (occupancy, "Occupancy probability", "viridis"),
        (thresholded, "Occupancy thresholded", "gray"),
        (absolute_error, "Absolute reconstruction error", "magma"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    for axis, (image, title, cmap) in zip(axes.flat, output_panels):
        axis.imshow(image, cmap=cmap, interpolation="nearest")
        axis.set_title(title, color="white")
        axis.axis("off")
    axes.flat[-1].axis("off")
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination / "hrnet_reconstruction_debug.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    feature_names = (
        ("hrnet_stage_4_branch_0", "320x320 branch"),
        ("hrnet_stage_4_branch_1", "160x160 branch"),
        ("hrnet_stage_4_branch_2", "80x80 branch"),
        ("hrnet_stage_4_branch_3", "40x40 branch"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="black")
    for axis, (key, title) in zip(axes, feature_names):
        feature = (
            activations[key]
            .detach()
            .float()
            .abs()
            .mean(dim=0)
            .cpu()
            .numpy()
        )
        axis.imshow(feature, cmap="viridis", interpolation="nearest")
        axis.set_title(title, color="white")
        axis.axis("off")
    figure.savefig(
        destination / "hrnet_branch_activations.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.visualize_samples_per_fault < 0:
        raise ValueError("visualization count cannot be negative")
    occupancy_thresholds = tuple(
        sorted(set(float(value) for value in (args.occupancy_thresholds or ())))
    )
    if any(not 0.0 < threshold < 1.0 for threshold in occupancy_thresholds):
        raise ValueError("occupancy thresholds must be strictly between 0 and 1")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain a coarse reconstruction model")
    model_payload = dict(checkpoint["model_config"])
    model_config = CoarseReconstructionConfig.from_dict(model_payload)

    selector_config = _load_selector_config(args.config)
    sample_paths = _split_paths(
        args.data_root, args.split, args.limit_samples, args.seed
    )
    metadata = {
        str(path): load_sample_metadata(path)
        for path in sample_paths
    }
    dataset = CoarseReconstructionDataset(
        sample_paths,
        args.radar_root,
        data_root=args.data_root,
        selector_config=selector_config,
        use_pointpillars=model_config.pointpillars_enabled,
    )
    geometry_payload = checkpoint.get("grid_geometry")
    grid_geometry = (
        BEVGridGeometry(**geometry_payload)
        if geometry_payload is not None
        else dataset.grid_geometry
    )
    model = CoarseReconstructionModel(
        model_config,
        grid_geometry=(grid_geometry if model_config.pointpillars_enabled else None),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
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
        checkpoint.get(
            "radar_enabled",
            not checkpoint.get("radar_disabled", False),
        )
    )
    epsilon = float(checkpoint.get("loss_config", {}).get("epsilon", 1.0e-8))
    use_amp = device.type == "cuda" and not args.no_amp
    records = []
    threshold_records: dict[float, list[dict]] = {
        threshold: [] for threshold in occupancy_thresholds
    }
    visualized = defaultdict(int)
    hrnet_debug_saved = False
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
                outputs = model(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                    radar_enabled=radar_enabled,
                )
            batch_size = inputs["faulty_bev"].shape[0]
            for index in range(batch_size):
                sample_outputs = {
                    key: outputs[key][index : index + 1]
                    for key in (
                        "reconstruction_mask",
                        "occupancy_logits",
                        "coarse_lidar_bev",
                    )
                }
                observability = (
                    inputs["observability_confidence"][index : index + 1]
                    if "observability_confidence" in inputs
                    else None
                )
                metrics = coarse_reconstruction_metrics(
                    sample_outputs,
                    inputs["faulty_bev"][index : index + 1],
                    inputs["clean_bev"][index : index + 1],
                    epsilon,
                    observability,
                    include_tolerant=True,
                )
                metrics.update(
                    coarse_reconstruction_range_metrics(
                        sample_outputs,
                        inputs["clean_bev"][index : index + 1],
                        x_range=(grid_geometry.x_min, grid_geometry.x_max),
                        y_range=(grid_geometry.y_min, grid_geometry.y_max),
                        epsilon=epsilon,
                        include_tolerant=True,
                    )
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
                    "repair_cells": int(
                        inputs["reconstruction_mask"][index].sum()
                    ),
                    **{key: float(value) for key, value in metrics.items()},
                }
                coarse_counts = _occupancy_counts(
                    torch.sigmoid(sample_outputs["occupancy_logits"]),
                    inputs["clean_bev"][index : index + 1, 0:1],
                    sample_outputs["reconstruction_mask"],
                )
                faulty_counts = _occupancy_counts(
                    inputs["faulty_bev"][index : index + 1, 0:1],
                    inputs["clean_bev"][index : index + 1, 0:1],
                    sample_outputs["reconstruction_mask"],
                )
                record.update(
                    {f"coarse_{key}": value for key, value in coarse_counts.items()}
                )
                record.update(
                    {f"faulty_{key}": value for key, value in faulty_counts.items()}
                )
                coarse_tolerant_counts = _tolerant_occupancy_counts(
                    torch.sigmoid(sample_outputs["occupancy_logits"]),
                    inputs["clean_bev"][index : index + 1, 0:1],
                    sample_outputs["reconstruction_mask"],
                )
                faulty_tolerant_counts = _tolerant_occupancy_counts(
                    inputs["faulty_bev"][index : index + 1, 0:1],
                    inputs["clean_bev"][index : index + 1, 0:1],
                    sample_outputs["reconstruction_mask"],
                )
                record.update(
                    {
                        f"coarse_{key}": value
                        for key, value in coarse_tolerant_counts.items()
                    }
                )
                record.update(
                    {
                        f"faulty_{key}": value
                        for key, value in faulty_tolerant_counts.items()
                    }
                )
                record["target_occupied_cells"] = (
                    coarse_counts["tp"] + coarse_counts["fn"]
                )
                coarse_probability = torch.sigmoid(
                    sample_outputs["occupancy_logits"]
                )
                for threshold in occupancy_thresholds:
                    sweep_coarse_counts = _occupancy_counts(
                        coarse_probability,
                        inputs["clean_bev"][index : index + 1, 0:1],
                        sample_outputs["reconstruction_mask"],
                        prediction_threshold=threshold,
                    )
                    sweep_coarse_tolerant_counts = _tolerant_occupancy_counts(
                        coarse_probability,
                        inputs["clean_bev"][index : index + 1, 0:1],
                        sample_outputs["reconstruction_mask"],
                        prediction_threshold=threshold,
                    )
                    sweep_record = {
                        "sample_path": sample_path,
                        "fault": fault,
                        "severity": severity,
                        "fault_group": record["fault_group"],
                        "sequence_id": record["sequence_id"],
                        "frame_id": record["frame_id"],
                        "repair_cells": record["repair_cells"],
                        "target_occupied_cells": record[
                            "target_occupied_cells"
                        ],
                        **{
                            f"coarse_{key}": value
                            for key, value in sweep_coarse_counts.items()
                        },
                        **{
                            f"coarse_{key}": value
                            for key, value in sweep_coarse_tolerant_counts.items()
                        },
                        **{
                            f"faulty_{key}": value
                            for key, value in faulty_counts.items()
                        },
                        **{
                            f"faulty_{key}": value
                            for key, value in faulty_tolerant_counts.items()
                        },
                    }
                    threshold_records[threshold].append(sweep_record)
                record["exact_iou_improvement"] = (
                    record["coarse_occupancy_exact_iou"]
                    - record["faulty_occupancy_exact_iou"]
                )
                record["tolerant_0_5m_iou_improvement"] = (
                    record["coarse_occupancy_tolerant_0_5m_iou"]
                    - record["faulty_occupancy_tolerant_0_5m_iou"]
                )
                records.append(record)
                if not hrnet_debug_saved:
                    _save_hrnet_debug(
                        args.output_root / "visualizations" / "hrnet_debug",
                        inputs["clean_bev"][index],
                        inputs["faulty_bev"][index],
                        outputs["coarse_lidar_bev"][index],
                        inputs["reconstruction_mask"][index],
                        outputs["occupancy_logits"][index],
                        {
                            key: outputs[key][index]
                            for key in (
                                "hrnet_stage_4_branch_0",
                                "hrnet_stage_4_branch_1",
                                "hrnet_stage_4_branch_2",
                                "hrnet_stage_4_branch_3",
                            )
                        },
                    )
                    hrnet_debug_saved = True
                if (
                    visualized[record["fault_group"]]
                    < args.visualize_samples_per_fault
                ):
                    visual_index = visualized[record["fault_group"]]
                    _save_comparison(
                        args.output_root
                        / "visualizations"
                        / record["fault_group"]
                        / f"{visual_index:03d}_{Path(sample_path).stem}.png",
                        inputs["clean_bev"][index],
                        inputs["faulty_bev"][index],
                        inputs["radar_bev"][index],
                        outputs["coarse_lidar_bev"][index],
                        inputs["reconstruction_mask"][index],
                        record,
                    )
                    visualized[record["fault_group"]] += 1
            completed += batch_size
            if completed % 500 < batch_size or completed == len(dataset):
                print(f"Evaluated {completed}/{len(dataset)} samples", flush=True)

    by_fault = _group_summaries(records, lambda record: record["fault"])
    by_fault_severity = _group_summaries(
        records, lambda record: record["fault_group"]
    )
    threshold_sweep = [
        _coarse_threshold_summary(threshold_records[threshold], threshold)
        for threshold in occupancy_thresholds
    ]
    threshold_sweep_by_fault = []
    for threshold in occupancy_thresholds:
        grouped_records: dict[str, list[dict]] = defaultdict(list)
        for record in threshold_records[threshold]:
            grouped_records[str(record["fault_group"])].append(record)
        for fault_group, group_records in sorted(grouped_records.items()):
            threshold_sweep_by_fault.append(
                {
                    "fault_group": fault_group,
                    **_coarse_threshold_summary(group_records, threshold),
                }
            )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "radar_enabled": radar_enabled,
        "clean_and_faulty_occupancy_threshold": 0.5,
        "coarse_occupancy_thresholds": list(occupancy_thresholds),
        "threshold_sweep": threshold_sweep,
        "threshold_sweep_by_fault": threshold_sweep_by_fault,
        "overall": summarize_records(records),
        "by_fault": by_fault,
        "by_fault_severity": by_fault_severity,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_root / "per_sample_metrics.csv", records)
    summary_rows = []
    for group_type, groups in (
        ("fault", by_fault),
        ("fault_severity", by_fault_severity),
    ):
        for group, values in groups.items():
            summary_rows.append(
                {"group_type": group_type, "group": group, **values}
            )
    write_csv_rows(args.output_root / "by_fault_metrics.csv", summary_rows)
    if threshold_sweep:
        write_csv_rows(
            args.output_root / "occupancy_threshold_sweep.csv",
            threshold_sweep,
        )
        write_csv_rows(
            args.output_root / "occupancy_threshold_sweep_by_fault.csv",
            threshold_sweep_by_fault,
        )
        atomic_write_json(
            args.output_root / "occupancy_threshold_sweep.json",
            {
                "clean_and_faulty_threshold": 0.5,
                "rows": threshold_sweep,
                "by_fault": threshold_sweep_by_fault,
            },
        )
    atomic_write_json(args.output_root / "summary.json", summary)
    print()
    print(f"PER-FAULT {args.split.upper()} RESULTS")
    _print_table(by_fault_severity)
    if threshold_sweep:
        _print_coarse_threshold_sweep(threshold_sweep)
    print(f"\nSaved evaluation to {args.output_root}")


if __name__ == "__main__":
    main()
