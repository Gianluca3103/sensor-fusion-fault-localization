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
from PFS.training_utils import _split_paths, resolve_device, seed_everything
from models.reconstruction_head import (
    CoarseReconstructionConfig,
    CoarseReconstructionDataset,
    CoarseReconstructionModel,
    BEVGridGeometry,
    build_selector_config,
    coarse_reconstruction_collate,
    coarse_reconstruction_metrics,
    coarse_reconstruction_range_metrics,
    load_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/coarse_reconstruction.json"),
        help="Configuration used to validate the cached Fault Selector masks.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
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
    if "lidar_input_bev" in batch:
        moved["lidar_input_bev"] = batch["lidar_input_bev"].to(
            device, non_blocking=True
        )
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
) -> dict[str, int]:
    predicted = probability >= 0.5
    occupied = target >= 0.5
    valid = mask > 0
    return {
        "tp": int((predicted & occupied & valid).sum()),
        "fp": int((predicted & ~occupied & valid).sum()),
        "fn": int((~predicted & occupied & valid).sum()),
        "tn": int((~predicted & ~occupied & valid).sum()),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


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
    }
    metric_keys = [
        key
        for key, value in records[0].items()
        if key not in ignored
        and not key.endswith(("_tp", "_fp", "_fn", "_tn"))
        and isinstance(value, (int, float))
    ]
    summary = {
        "samples": len(records),
        "repair_cells": sum(record["repair_cells"] for record in records),
    }
    for key in metric_keys:
        if key == "repair_cells":
            continue
        summary[f"macro/{key}"] = sum(record[key] for record in records) / len(records)
    for prefix in ("coarse", "faulty"):
        tp = sum(record[f"{prefix}_tp"] for record in records)
        fp = sum(record[f"{prefix}_fp"] for record in records)
        fn = sum(record[f"{prefix}_fn"] for record in records)
        tn = sum(record[f"{prefix}_tn"] for record in records)
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
        f"{'Fault':<28} {'N':>6} {'Faulty IoU':>11} {'Coarse IoU':>11} "
        f"{'Improvement':>12} {'F1@0.5m':>10} {'Halluc.':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, summary in groups.items():
        print(
            f"{name:<28} {summary['samples']:6d} "
            f"{summary['micro/faulty_iou']:10.2%} "
            f"{summary['micro/coarse_iou']:10.2%} "
            f"{summary['micro/iou_improvement']:+11.2%} "
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
    clean = _bev_rgb(clean_bev)
    coarse = _bev_rgb(coarse_bev)
    radar = (
        radar_bev.detach()
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )
    if radar.shape[0] != 4:
        raise ValueError(
            f"Expected four radar BEV channels for visualization, got {radar.shape}"
        )
    radar_composite = np.stack(
        (
            radar[2],
            radar[3],
            np.maximum(radar[1], 0.15 * radar[0]),
        ),
        axis=-1,
    ).clip(0.0, 1.0)
    faulty_support = faulty_bev[0].detach().cpu().numpy() >= 0.5
    radar_support = (
        radar_bev.detach()
        .to(dtype=torch.float32)
        .abs()
        .amax(dim=0)
        .cpu()
        .numpy()
        > 0.0
    )
    faulty_radar_overlay = np.stack(
        (
            radar_support,
            faulty_support,
            faulty_support | radar_support,
        ),
        axis=-1,
    ).astype(np.float32)
    mask = reconstruction_mask.detach().bool().squeeze().cpu().numpy()
    figure, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor="black")
    panels = (
        (clean, "Clean LiDAR BEV", None),
        (coarse, "Coarse reconstructed LiDAR BEV", None),
        (
            faulty_radar_overlay,
            "Faulty LiDAR + trusted radar\nCyan: LiDAR | Magenta: radar | White: overlap",
            None,
        ),
        (
            radar_composite,
            "Radar composite\nR: speed | G: height | B: power/occupancy",
            None,
        ),
        (radar[0], "Radar 0: Static occupancy", "gray"),
        (radar[1], "Radar 1: Normalized power", "inferno"),
        (radar[2], "Radar 2: Dynamic speed", "turbo"),
        (radar[3], "Radar 3: Robust upper height", "viridis"),
    )
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        axis.imshow(
            image,
            cmap=cmap,
            vmin=0.0 if cmap else None,
            vmax=1.0 if cmap else None,
            interpolation="nearest",
        )
        axis.contour(mask.astype(np.uint8), levels=(0.5,), colors="cyan", linewidths=0.7)
        axis.set_title(title, color="white")
        axis.axis("off")
    figure.suptitle(
        f"{record['fault_group']} | sequence {record['sequence_id']} | "
        f"frame {record['frame_id']} | "
        f"faulty IoU {record['faulty_occupancy_exact_iou']:.2%} -> "
        f"coarse IoU {record['coarse_occupancy_exact_iou']:.2%}",
        color="white",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=150,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


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
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain a coarse reconstruction model")
    model_payload = dict(checkpoint["model_config"])
    if "global_channel_multipliers" in model_payload:
        model_payload["global_channel_multipliers"] = tuple(
            model_payload["global_channel_multipliers"]
        )
    model_config = CoarseReconstructionConfig.from_dict(model_payload)

    selector_config = build_selector_config(load_config(args.config))
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
        use_pointpillars=model_config.pointpillars.enabled,
    )
    geometry_payload = checkpoint.get("grid_geometry")
    grid_geometry = (
        BEVGridGeometry(**geometry_payload)
        if geometry_payload is not None
        else dataset.grid_geometry
    )
    model = CoarseReconstructionModel(
        model_config,
        grid_geometry=(grid_geometry if model_config.pointpillars.enabled else None),
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
    radar_mode = str(checkpoint.get("radar_mode", "full"))
    use_global_map = bool(checkpoint.get("global_map_enabled", True))
    epsilon = float(checkpoint.get("loss_config", {}).get("epsilon", 1.0e-8))
    use_amp = device.type == "cuda" and not args.no_amp
    records = []
    visualized = defaultdict(int)
    hrnet_debug_saved = False
    completed = 0

    with torch.inference_mode():
        for batch in loader:
            inputs = _move_batch(batch, device)
            local_radar_bev = None
            radar_enabled = radar_mode != "none"
            local_radar_enabled = radar_mode == "full"
            if radar_mode == "none":
                inputs["radar_bev"] = torch.zeros_like(inputs["radar_bev"])
            elif radar_mode == "global-only":
                if not model_config.pointpillars.enabled:
                    local_radar_bev = torch.zeros_like(inputs["radar_bev"])
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
                    lidar_input_bev=inputs.get("lidar_input_bev"),
                    local_radar_bev=local_radar_bev,
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                    radar_enabled=radar_enabled,
                    local_radar_enabled=local_radar_enabled,
                    use_global_map=use_global_map,
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
                record["exact_iou_improvement"] = (
                    record["coarse_occupancy_exact_iou"]
                    - record["faulty_occupancy_exact_iou"]
                )
                records.append(record)
                if model_config.backbone == "hrnet" and not hrnet_debug_saved:
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
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "radar_mode": radar_mode,
        "global_map_enabled": use_global_map,
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
    atomic_write_json(args.output_root / "summary.json", summary)
    print()
    print("PER-FAULT VALIDATION RESULTS")
    _print_table(by_fault_severity)
    print(f"\nSaved evaluation to {args.output_root}")


if __name__ == "__main__":
    main()
