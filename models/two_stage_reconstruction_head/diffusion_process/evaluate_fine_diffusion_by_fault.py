"""Evaluate fine diffusion with the same metrics used by coarse reconstruction."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    coarse_reconstruction_collate,
    coarse_reconstruction_metrics,
    coarse_reconstruction_range_metrics,
    load_frozen_coarse_model,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _load_selector_config,
    _move_batch,
    _occupancy_counts,
    _passes_selector_for_metrics,
    _safe_ratio,
    _target_occupied_cells,
)


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
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--visualize-samples-per-fault",
        type=int,
        default=5,
        help="Comparison PNGs saved for each fault group; 0 disables.",
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


def _diffusion_config_from_checkpoint(payload: dict) -> FineDiffusionConfig:
    valid = {item.name for item in fields(FineDiffusionConfig)}
    config_payload = {
        key: value for key, value in dict(payload).items() if key in valid
    }
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
) -> dict[str, torch.Tensor]:
    outputs = _stage_outputs(bev, mask, occupancy_logits=occupancy_logits)
    metrics = coarse_reconstruction_metrics(
        outputs,
        faulty,
        clean,
        epsilon,
        observability,
        include_tolerant=True,
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


def _print_table(
    groups: dict[str, dict], *, baseline_label: str = "Coarse"
) -> None:
    header = (
        f"{'Fault':<28} {'N':>6} {'Used':>6} {'Faulty IoU':>11} "
        f"{f'{baseline_label} IoU':>15} {'Fine IoU':>10} {'Fine-Faulty':>12} "
        f"{f'Fine-{baseline_label}':>16} {'Faulty@0.5m':>13} "
        f"{f'{baseline_label}@0.5m':>17} "
        f"{'Fine@0.5m':>11} {'Fine-Faulty@0.5m':>18} {'Fine F1@0.5m':>14} "
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
            f"{summary['macro/faulty_occupancy_tolerant_0_5m_iou']:12.2%} "
            f"{summary['macro/coarse_occupancy_tolerant_0_5m_iou']:16.2%} "
            f"{summary['macro/fine_occupancy_tolerant_0_5m_iou']:10.2%} "
            f"{summary['macro/fine_tolerant_0_5m_iou_improvement']:+17.2%} "
            f"{summary['macro/fine_occupancy_tolerant_0_5m_f1']:13.2%} "
            f"{summary['macro/fine_occupancy_hallucination_rate']:12.2%}"
        )


def _save_comparison(
    destination: Path,
    clean_bev: torch.Tensor,
    faulty_bev: torch.Tensor,
    coarse_bev: torch.Tensor,
    fine_bev: torch.Tensor,
    reconstruction_mask: torch.Tensor,
    record: dict,
) -> None:
    def occupancy(bev: torch.Tensor):
        return bev.detach().float().cpu()[0].clamp(0.0, 1.0).numpy()

    clean = occupancy(clean_bev)
    faulty = occupancy(faulty_bev)
    coarse = occupancy(coarse_bev)
    fine = occupancy(fine_bev)
    mask = reconstruction_mask.detach().bool().squeeze().cpu().numpy()
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor="black")
    panels = (
        (clean, "Clean occupancy"),
        (faulty, "Faulty occupancy"),
        (coarse, "Coarse occupancy"),
        (fine, "Fine diffusion occupancy"),
    )
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(
            image,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.contour(
            mask.astype("uint8"),
            levels=(0.5,),
            colors="cyan",
            linewidths=0.8,
        )
        axis.set_title(title, color="white")
        axis.axis("off")
    figure.suptitle(
        f"{record['fault_group']} | frame {record['frame_id']} | "
        f"faulty {record['faulty_occupancy_exact_iou']:.2%}, "
        f"coarse {record['coarse_occupancy_exact_iou']:.2%}, "
        f"fine {record['fine_occupancy_exact_iou']:.2%}",
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


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.visualize_samples_per_fault < 0:
        raise ValueError("visualization count cannot be negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "diffusion_state_dict" not in checkpoint or "diffusion_config" not in checkpoint:
        raise ValueError("Checkpoint does not contain a fine diffusion model")

    diffusion_config = _diffusion_config_from_checkpoint(checkpoint["diffusion_config"])
    normalizer = _normalizer_from_fine_config(args.fine_config, diffusion_config)
    if diffusion_config.bypass_coarse_reconstruction:
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
    diffusion = FineDiffusionRefiner(diffusion_config, normalizer).to(device)
    diffusion.load_state_dict(checkpoint["diffusion_state_dict"], strict=True)
    pipeline = FrozenCoarseFineDiffusionPipeline(coarse, diffusion).to(device)
    pipeline.eval()

    epsilon = float(coarse_checkpoint.get("loss_config", {}).get("epsilon", 1.0e-8))
    use_amp = device.type == "cuda" and not args.no_amp
    sampling_steps = args.sampling_steps or diffusion_config.sampling_steps
    records = []
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
                record["target_occupied_cells"] = record["fine_tp"] + record["fine_fn"]
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
                records.append(record)
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
                        clean[0],
                        faulty[0],
                        coarse_sample[0],
                        fine_sample[0],
                        sample_mask[0],
                        record,
                    )
                    visualized[record["fault_group"]] += 1
            completed += batch_size
            if completed % 500 < batch_size or completed == len(dataset):
                print(f"Evaluated {completed}/{len(dataset)} samples", flush=True)

    by_fault = _group_summaries(records, lambda record: record["fault"])
    by_fault_severity = _group_summaries(records, lambda record: record["fault_group"])
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
            summary_rows.append({"group_type": group_type, "group": group, **values})
    write_csv_rows(args.output_root / "by_fault_metrics.csv", summary_rows)
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
    )
    print(f"\nSaved evaluation to {args.output_root}")


if __name__ == "__main__":
    main()
