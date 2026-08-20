"""Train local cropped residual diffusion after a frozen coarse reconstructor."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import atomic_torch_save, atomic_write_json, write_csv_rows
from models.Fault_Localization.training_utils import (
    _split_paths,
    capture_rng_state,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from models.two_stage_reconstruction_head import (
    BEVChannelNormalization,
    CoarseReconstructionDataset,
    FineDiffusionConfig,
    FineDiffusionRefiner,
    FrozenCoarseFineDiffusionPipeline,
    build_selector_config,
    coarse_reconstruction_collate,
    load_frozen_coarse_model,
    reconstruction_stage_metrics,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument(
        "--coarse-checkpoint",
        help="Required unless fine_diffusion.bypass_coarse_reconstruction is true.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "configs" / "fine_diffusion.json")
    )
    parser.add_argument(
        "--selector-config",
        help="Config whose fault_selector section validates the cached masks.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--resume")
    return parser.parse_args()


def _load_components(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    section = dict(payload.get("fine_diffusion", {}))
    normalization_payload = dict(section.pop("normalization", {}))
    valid = {item.name for item in fields(FineDiffusionConfig)}
    unknown = sorted(set(section) - valid)
    if unknown:
        raise ValueError("Unknown fine_diffusion settings: " + ", ".join(unknown))
    config = FineDiffusionConfig(**section)
    config.validate()
    normalizer = BEVChannelNormalization(
        means=normalization_payload.get(
            "channel_means", (0.0,) * config.lidar_channels
        ),
        stds=normalization_payload.get(
            "channel_stds", (1.0,) * config.lidar_channels
        ),
        epsilon=float(normalization_payload.get("epsilon", 1.0e-6)),
        source=normalization_payload.get("source", "configured"),
    )
    return payload, config, normalizer


def _move_batch(batch, device):
    mapping = {
        "clean_lidar_bev": "clean_bev",
        "faulty_lidar_bev": "faulty_bev",
        "radar_bev": "radar_bev",
        "reconstruction_mask": "reconstruction_mask",
        "healthy_context_mask": "healthy_context_mask",
        "halo_mask": "halo_mask",
    }
    moved = {
        target: batch[source].to(device, non_blocking=True)
        for target, source in mapping.items()
    }
    dtype = moved["faulty_lidar_bev"].dtype
    for key in ("reconstruction_mask", "healthy_context_mask", "halo_mask"):
        moved[key] = moved[key].to(dtype=dtype)
    for key in ("faulty_lidar_points", "radar_points"):
        if key in batch:
            moved[key] = tuple(
                points.to(device, non_blocking=True) for points in batch[key]
            )
    return moved


def _mean_statistics(statistics: dict) -> dict[str, float]:
    return {
        key: float(value.detach()) if torch.is_tensor(value) else float(value)
        for key, value in statistics.items()
    }


def _run_epoch(
    pipeline,
    loader,
    device,
    *,
    optimizer=None,
    scaler=None,
    grad_clip=0.0,
    use_amp=False,
):
    training = optimizer is not None
    pipeline.train(training)
    totals: dict[str, float] = {}
    samples = optimizer_steps = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                output = pipeline(**batch)
                loss = output["loss"]
            if training:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        pipeline.diffusion.parameters(), grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    optimizer_steps += 1
        count = batch["clean_lidar_bev"].shape[0]
        values = {
            "loss": float(loss.detach()),
            "diffusion_loss": float(output["diffusion_loss"].detach()),
            "exact_reconstruction_loss": float(
                output["exact_reconstruction_loss"].detach()
            ),
            "degradation_loss": float(output["degradation_loss"].detach()),
            "residual_regularization_loss": float(
                output["residual_regularization_loss"].detach()
            ),
            "coarse_exact_reconstruction_loss": float(
                output["coarse_exact_reconstruction_loss"].detach()
            ),
            **_mean_statistics(output["statistics"]),
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value * count
        samples += count
        if pipeline.coarse_model is not None and any(
            parameter.grad is not None
            for parameter in pipeline.coarse_model.parameters()
        ):
            raise RuntimeError("Frozen coarse model unexpectedly received gradients")
    result = {key: value / max(samples, 1) for key, value in totals.items()}
    if training:
        result["optimizer_steps"] = optimizer_steps
    return result


@torch.inference_mode()
def _run_sampled_validation(
    pipeline,
    loader,
    device,
    *,
    sampling_steps,
    use_amp=False,
    seed=0,
):
    """Run the deployment sampler; clean LiDAR is used only after prediction."""

    pipeline.eval()
    counts = {
        "fine_tp": 0.0,
        "fine_fp": 0.0,
        "fine_fn": 0.0,
        "coarse_tp": 0.0,
        "coarse_fp": 0.0,
        "coarse_fn": 0.0,
        "empty": 0.0,
    }
    crop_height = crop_width = samples = 0.0
    generator = torch.Generator(device=device).manual_seed(seed)
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=use_amp,
        ):
            sampled = pipeline.sample(
                batch["faulty_lidar_bev"],
                batch["radar_bev"],
                batch["reconstruction_mask"],
                batch["healthy_context_mask"],
                batch["halo_mask"],
                faulty_lidar_points=batch.get("faulty_lidar_points"),
                radar_points=batch.get("radar_points"),
                sampling_steps=sampling_steps,
                generator=generator,
            )

        selected = batch["reconstruction_mask"] > 0.5
        expected = batch["clean_lidar_bev"][:, 0:1] >= 0.5
        coarse = sampled["coarse_lidar_bev"][:, 0:1] >= 0.5
        fine = sampled["final_lidar_bev"][:, 0:1] >= 0.5
        for prefix, predicted in (("coarse", coarse), ("fine", fine)):
            counts[f"{prefix}_tp"] += float(
                (predicted & expected & selected).sum()
            )
            counts[f"{prefix}_fp"] += float(
                (predicted & ~expected & selected).sum()
            )
            counts[f"{prefix}_fn"] += float(
                (~predicted & expected & selected).sum()
            )
        counts["empty"] += float((~expected & selected).sum())

        batch_size = batch["clean_lidar_bev"].shape[0]
        boxes = sampled.get("crop_boxes")
        if boxes is None:
            crop_height += batch_size
            crop_width += batch_size
        else:
            crop_height += float((boxes[:, 1] - boxes[:, 0]).sum())
            crop_width += float((boxes[:, 3] - boxes[:, 2]).sum())
        samples += batch_size

    epsilon = 1.0e-8
    result = {
        "samples": samples,
        "average_crop_height": crop_height / max(samples, 1.0),
        "average_crop_width": crop_width / max(samples, 1.0),
    }
    for prefix in ("coarse", "fine"):
        tp = counts[f"{prefix}_tp"]
        fp = counts[f"{prefix}_fp"]
        fn = counts[f"{prefix}_fn"]
        result[f"{prefix}_exact_occupancy_iou"] = tp / (
            tp + fp + fn + epsilon
        )
        result[f"{prefix}_exact_occupancy_precision"] = tp / (
            tp + fp + epsilon
        )
        result[f"{prefix}_exact_occupancy_recall"] = tp / (
            tp + fn + epsilon
        )
    result["fine_false_positive_occupancy_rate"] = counts["fine_fp"] / (
        counts["empty"] + epsilon
    )
    result["fine_minus_coarse_exact_iou"] = (
        result["fine_exact_occupancy_iou"]
        - result["coarse_exact_occupancy_iou"]
    )
    return result


@torch.no_grad()
def _sampling_metrics(pipeline, loader, device, maximum, sampling_steps):
    rows = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        sampled = pipeline.sample(
            batch["faulty_lidar_bev"],
            batch["radar_bev"],
            batch["reconstruction_mask"],
            batch["healthy_context_mask"],
            batch["halo_mask"],
            faulty_lidar_points=batch.get("faulty_lidar_points"),
            radar_points=batch.get("radar_points"),
            sampling_steps=sampling_steps,
        )
        for index in range(batch["clean_lidar_bev"].shape[0]):
            metrics = reconstruction_stage_metrics(
                batch["faulty_lidar_bev"][index : index + 1]
                * (1.0 - batch["reconstruction_mask"][index : index + 1]),
                sampled["coarse_lidar_bev"][index : index + 1],
                sampled["final_lidar_bev"][index : index + 1],
                batch["faulty_lidar_bev"][index : index + 1],
                batch["clean_lidar_bev"][index : index + 1],
                batch["reconstruction_mask"][index : index + 1],
            )
            rows.append(
                {
                    "sample_path": raw_batch["sample_path"][index],
                    "exact_iou": float(metrics["final"]["occupancy"]["iou"]),
                    "coarse_exact_iou": float(
                        metrics["coarse"]["occupancy"]["iou"]
                    ),
                    "exact_iou_improvement_vs_coarse": float(
                        metrics["final"]["occupancy"]["iou"]
                        - metrics["coarse"]["occupancy"]["iou"]
                    ),
                    "exact_f1": float(metrics["final"]["occupancy"]["f1"]),
                    "exact_precision": float(
                        metrics["final"]["occupancy"]["precision"]
                    ),
                    "exact_recall": float(metrics["final"]["occupancy"]["recall"]),
                    "false_positive_cells": float(
                        metrics["final"]["occupancy"]["fp"]
                    ),
                    "diffusion_mae_improvement": float(
                        metrics["diffusion_improvement"]
                    ),
                }
            )
            if len(rows) >= maximum:
                return rows
    return rows


def main():
    args = _parse_args()
    payload, config, normalizer = _load_components(args.config)
    selector_payload = payload
    if args.selector_config:
        with Path(args.selector_config).open("r", encoding="utf-8") as handle:
            selector_payload = json.load(handle)
    selector = build_selector_config(selector_payload)
    training = dict(payload.get("training", {}))
    epochs = args.epochs or int(training.get("epochs", 50))
    batch_size = args.batch_size or int(training.get("batch_size", 4))
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training.get("num_workers", 4))
    )
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    if config.bypass_coarse_reconstruction:
        coarse = None
        coarse_checkpoint = None
        use_pointpillars = False
    else:
        if not args.coarse_checkpoint:
            raise ValueError(
                "--coarse-checkpoint is required unless coarse reconstruction "
                "is bypassed"
            )
        coarse, coarse_checkpoint = load_frozen_coarse_model(
            args.coarse_checkpoint, device, allow_pointpillars=True
        )
        use_pointpillars = coarse.config.pointpillars_enabled
    dataset_options = {
        "radar_root": args.radar_root,
        "data_root": args.data_root,
        "selector_config": selector,
        "use_pointpillars": use_pointpillars,
    }
    train_dataset = CoarseReconstructionDataset(
        _split_paths(args.data_root, "train", args.limit_train_samples, seed),
        **dataset_options,
    )
    val_dataset = CoarseReconstructionDataset(
        _split_paths(args.data_root, "val", args.limit_val_samples, seed),
        **dataset_options,
    )
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "collate_fn": coarse_reconstruction_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    diffusion = FineDiffusionRefiner(config, normalizer).to(device)
    pipeline = FrozenCoarseFineDiffusionPipeline(coarse, diffusion).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=float(training.get("learning_rate", 2.0e-4)),
        weight_decay=float(training.get("weight_decay", 1.0e-3)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    start_epoch, best, history = 1, float("-inf"), []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        diffusion.load_state_dict(checkpoint["diffusion_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(
            checkpoint.get(
                "best_sampled_validation_iou_improvement", float("-inf")
            )
        )
        history = list(checkpoint.get("history", []))
    atomic_write_json(
        output_root / "resolved_config.json",
        {
            "payload": payload,
            "fine_diffusion": config.to_dict(),
            "normalization": normalizer.metadata(),
            "coarse_model_config": (
                coarse.config.to_dict() if coarse is not None else None
            ),
            "args": vars(args),
        },
    )
    parameters = sum(p.numel() for p in diffusion.parameters())
    print(
        f"Training samples: {len(train_dataset)}; validation: {len(val_dataset)}; "
        f"parameters: {parameters:,}; PointPillars coarse: {use_pointpillars}; "
        f"bypass coarse: {config.bypass_coarse_reconstruction}"
    )
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_stats = _run_epoch(
            pipeline,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=float(training.get("grad_clip", 1.0)),
            use_amp=use_amp,
        )
        validation_started = time.perf_counter()
        val_stats = _run_sampled_validation(
            pipeline,
            val_loader,
            device,
            sampling_steps=config.sampling_steps,
            use_amp=use_amp,
            seed=seed + 10_000,
        )
        validation_seconds = time.perf_counter() - validation_started
        if train_stats.get("optimizer_steps", 0) > 0:
            scheduler.step()
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "runtime/epoch_seconds": elapsed,
            "runtime/sampled_validation_seconds": validation_seconds,
        }
        row.update({f"train/{key}": value for key, value in train_stats.items()})
        row.update({f"val/{key}": value for key, value in val_stats.items()})
        history.append(row)
        baseline_label = (
            "erased faulty"
            if config.bypass_coarse_reconstruction
            else "coarse"
        )
        print(
            f"\nepoch {epoch:03d}/{epochs:03d} | {elapsed:.1f}s | "
            f"sampled validation {validation_seconds:.1f}s\n"
            f"  train | loss {train_stats['loss']:.6f} | "
            f"diffusion {train_stats['diffusion_loss']:.6f} | "
            f"reconstruction {train_stats['exact_reconstruction_loss']:.6f} | "
            f"degradation {train_stats['degradation_loss']:.6f} | "
            f"residual {train_stats['residual_regularization_loss']:.6f}\n"
            f"  sampled validation | {int(val_stats['samples'])} samples | "
            f"crop {val_stats['average_crop_height']:.1f}x"
            f"{val_stats['average_crop_width']:.1f}\n"
            f"    exact mask IoU   | {baseline_label} "
            f"{100.0 * val_stats['coarse_exact_occupancy_iou']:.2f}% "
            f"-> fine {100.0 * val_stats['fine_exact_occupancy_iou']:.2f}% "
            f"({100.0 * val_stats['fine_minus_coarse_exact_iou']:+.2f} pp)\n"
            f"    fine occupancy   | precision "
            f"{100.0 * val_stats['fine_exact_occupancy_precision']:.2f}% | "
            f"recall {100.0 * val_stats['fine_exact_occupancy_recall']:.2f}% | "
            f"false-positive rate "
            f"{100.0 * val_stats['fine_false_positive_occupancy_rate']:.2f}%"
        )
        score = val_stats["fine_minus_coarse_exact_iou"]
        improved = score > best
        best = max(best, score)
        checkpoint = {
            "epoch": epoch,
            "diffusion_state_dict": diffusion.state_dict(),
            "diffusion_config": config.to_dict(),
            "coarse_checkpoint": (
                str(Path(args.coarse_checkpoint).resolve())
                if args.coarse_checkpoint
                else None
            ),
            "coarse_model_config": (
                coarse_checkpoint["model_config"]
                if coarse_checkpoint is not None
                else None
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_sampled_validation_iou_improvement": best,
            "history": history,
            "rng_state": capture_rng_state(),
        }
        atomic_torch_save(checkpoint, output_root / "latest_checkpoint.pt")
        if improved:
            atomic_torch_save(
                checkpoint, output_root / "best_sampled_validation_iou.pt"
            )
        write_csv_rows(output_root / "history.csv", history)
    rows = _sampling_metrics(
        pipeline,
        val_loader,
        device,
        int(payload.get("evaluation", {}).get("max_sampling_samples", 8)),
        config.sampling_steps,
    )
    write_csv_rows(output_root / "sampling_metrics.csv", rows)
    atomic_write_json(output_root / "sampling_metrics.json", rows)


if __name__ == "__main__":
    main()
