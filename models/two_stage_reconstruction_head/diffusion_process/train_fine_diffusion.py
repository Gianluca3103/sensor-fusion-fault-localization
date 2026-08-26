"""Train local cropped residual diffusion after a frozen coarse reconstructor."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
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
    ResidualChannelNormalization,
    build_selector_config,
    coarse_reconstruction_collate,
    load_frozen_coarse_model,
    estimate_training_residual_statistics,
    fine_diffusion_architecture_metadata,
    reconstruction_stage_metrics,
    validate_fine_diffusion_checkpoint_compatibility,
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
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--resume")
    parser.add_argument(
        "--residual-statistics-only",
        action="store_true",
        help="Estimate train-only coarse residual statistics and exit.",
    )
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
    bev_normalizer = BEVChannelNormalization(
        means=normalization_payload.get(
            "channel_means", (0.0,) * config.lidar_channels
        ),
        stds=normalization_payload.get(
            "channel_stds", (1.0,) * config.lidar_channels
        ),
        epsilon=float(normalization_payload.get("epsilon", 1.0e-6)),
        source=normalization_payload.get("source", "configured"),
    )
    return payload, config, bev_normalizer


def _residual_normalizer(metadata, config):
    raw = metadata.get("raw_channel_stds", metadata.get("channel_stds"))
    if raw is None:
        raise KeyError("Residual normalization requires channel stds")
    return ResidualChannelNormalization(
        raw,
        minimum_std=float(
            metadata.get("minimum_std", config.minimum_residual_std)
        ),
        source=metadata.get("source", "fine_diffusion_checkpoint"),
    )


def _resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in choices:
        raise ValueError("training.amp_dtype must be bfloat16 or float16")
    return choices[normalized]


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


class _BatchProgress:
    """Small dependency-free progress display that also remains useful in logs."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(int(total), 1)
        self.started = time.perf_counter()
        # Refresh roughly twice per percentage point without flooding stdout.
        self.interval = max(self.total // 200, 1)

    def update(
        self,
        completed: int,
        *,
        samples: int,
        loss: float | None = None,
        data_seconds: float | None = None,
        step_seconds: float | None = None,
    ):
        if completed != self.total and completed % self.interval != 0:
            return
        elapsed = max(time.perf_counter() - self.started, 1.0e-6)
        rate = completed / elapsed
        remaining = max(self.total - completed, 0)
        eta = remaining / max(rate, 1.0e-8)
        fraction = min(completed / self.total, 1.0)
        width = 24
        filled = min(int(fraction * width), width)
        bar = "=" * filled + (">" if filled < width else "")
        bar = bar.ljust(width, ".")
        message = (
            f"{self.label} [{bar}] {completed}/{self.total} "
            f"({100.0 * fraction:5.1f}%) | {samples} samples | "
            f"{rate:.2f} batches/s | ETA {eta / 60.0:.1f}m"
        )
        if loss is not None:
            message += f" | loss {loss:.5f}"
        if data_seconds is not None and step_seconds is not None:
            message += (
                f" | data {data_seconds / completed:.2f}s/batch"
                f" | compute {step_seconds / completed:.2f}s/batch"
            )
        if completed != self.total:
            print("\r" + message, end="", flush=True)
        else:
            print("\r" + message, flush=True)


def _run_epoch(
    pipeline,
    loader,
    device,
    *,
    optimizer=None,
    scaler=None,
    grad_clip=0.0,
    use_amp=False,
    amp_dtype=torch.bfloat16,
    progress_label="train",
    residual_regularization_weight=None,
):
    training = optimizer is not None
    pipeline.train(training)
    totals: dict[str, float] = {}
    samples = optimizer_steps = 0
    progress = _BatchProgress(progress_label, len(loader))
    total_data_seconds = total_step_seconds = 0.0
    previous_batch_finished = time.perf_counter()
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch_received = time.perf_counter()
        total_data_seconds += batch_received - previous_batch_finished
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = pipeline(
                    **batch,
                    return_diagnostics=False,
                    residual_regularization_weight=(
                        residual_regularization_weight if training else 0.0
                    ),
                )
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
            "residual_regularization_weight": float(
                output["residual_regularization_weight"].detach()
            ),
            "weighted_residual_regularization_loss": float(
                output["weighted_residual_regularization_loss"].detach()
            ),
            "coarse_exact_reconstruction_loss": float(
                output["coarse_exact_reconstruction_loss"].detach()
            ),
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value * count
        samples += count
        batch_finished = time.perf_counter()
        total_step_seconds += batch_finished - batch_received
        progress.update(
            batch_index,
            samples=samples,
            loss=values["loss"],
            data_seconds=total_data_seconds,
            step_seconds=total_step_seconds,
        )
        previous_batch_finished = batch_finished
    result = {key: value / max(samples, 1) for key, value in totals.items()}
    if training:
        result["optimizer_steps"] = optimizer_steps
    return result


def _residual_regularization_weight_for_epoch(
    config: FineDiffusionConfig, epoch: int
) -> float:
    """Linearly remove residual restraint after N completed epochs."""

    base = float(config.lambda_residual_regularization)
    decay_epochs = int(config.residual_regularization_decay_epochs)
    if decay_epochs == 0:
        return base
    completed_epochs = max(int(epoch) - 1, 0)
    fraction = max(0.0, 1.0 - completed_epochs / float(decay_epochs))
    return base * fraction


@torch.inference_mode()
def _run_sampled_validation(
    pipeline,
    loader,
    device,
    *,
    sampling_steps,
    use_amp=False,
    amp_dtype=torch.bfloat16,
    seed=0,
    progress_label="validation",
):
    """Run the deployment sampler; clean LiDAR is used only after prediction."""

    pipeline.eval()
    counts = {
        "fine_tp": 0.0,
        "fine_fp": 0.0,
        "fine_fn": 0.0,
        "fine_tolerant_matched_predictions": 0.0,
        "fine_tolerant_matched_targets": 0.0,
        "fine_prediction_count": 0.0,
        "coarse_tp": 0.0,
        "coarse_fp": 0.0,
        "coarse_fn": 0.0,
        "coarse_tolerant_matched_predictions": 0.0,
        "coarse_tolerant_matched_targets": 0.0,
        "coarse_prediction_count": 0.0,
        "target_count": 0.0,
    }
    samples = 0.0
    generator = torch.Generator(device=device).manual_seed(seed)
    offsets = torch.arange(-2, 3, device=device, dtype=torch.float32)
    rows, columns = torch.meshgrid(offsets, offsets, indexing="ij")
    tolerance_kernel = (
        torch.sqrt(rows.square() + columns.square()) * 0.2 <= 0.5 + 1.0e-6
    ).to(dtype=torch.float32)[None, None]

    def dilate(values):
        return F.conv2d(
            values.to(dtype=torch.float32), tolerance_kernel, padding=2
        ) > 0

    progress = _BatchProgress(progress_label, len(loader))
    total_data_seconds = total_step_seconds = 0.0
    previous_batch_finished = time.perf_counter()
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch_received = time.perf_counter()
        total_data_seconds += batch_received - previous_batch_finished
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
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
        expected_selected = expected & selected
        target_neighborhood = dilate(expected_selected)
        counts["target_count"] += float(expected_selected.sum())
        for prefix, predicted in (("coarse", coarse), ("fine", fine)):
            predicted_selected = predicted & selected
            counts[f"{prefix}_tp"] += float(
                (predicted & expected & selected).sum()
            )
            counts[f"{prefix}_fp"] += float(
                (predicted & ~expected & selected).sum()
            )
            counts[f"{prefix}_fn"] += float(
                (~predicted & expected & selected).sum()
            )
            prediction_neighborhood = dilate(predicted_selected)
            counts[f"{prefix}_tolerant_matched_predictions"] += float(
                (predicted_selected & target_neighborhood).sum()
            )
            counts[f"{prefix}_tolerant_matched_targets"] += float(
                (expected_selected & prediction_neighborhood).sum()
            )
            counts[f"{prefix}_prediction_count"] += float(
                predicted_selected.sum()
            )
        samples += batch["clean_lidar_bev"].shape[0]
        batch_finished = time.perf_counter()
        total_step_seconds += batch_finished - batch_received
        progress.update(
            batch_index,
            samples=int(samples),
            data_seconds=total_data_seconds,
            step_seconds=total_step_seconds,
        )
        previous_batch_finished = batch_finished

    epsilon = 1.0e-8
    result = {"samples": samples}
    for prefix in ("coarse", "fine"):
        tp = counts[f"{prefix}_tp"]
        fp = counts[f"{prefix}_fp"]
        fn = counts[f"{prefix}_fn"]
        result[f"{prefix}_exact_occupancy_iou"] = tp / (
            tp + fp + fn + epsilon
        )
        result[f"{prefix}_exact_occupancy_f1"] = 2.0 * tp / (
            2.0 * tp + fp + fn + epsilon
        )
        tolerant_precision = counts[
            f"{prefix}_tolerant_matched_predictions"
        ] / (counts[f"{prefix}_prediction_count"] + epsilon)
        tolerant_recall = counts[f"{prefix}_tolerant_matched_targets"] / (
            counts["target_count"] + epsilon
        )
        tolerant_f1 = (
            2.0
            * tolerant_precision
            * tolerant_recall
            / (tolerant_precision + tolerant_recall + epsilon)
        )
        result[f"{prefix}_tolerant_0_5m_f1"] = tolerant_f1
        result[f"{prefix}_tolerant_0_5m_iou"] = tolerant_f1 / (
            2.0 - tolerant_f1 + epsilon
        )
    result["fine_minus_coarse_exact_iou"] = (
        result["fine_exact_occupancy_iou"]
        - result["coarse_exact_occupancy_iou"]
    )
    result["fine_minus_coarse_exact_f1"] = (
        result["fine_exact_occupancy_f1"]
        - result["coarse_exact_occupancy_f1"]
    )
    result["fine_minus_coarse_tolerant_0_5m_iou"] = (
        result["fine_tolerant_0_5m_iou"]
        - result["coarse_tolerant_0_5m_iou"]
    )
    result["fine_minus_coarse_tolerant_0_5m_f1"] = (
        result["fine_tolerant_0_5m_f1"]
        - result["coarse_tolerant_0_5m_f1"]
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
    payload, config, bev_normalizer = _load_components(args.config)
    selector_payload = payload
    if args.selector_config:
        with Path(args.selector_config).open("r", encoding="utf-8") as handle:
            selector_payload = json.load(handle)
    selector = build_selector_config(selector_payload)
    training = dict(payload.get("training", {}))
    epochs = args.epochs or int(training.get("epochs", 50))
    batch_size = args.batch_size or int(training.get("batch_size", 4))
    validation_batch_size = args.validation_batch_size or int(
        training.get("validation_batch_size", batch_size)
    )
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training.get("num_workers", 4))
    )
    seed = int(training.get("seed", 42))
    validation_interval = max(
        1, int(training.get("sampled_validation_interval_epochs", 1))
    )
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
        if config.use_pointpillars_conditioning:
            if not use_pointpillars:
                raise ValueError(
                    "Configured fine PointPillars conditioning requires a "
                    "PointPillars coarse checkpoint"
                )
            if coarse.config.lidar_channels != config.lidar_pillar_channels:
                raise ValueError("Fine/coarse LiDAR PointPillars channels differ")
            if coarse.config.radar_channels != config.radar_pillar_channels:
                raise ValueError("Fine/coarse radar PointPillars channels differ")
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
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "collate_fn": coarse_reconstruction_collate,
    }
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        **loader_options,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=False)
        if args.resume
        else None
    )
    if resume_checkpoint is not None:
        validate_fine_diffusion_checkpoint_compatibility(
            resume_checkpoint, config
        )
        residual_metadata = resume_checkpoint.get("residual_normalization")
        if residual_metadata is None:
            raise ValueError(
                "Fine Diffusion checkpoint lacks residual normalization. "
                "Start a fresh run so train-only statistics can be estimated."
            )
        residual_statistics = resume_checkpoint.get("residual_statistics")
        residual_normalizer = _residual_normalizer(
            residual_metadata, config
        )
        bev_metadata = resume_checkpoint.get("bev_normalization")
        if bev_metadata is not None:
            bev_normalizer = BEVChannelNormalization(
                means=bev_metadata["means"],
                stds=bev_metadata["stds"],
                epsilon=float(bev_metadata["epsilon"]),
                source=bev_metadata.get("source", "fine_diffusion_checkpoint"),
            )
    else:
        residual_statistics = None
        if args.residual_statistics_only:
            statistics_loader_options = dict(loader_options)
            statistics_loader_options["persistent_workers"] = False
            statistics_loader = DataLoader(
                train_dataset, shuffle=False, **statistics_loader_options
            )

            def coarse_for_statistics(batch):
                if config.bypass_coarse_reconstruction:
                    return batch["faulty_lidar_bev"] * (
                        1.0 - batch["reconstruction_mask"]
                    )
                coarse.eval()
                output = coarse(
                    batch["faulty_lidar_bev"],
                    batch["radar_bev"],
                    batch["reconstruction_mask"],
                    batch["healthy_context_mask"],
                    batch["halo_mask"],
                    faulty_lidar_points=batch.get("faulty_lidar_points"),
                    radar_points=batch.get("radar_points"),
                )
                return output["coarse_lidar_bev"]

            print("Estimating coarse-to-clean residual statistics from TRAIN only...")
            residual_statistics = estimate_training_residual_statistics(
                statistics_loader,
                move_batch=lambda raw: _move_batch(raw, device),
                coarse_forward=coarse_for_statistics,
                channels=config.lidar_channels,
                minimum_std=config.minimum_residual_std,
            )
            residual_metadata = {
                "raw_channel_stds": residual_statistics["raw_channel_stds"],
                "channel_stds": residual_statistics["effective_channel_stds"],
                "minimum_std": config.minimum_residual_std,
                "source": "training_split_coarse_to_clean_residuals",
            }
        else:
            unit_stds = [1.0] * config.lidar_channels
            residual_metadata = {
                "raw_channel_stds": unit_stds,
                "channel_stds": unit_stds,
                "minimum_std": config.minimum_residual_std,
                "source": "fixed_unit_residual_scaling",
            }
            print(
                "Using fixed unit residual scaling; "
                "skipping the train-set residual-statistics pass."
            )
        residual_normalizer = _residual_normalizer(
            residual_metadata, config
        )
        if residual_statistics is not None:
            for item in residual_statistics["channels"]:
                print(
                    f"  channel {item['channel']}: mean={item['mean']:.8f}, "
                    f"raw_std={item['raw_std']:.8f}, "
                    f"effective_std={item['effective_std']:.8f}, "
                    f"mean_abs={item['mean_absolute_value']:.8f}, "
                    f"p95_abs={item['p95_absolute_value']:.8f}, "
                    f"zero={item['fraction_approximately_zero']:.2%}, "
                    f"n={item['sample_count']}"
                )

    if residual_statistics is not None:
        atomic_write_json(
            output_root / "residual_statistics.json", residual_statistics
        )
    if args.residual_statistics_only:
        print("Residual-statistics-only pass complete; training was not started.")
        return

    diffusion = FineDiffusionRefiner(
        config, bev_normalizer, residual_normalizer
    ).to(device)
    pipeline = FrozenCoarseFineDiffusionPipeline(coarse, diffusion).to(device)
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=float(training.get("learning_rate", 2.0e-4)),
        weight_decay=float(training.get("weight_decay", 1.0e-3)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(training.get("amp_dtype", "bfloat16"))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )
    start_epoch, best, history = 1, float("-inf"), []
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        diffusion.load_state_dict(checkpoint["diffusion_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler.is_enabled():
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
            "normalization": bev_normalizer.metadata(),
            "residual_normalization": residual_normalizer.metadata(),
            "fine_diffusion_architecture": fine_diffusion_architecture_metadata(
                config
            ),
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
        f"Fine PointPillars conditioning: {config.use_pointpillars_conditioning}; "
        f"bypass coarse: {config.bypass_coarse_reconstruction}; "
        f"AMP: {str(amp_dtype).removeprefix('torch.') if use_amp else 'off'}; "
        f"sampled validation every {validation_interval} epoch(s)"
    )
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        residual_regularization_weight = (
            _residual_regularization_weight_for_epoch(config, epoch)
        )
        train_stats = _run_epoch(
            pipeline,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=float(training.get("grad_clip", 1.0)),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            progress_label=f"epoch {epoch:03d}/{epochs:03d} train",
            residual_regularization_weight=residual_regularization_weight,
        )
        run_validation = epoch % validation_interval == 0 or epoch == epochs
        val_stats = None
        validation_seconds = 0.0
        if run_validation:
            validation_started = time.perf_counter()
            val_stats = _run_sampled_validation(
                pipeline,
                val_loader,
                device,
                sampling_steps=config.sampling_steps,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                seed=seed + 10_000,
                progress_label=f"epoch {epoch:03d}/{epochs:03d} val",
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
        if val_stats is not None:
            row.update({f"val/{key}": value for key, value in val_stats.items()})
        history.append(row)
        baseline_label = (
            "erased faulty"
            if config.bypass_coarse_reconstruction
            else "coarse"
        )
        print(
            f"\nepoch {epoch:03d}/{epochs:03d} | {elapsed:.1f}s\n"
            f"  train | loss {train_stats['loss']:.6f} | "
            f"diffusion {train_stats['diffusion_loss']:.6f} | "
            f"reconstruction {train_stats['exact_reconstruction_loss']:.6f} | "
            f"degradation {train_stats['degradation_loss']:.6f} | "
            f"residual {train_stats['residual_regularization_loss']:.6f} "
            f"(weight {train_stats['residual_regularization_weight']:.5f}, "
            f"weighted "
            f"{train_stats['weighted_residual_regularization_loss']:.6f})"
        )
        if val_stats is not None:
            print(
                f"  sampled validation | {validation_seconds:.1f}s | "
                f"{int(val_stats['samples'])} samples\n"
                f"    exact IoU/F1     | {baseline_label} "
                f"{100.0 * val_stats['coarse_exact_occupancy_iou']:.2f}% "
                f"/ {100.0 * val_stats['coarse_exact_occupancy_f1']:.2f}% "
                f"-> fine {100.0 * val_stats['fine_exact_occupancy_iou']:.2f}% "
                f"/ {100.0 * val_stats['fine_exact_occupancy_f1']:.2f}%\n"
                f"    exact improvement| IoU "
                f"{100.0 * val_stats['fine_minus_coarse_exact_iou']:+.2f} pp | "
                f"F1 {100.0 * val_stats['fine_minus_coarse_exact_f1']:+.2f} pp\n"
                f"    0.5m IoU/F1      | {baseline_label} "
                f"{100.0 * val_stats['coarse_tolerant_0_5m_iou']:.2f}% "
                f"/ {100.0 * val_stats['coarse_tolerant_0_5m_f1']:.2f}% "
                f"-> fine {100.0 * val_stats['fine_tolerant_0_5m_iou']:.2f}% "
                f"/ {100.0 * val_stats['fine_tolerant_0_5m_f1']:.2f}%\n"
                f"    0.5m improvement | IoU "
                f"{100.0 * val_stats['fine_minus_coarse_tolerant_0_5m_iou']:+.2f} pp | "
                f"F1 {100.0 * val_stats['fine_minus_coarse_tolerant_0_5m_f1']:+.2f} pp"
            )
            score = val_stats["fine_minus_coarse_exact_iou"]
            improved = score > best
            best = max(best, score)
        else:
            print(
                f"  sampled validation | skipped; next at epoch "
                f"{min(((epoch // validation_interval) + 1) * validation_interval, epochs)}"
            )
            improved = False
        checkpoint = {
            "epoch": epoch,
            "diffusion_state_dict": diffusion.state_dict(),
            "diffusion_config": config.to_dict(),
            "fine_diffusion_architecture": fine_diffusion_architecture_metadata(
                config
            ),
            "residual_normalization": residual_normalizer.metadata(),
            "bev_normalization": bev_normalizer.metadata(),
            "residual_statistics": residual_statistics,
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
    maximum_sampling_samples = int(
        payload.get("evaluation", {}).get("max_sampling_samples", 0)
    )
    if maximum_sampling_samples > 0:
        rows = _sampling_metrics(
            pipeline,
            val_loader,
            device,
            maximum_sampling_samples,
            config.sampling_steps,
        )
        write_csv_rows(output_root / "sampling_metrics.csv", rows)
        atomic_write_json(output_root / "sampling_metrics.json", rows)


if __name__ == "__main__":
    main()
