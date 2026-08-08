"""Train masked direct-BEV residual diffusion after a frozen coarse model."""

from __future__ import annotations

import argparse
import hashlib
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
from PFS.training_utils import (
    _split_paths,
    capture_rng_state,
    resolve_device,
    restore_rng_state,
    seed_everything,
)
from models.reconstruction_head import (
    BEVChannelNormalization,
    CoarseReconstructionDataset,
    DiffusionProcessConfig,
    FrozenCoarseDiffusionPipeline,
    MaskedResidualDiffusion,
    ResidualDiffusionSampler,
    ResidualDiffusionUNetConfig,
    build_selector_config,
    load_frozen_coarse_model,
    reconstruction_stage_metrics,
    validate_diffusion_checkpoint_compatibility,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--coarse-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "residual_diffusion.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument("--resume")
    return parser.parse_args()


def _load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Residual-diffusion config must decode to a mapping")
    return payload


def _require_contract(section):
    if not section.get("enabled", True):
        raise ValueError("residual_diffusion.enabled must be true")
    if section.get("freeze_coarse_model", True) is not True:
        raise ValueError("residual_diffusion.freeze_coarse_model must be true")
    if section.get("detach_coarse_output", True) is not True:
        raise ValueError("residual_diffusion.detach_coarse_output must be true")
    required_true = (
        "restrict_noise_to_reconstruction_mask",
        "restrict_forward_process_to_reconstruction_mask",
        "restrict_loss_to_reconstruction_mask",
        "restrict_reverse_process_to_reconstruction_mask",
        "restrict_final_residual_to_reconstruction_mask",
    )
    for key in required_true:
        if section.get(key, True) is not True:
            raise ValueError(f"residual_diffusion.{key} must be true")
    conditioning = section.get("conditioning", {})
    for key in ("use_coarse_bev", "use_reconstruction_mask", "use_radar", "use_halo"):
        if conditioning.get(key, True) is not True:
            raise ValueError(f"conditioning.{key} must be true")
    for key in ("use_global_context_map", "use_attention_context"):
        if conditioning.get(key, False) is not False:
            raise ValueError(f"conditioning.{key} must be false in residual diffusion")
    unet = section.get("unet", {})
    if unet.get("input_channels", 11) != 11 or unet.get("output_channels", 3) != 3:
        raise ValueError(
            "Local-radar residual diffusion requires exactly 11 input and 3 output channels"
        )
    if unet.get("normalization", "group_norm") != "group_norm":
        raise ValueError("Only GroupNorm is supported")
    if unet.get("activation", "silu") != "silu":
        raise ValueError("Only SiLU is supported")
    loss = section.get("loss", {})
    if loss.get("type", "masked_epsilon_mse") != "masked_epsilon_mse":
        raise ValueError("Only masked_epsilon_mse is supported")
    if loss.get("snr_weighting", False):
        raise ValueError("SNR weighting is disabled in this first version")
    sampling = section.get("sampling", {})
    if sampling.get("method", "ddpm") != "ddpm":
        raise ValueError("Only DDPM sampling is implemented")
    timesteps = int(section.get("num_train_timesteps", 1000))
    if int(sampling.get("num_inference_steps", timesteps)) != timesteps:
        raise ValueError("DDPM num_inference_steps must equal num_train_timesteps")


def _build_components(payload):
    section = payload.get("residual_diffusion", {})
    _require_contract(section)
    unet = section.get("unet", {})
    loss = section.get("loss", {})
    model_config = ResidualDiffusionUNetConfig(
        lidar_channels=3,
        radar_channels=int(unet.get("radar_channels", 4)),
        base_channels=int(unet.get("base_channels", 32)),
        channel_multipliers=tuple(unet.get("channel_multipliers", (1, 2, 4, 8))),
        residual_blocks_per_level=int(unet.get("residual_blocks_per_level", 2)),
        time_embedding_dim=int(unet.get("time_embedding_dim", 256)),
        dropout=float(unet.get("dropout", 0.0)),
    )
    process_config = DiffusionProcessConfig(
        num_train_timesteps=int(section.get("num_train_timesteps", 1000)),
        noise_schedule=section.get("noise_schedule", "cosine"),
        prediction_type=section.get("prediction_type", "epsilon"),
        denominator_epsilon=float(loss.get("denominator_epsilon", 1e-8)),
    )
    normalization = section.get("normalization", {})
    normalizer = BEVChannelNormalization(
        means=normalization.get("channel_means", (0.0, 0.0, 0.0)),
        stds=normalization.get("channel_stds", (1.0, 1.0, 1.0)),
        epsilon=float(normalization.get("epsilon", 1e-6)),
        source=normalization.get("source", "configured_training_statistics"),
    )
    selector = build_selector_config(payload)
    model_config.validate()
    process_config.validate()
    return model_config, process_config, normalizer, selector


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
    mask_dtype = moved["faulty_lidar_bev"].dtype
    for key in ("reconstruction_mask", "healthy_context_mask", "halo_mask"):
        moved[key] = moved[key].to(dtype=mask_dtype)
    return moved


def _synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape_log(inputs, output):
    keys = (
        "coarse_lidar_bev", "erased_lidar_bev", "residual_gt", "residual_t",
        "epsilon", "epsilon_pred", "diffusion_input", "reconstruction_mask",
        "local_radar", "active_mask",
    )
    result = {key: list(value.shape) for key, value in inputs.items()}
    result.update({key: list(output[key].shape) for key in keys})
    return result


def _run_epoch(pipeline, loader, device, *, optimizer=None, scaler=None, grad_clip=0.0, use_amp=False):
    training = optimizer is not None
    pipeline.train(training)
    total = {"diffusion_loss": 0.0, "residual_target_mean_abs": 0.0, "coarse_masked_mae": 0.0}
    samples = 0
    optimizer_steps = 0
    shapes = None
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16, enabled=use_amp):
                output = pipeline(**batch)
                loss = output["diffusion_loss"]
            if training:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(pipeline.diffusion.unet.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    optimizer_steps += 1
        with torch.no_grad():
            mask = batch["reconstruction_mask"]
            denominator = (3 * mask.sum()).clamp_min(1e-8)
            residual_abs = output["residual_gt"].abs().sum() / denominator
            coarse_mae = (mask * (output["coarse_lidar_bev"] - batch["clean_lidar_bev"])).abs().sum() / denominator
        count = batch["clean_lidar_bev"].shape[0]
        samples += count
        total["diffusion_loss"] += float(loss.detach()) * count
        total["residual_target_mean_abs"] += float(residual_abs) * count
        total["coarse_masked_mae"] += float(coarse_mae) * count
        if shapes is None:
            shapes = _shape_log(batch, output)
        if any(parameter.grad is not None for parameter in pipeline.coarse_model.parameters()):
            raise RuntimeError("Frozen coarse model unexpectedly received gradients")
    stats = {key: value / max(samples, 1) for key, value in total.items()}
    if training:
        stats["optimizer_steps"] = optimizer_steps
    return stats, shapes


def _flatten_metrics(metrics):
    row = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}/{key}" if prefix else key, child)
        elif torch.is_tensor(value) and value.ndim == 1:
            for channel, child in enumerate(value):
                row[f"{prefix}_ch{channel}"] = float(child)
        elif torch.is_tensor(value):
            row[prefix] = float(value)
        elif isinstance(value, (int, float)):
            row[prefix] = float(value)

    visit("", metrics)
    return row


@torch.no_grad()
def _final_sampling_evaluation(pipeline, loader, device, section, output_root):
    sampler = ResidualDiffusionSampler(pipeline.diffusion)
    evaluation = section.get("evaluation", {})
    sampling = section.get("sampling", {})
    maximum = int(evaluation.get("max_sampling_samples", 8))
    seed = int(evaluation.get("seed", 12345))
    generator = torch.Generator(device=device.type).manual_seed(seed)
    rows = []
    sampling_shapes = None
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        coarse, erased = pipeline.coarse_forward(
            batch["faulty_lidar_bev"], batch["radar_bev"], batch["reconstruction_mask"],
            batch["healthy_context_mask"], batch["halo_mask"],
        )
        sampled = sampler.sample(
            coarse,
            batch["radar_bev"],
            batch["reconstruction_mask"],
            batch["halo_mask"],
            faulty_lidar_bev=batch["faulty_lidar_bev"],
            generator=generator,
            save_intermediate_steps=bool(sampling.get("save_intermediate_steps", False)),
            intermediate_stride=int(sampling.get("intermediate_stride", 100)),
        )
        if sampling_shapes is None:
            sampling_shapes = {
                key: list(sampled[key].shape)
                for key in (
                    "coarse_lidar_bev",
                    "residual_pred",
                    "final_lidar_bev",
                    "reconstruction_mask",
                    "local_radar",
                    "active_mask",
                )
            }
        for index in range(coarse.shape[0]):
            one = reconstruction_stage_metrics(
                erased[index:index+1], coarse[index:index+1], sampled["final_lidar_bev"][index:index+1],
                batch["faulty_lidar_bev"][index:index+1], batch["clean_lidar_bev"][index:index+1],
                batch["reconstruction_mask"][index:index+1],
            )
            row = {"sample_path": raw_batch["sample_path"][index]}
            row.update(_flatten_metrics(one))
            rows.append(row)
            if len(rows) >= maximum:
                break
        if len(rows) >= maximum:
            break
    write_csv_rows(output_root / "final_sampling_metrics.csv", rows)
    atomic_write_json(output_root / "final_sampling_metrics.json", rows)
    atomic_write_json(output_root / "final_sampling_shapes.json", sampling_shapes or {})
    return rows


def _file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def main():
    args = _parse_args()
    payload = _load_config(args.config)
    model_config, process_config, normalizer, selector_config = _build_components(payload)
    training = dict(payload.get("training", {}))
    epochs = args.epochs if args.epochs is not None else int(training.get("epochs", 50))
    batch_size = args.batch_size if args.batch_size is not None else int(training.get("batch_size", 4))
    workers = args.num_workers if args.num_workers is not None else int(training.get("num_workers", 4))
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    radar_root = Path(args.radar_root)
    data_root = Path(args.data_root)
    dataset_options = {
        "radar_root": radar_root,
        "data_root": data_root,
        "selector_config": selector_config,
    }
    train_dataset = CoarseReconstructionDataset(
        _split_paths(args.data_root, "train", args.limit_train_samples, seed),
        **dataset_options,
    )
    val_dataset = CoarseReconstructionDataset(
        _split_paths(args.data_root, "val", args.limit_val_samples, seed),
        **dataset_options,
    )
    loader_options = {"batch_size": batch_size, "num_workers": workers, "pin_memory": device.type == "cuda", "persistent_workers": workers > 0}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    coarse, _coarse_checkpoint = load_frozen_coarse_model(args.coarse_checkpoint, device)
    diffusion = MaskedResidualDiffusion(model_config, process_config, normalizer).to(device)
    pipeline = FrozenCoarseDiffusionPipeline(coarse, diffusion).to(device)
    optimizer = torch.optim.AdamW(diffusion.unet.parameters(), lr=float(training.get("learning_rate", 2e-4)), weight_decay=float(training.get("weight_decay", 1e-3)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, history, best = 1, [], float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("coarse_checkpoint_identity") != _file_identity(args.coarse_checkpoint):
            raise ValueError("Resume coarse-checkpoint identity does not match")
        validate_diffusion_checkpoint_compatibility(checkpoint, diffusion)
        diffusion.load_state_dict(checkpoint["diffusion_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint["epoch"]) + 1
        history = list(checkpoint.get("history", []))
        best = float(checkpoint.get("best_validation_loss", best))
    identity = _file_identity(args.coarse_checkpoint)
    atomic_write_json(output_root / "resolved_config.json", {"payload": payload, "model": model_config.to_dict(), "process": process_config.to_dict(), "normalization": normalizer.metadata(), "coarse_checkpoint_identity": identity, "args": vars(args)})
    print(f"Training samples: {len(train_dataset)}; validation: {len(val_dataset)}")
    print(f"Device: {device}; AMP: {use_amp}; coarse model frozen: true")
    for epoch in range(start_epoch, epochs + 1):
        _synchronize_device(device)
        train_started = time.perf_counter()
        train_stats, train_shapes = _run_epoch(pipeline, train_loader, device, optimizer=optimizer, scaler=scaler, grad_clip=float(training.get("grad_clip", 1.0)), use_amp=use_amp)
        _synchronize_device(device)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        with torch.no_grad():
            val_stats, val_shapes = _run_epoch(pipeline, val_loader, device, use_amp=use_amp)
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = train_seconds + validation_seconds
        if train_stats["optimizer_steps"] > 0:
            scheduler.step()
        if epoch == start_epoch:
            atomic_write_json(output_root / "debug_batch_shapes.json", {"train": train_shapes, "validation": val_shapes})
            print("Debug tensor shapes:", json.dumps(val_shapes, indent=2))
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "runtime/train_seconds": train_seconds,
            "runtime/validation_seconds": validation_seconds,
            "runtime/epoch_seconds": epoch_seconds,
        }
        row.update({f"train/{key}": value for key, value in train_stats.items()})
        row.update({f"val/{key}": value for key, value in val_stats.items()})
        history.append(row)
        print(
            f"epoch {epoch:03d}: "
            f"train/diffusion_loss={train_stats['diffusion_loss']:.6f} "
            f"val/diffusion_loss={val_stats['diffusion_loss']:.6f} "
            f"train/residual_target_mean_abs={train_stats['residual_target_mean_abs']:.6f} "
            f"train/coarse_masked_mae={train_stats['coarse_masked_mae']:.6f} "
            f"train_time={train_seconds:.1f}s "
            f"val_time={validation_seconds:.1f}s "
            f"epoch_time={epoch_seconds:.1f}s"
        )
        improved = val_stats["diffusion_loss"] < best
        if improved:
            best = val_stats["diffusion_loss"]
        checkpoint = {
            "epoch": epoch, "diffusion_state_dict": diffusion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "diffusion_config": {"unet": model_config.to_dict(), "process": process_config.to_dict()},
            "coarse_checkpoint_identity": identity, "bev_normalization": normalizer.metadata(),
            "history": history, "best_validation_loss": best, "rng_state": capture_rng_state(),
        }
        atomic_torch_save(checkpoint, output_root / "latest_checkpoint.pt")
        if improved:
            atomic_torch_save(checkpoint, output_root / "best_validation_loss.pt")
        history_fields = sorted({key for history_row in history for key in history_row})
        write_csv_rows(output_root / "history.csv", history, fieldnames=history_fields)
    rows = _final_sampling_evaluation(pipeline, val_loader, device, payload["residual_diffusion"], output_root)
    print(f"Final DDPM sampling metrics written for {len(rows)} validation samples")


if __name__ == "__main__":
    main()
