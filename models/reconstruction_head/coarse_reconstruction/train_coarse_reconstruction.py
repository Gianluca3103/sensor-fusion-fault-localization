"""Train deterministic direct-BEV coarse LiDAR reconstruction independently."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict
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

from Fault_Localization_Model.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    write_csv_rows,
)
from PFS.training_utils import _split_paths, resolve_device, seed_everything
from models.reconstruction_head import (
    CoarseReconstructionDataset,
    CoarseReconstructionModel,
    MaskedBEVReconstructionLoss,
    build_configs,
    coarse_reconstruction_metrics,
    load_config,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "coarse_reconstruction.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument(
        "--disable-radar",
        action="store_true",
        help=(
            "Run a radar-input ablation by replacing radar BEVs with zeros "
            "during both training and validation."
        ),
    )
    parser.add_argument(
        "--radar-mode",
        choices=("full", "global-only", "none"),
        default="full",
        help=(
            "Radar conditioning mode: full uses local and global radar, "
            "global-only removes radar from the local U-Net, and none removes "
            "radar measurements from both branches."
        ),
    )
    parser.add_argument(
        "--disable-global-map",
        action="store_true",
        help=(
            "Use only the local U-Net by skipping both global encoders, "
            "global fusion, cross-attention, and bottleneck fusion."
        ),
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write live training metrics for TensorBoard.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        help="TensorBoard directory (default: <output-root>/tensorboard).",
    )
    return parser.parse_args()


def _create_tensorboard_writer(enabled: bool, log_dir: Path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard tracking requires the 'tensorboard' package. "
            "Install the project requirements or run: "
            "python -m pip install tensorboard"
        ) from exc
    writer = SummaryWriter(log_dir=str(log_dir))
    atexit.register(writer.close)
    return writer


def _write_tensorboard_epoch(writer, row: dict, optimizer, epoch: int) -> None:
    if writer is None:
        return
    for name, value in row.items():
        if name == "epoch" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(name, value, epoch)
    for index, group in enumerate(optimizer.param_groups):
        writer.add_scalar(
            f"optimizer/learning_rate_group_{index}",
            float(group["lr"]),
            epoch,
        )
    writer.flush()


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
    if "observability_confidence" in batch:
        moved["observability_confidence"] = batch[
            "observability_confidence"
        ].to(device, non_blocking=True)
    mask_dtype = moved["faulty_bev"].dtype
    for key in ("reconstruction_mask", "healthy_context_mask", "halo_mask"):
        moved[key] = moved[key].to(dtype=mask_dtype)
    return moved


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape_log(inputs: dict, outputs: dict) -> dict:
    names = (
        "erased_lidar_bev",
        "replacement_raw",
        "replacement_bev",
        "occupancy_logits",
        "predicted_density",
        "predicted_height",
        "coarse_lidar_bev",
        "reconstruction_mask",
        "healthy_context_mask",
        "halo_mask",
        "local_input",
        "global_lidar_input",
        "local_bottleneck",
        "query_tokens",
        "context_tokens",
        "attention_context",
        "fused_bottleneck",
        "global_context_map",
    )
    result = {key: list(value.shape) for key, value in inputs.items()}
    result.update(
        {key: list(outputs[key].shape) for key in names if key in outputs}
    )
    return result


def _summarize_active_fractions(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty active-fraction collection")
    fractions = torch.tensor(values, dtype=torch.float64)
    return {
        "median": float(torch.quantile(fractions, 0.5)),
        "p90": float(torch.quantile(fractions, 0.9)),
        "maximum": float(fractions.max()),
    }


def _active_fraction_recommendation(summary: dict[str, float]) -> str:
    if summary["median"] >= 0.25:
        return "Keep the dense U-Net: typical active coverage is at least 25%."
    if summary["p90"] <= 0.15:
        return (
            "Test cropped dense processing first: at least 90% of samples use "
            "no more than 15% of the BEV. Consider a sparse U-Net only if "
            "cropping remains too expensive."
        )
    return (
        "Keep the dense U-Net for now and profile cropped dense processing: "
        "active coverage is neither consistently sparse nor typically above 25%."
    )


def _save_conditioning(
    output_root: Path,
    epoch: int,
    batch: dict,
    outputs: dict,
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return
    destination = output_root / "conditioning" / f"epoch_{epoch:03d}"
    keys = (
        "coarse_lidar_bev",
        "replacement_raw",
        "replacement_bev",
        "occupancy_logits",
        "reconstruction_mask",
    )
    count = min(max_samples, outputs["coarse_lidar_bev"].shape[0])
    for index in range(count):
        payload = {key: outputs[key][index].detach().cpu() for key in keys}
        if "attention_weights" in outputs:
            payload["attention_weights"] = outputs["attention_weights"][
                index
            ].detach().cpu()
        payload["sample_path"] = batch["sample_path"][index]
        atomic_torch_save(payload, destination / f"sample_{index:03d}.pt")


def _run_epoch(
    model,
    loader,
    loss_fn,
    device,
    *,
    optimizer=None,
    scaler=None,
    grad_clip=0.0,
    use_amp=False,
    return_attention=False,
    conditioning_callback=None,
    active_fraction_samples=None,
    radar_mode="full",
    use_global_map=True,
):
    training = optimizer is not None
    model.train(training)
    sums = {}
    samples = 0
    logged_shapes = None
    for batch_index, batch in enumerate(loader):
        if active_fraction_samples is not None:
            active_mask = torch.maximum(
                batch["reconstruction_mask"], batch["halo_mask"]
            )
            fractions = active_mask.flatten(1).float().mean(dim=1)
            active_fraction_samples.extend(fractions.tolist())
        inputs = _move_batch(batch, device)
        local_radar_bev = None
        if radar_mode == "none":
            inputs["radar_bev"] = torch.zeros_like(inputs["radar_bev"])
        elif radar_mode == "global-only":
            local_radar_bev = torch.zeros_like(inputs["radar_bev"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
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
                    local_radar_bev=local_radar_bev,
                    use_global_map=use_global_map,
                    return_attention_weights=return_attention and batch_index == 0,
                )
                losses = loss_fn(
                    outputs,
                    inputs["clean_bev"],
                    inputs.get("observability_confidence"),
                )
            if training:
                scaler.scale(losses["loss"]).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        with torch.no_grad():
            metrics = coarse_reconstruction_metrics(
                outputs,
                inputs["faulty_bev"],
                inputs["clean_bev"],
                loss_fn.config.epsilon,
                inputs.get("observability_confidence"),
                include_tolerant=not training,
            )
        batch_size = inputs["faulty_bev"].shape[0]
        samples += batch_size
        values = {**losses, **metrics}
        for key, value in values.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
        if logged_shapes is None:
            logged_shapes = _shape_log(inputs, outputs)
        if conditioning_callback is not None and batch_index == 0:
            conditioning_callback(batch, outputs)
    return {key: value / max(samples, 1) for key, value in sums.items()}, logged_shapes


def main():
    args = _parse_args()
    if args.disable_radar and args.radar_mode != "full":
        raise ValueError("Use either --disable-radar or --radar-mode, not both")
    radar_mode = "none" if args.disable_radar else args.radar_mode
    payload = load_config(args.config)
    model_config, loss_config, selector_config = build_configs(payload)
    training = dict(payload.get("training", {}))
    epochs = args.epochs or int(training.get("epochs", 50))
    batch_size = args.batch_size or int(training.get("batch_size", 8))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training.get("num_workers", 4))
    )
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = Path(
        args.tensorboard_log_dir or output_root / "tensorboard"
    )
    tensorboard_writer = _create_tensorboard_writer(
        args.tensorboard,
        tensorboard_log_dir,
    )
    data_root = Path(args.data_root)
    radar_root = Path(args.radar_root)
    train_paths = _split_paths(
        data_root,
        "train",
        args.limit_train_samples,
        seed,
    )
    val_paths = _split_paths(
        data_root,
        "val",
        args.limit_val_samples,
        seed,
    )
    dataset_options = {
        "radar_root": radar_root,
        "data_root": data_root,
        "selector_config": selector_config,
    }
    train_dataset = CoarseReconstructionDataset(train_paths, **dataset_options)
    val_dataset = CoarseReconstructionDataset(val_paths, **dataset_options)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model = CoarseReconstructionModel(model_config).to(device)
    loss_fn = MaskedBEVReconstructionLoss(loss_config)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 2.0e-4)),
        weight_decay=float(training.get("weight_decay", 1.0e-3)),
    )
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(training.get("grad_clip", 1.0))
    save_conditioning_samples = int(training.get("save_conditioning_samples", 4))
    return_attention = bool(
        payload["coarse_reconstruction"]["global_context"].get(
            "return_attention_during_validation", True
        )
    )
    atomic_write_json(
        output_root / "resolved_config.json",
        {
            "model": model_config.to_dict(),
            "loss": asdict(loss_config),
            "selector": selector_config.__dict__,
            "training": training,
            "args": vars(args),
        },
    )
    print(f"Training samples: {len(train_dataset)}; validation: {len(val_dataset)}")
    print(f"Device: {device}; AMP: {use_amp}")
    print(f"Radar mode: {radar_mode}")
    print(f"Global map enabled: {not args.disable_global_map}")
    history = []
    best_validation = float("inf")
    active_fraction_profile = None
    for epoch in range(1, epochs + 1):
        train_active_fractions = [] if epoch == 1 else None
        val_active_fractions = [] if epoch == 1 else None
        _synchronize_device(device)
        train_started = time.perf_counter()
        train_stats, train_shapes = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            use_amp=use_amp,
            active_fraction_samples=train_active_fractions,
            radar_mode=radar_mode,
            use_global_map=not args.disable_global_map,
        )
        _synchronize_device(device)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        with torch.no_grad():
            val_stats, val_shapes = _run_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                use_amp=use_amp,
                return_attention=return_attention,
                conditioning_callback=lambda batch, outputs: _save_conditioning(
                    output_root,
                    epoch,
                    batch,
                    outputs,
                    save_conditioning_samples,
                ),
                active_fraction_samples=val_active_fractions,
                radar_mode=radar_mode,
                use_global_map=not args.disable_global_map,
            )
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = train_seconds + validation_seconds
        if epoch == 1:
            train_active_summary = _summarize_active_fractions(
                train_active_fractions
            )
            val_active_summary = _summarize_active_fractions(val_active_fractions)
            combined_active_summary = _summarize_active_fractions(
                train_active_fractions + val_active_fractions
            )
            active_fraction_profile = {
                "definition": (
                    "Per-sample mean of reconstruction_mask OR halo_mask over "
                    "the complete BEV grid"
                ),
                "train": train_active_summary,
                "validation": val_active_summary,
                "combined": combined_active_summary,
                "recommendation": _active_fraction_recommendation(
                    combined_active_summary
                ),
            }
            atomic_write_json(
                output_root / "active_fraction_profile.json",
                active_fraction_profile,
            )
            atomic_write_json(
                output_root / "debug_batch_shapes.json",
                {"train": train_shapes, "validation": val_shapes},
            )
            print("Debug tensor shapes:", json.dumps(val_shapes, indent=2))
            print(
                "Active-mask coverage: "
                f"median={combined_active_summary['median']:.2%}, "
                f"p90={combined_active_summary['p90']:.2%}, "
                f"max={combined_active_summary['maximum']:.2%}"
            )
            print(
                "Architecture recommendation: "
                f"{active_fraction_profile['recommendation']}"
            )
        row = {
            "epoch": epoch,
            "runtime/train_seconds": train_seconds,
            "runtime/validation_seconds": validation_seconds,
            "runtime/epoch_seconds": epoch_seconds,
        }
        row.update({f"train/{key}": value for key, value in train_stats.items()})
        row.update({f"val/{key}": value for key, value in val_stats.items()})
        if epoch == 1:
            for split, summary in (
                ("train", train_active_summary),
                ("val", val_active_summary),
                ("combined", combined_active_summary),
            ):
                row.update(
                    {
                        f"active_fraction/{split}_{key}": value
                        for key, value in summary.items()
                    }
                )
        history.append(row)
        _write_tensorboard_epoch(
            tensorboard_writer,
            row,
            optimizer,
            epoch,
        )
        observability_log = ""
        if loss_config.observability_weighting.enabled:
            observability_log = (
                "val/empty_observability="
                f"{val_stats['mean_empty_observability_repair']:.3f} "
                "val/empty_weight="
                f"{val_stats['mean_empty_occupancy_weight']:.3f} "
                "val/high_obs_hallucination="
                f"{val_stats['hallucination_rate_high_observability']:.3%} "
            )
        print(
            f"epoch {epoch:03d}: train/loss={train_stats['loss']:.6f} "
            f"train/occupancy={train_stats['loss_occupancy']:.6f} "
            f"train/density={train_stats['loss_density']:.6f} "
            f"train/height={train_stats['loss_height']:.6f} "
            "train/exact_F1="
            f"{train_stats['coarse_occupancy_exact_f1']:.3%} "
            "train/exact_IoU="
            f"{train_stats['coarse_occupancy_exact_iou']:.3%} "
            f"val/loss={val_stats['loss']:.6f} "
            f"val/occupancy={val_stats['loss_occupancy']:.6f} "
            f"val/density={val_stats['loss_density']:.6f} "
            f"val/height={val_stats['loss_height']:.6f} "
            f"val/exact_F1={val_stats['coarse_occupancy_exact_f1']:.3%} "
            f"val/exact_IoU={val_stats['coarse_occupancy_exact_iou']:.3%} "
            "val/F1@0.5m="
            f"{val_stats['coarse_occupancy_tolerant_0_5m_f1']:.3%} "
            "val/IoU@0.5m="
            f"{val_stats['coarse_occupancy_tolerant_0_5m_iou']:.3%} "
            f"val/hallucination="
            f"{val_stats['coarse_occupancy_hallucination_rate']:.3%} "
            f"val/height_mae={val_stats['coarse_height_mae_m']:.3f}m "
            f"{observability_log}"
            f"outside_change={val_stats['outside_mask_max_change']:.3e} "
            f"train_time={train_seconds:.1f}s "
            f"val_time={validation_seconds:.1f}s "
            f"epoch_time={epoch_seconds:.1f}s",
            flush=True,
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": model_config.to_dict(),
            "loss_config": asdict(loss_config),
            "active_fraction_profile": active_fraction_profile,
            "radar_mode": radar_mode,
            "radar_disabled": radar_mode == "none",
            "global_map_enabled": not args.disable_global_map,
            "history": history,
        }
        atomic_torch_save(checkpoint, output_root / "last_checkpoint.pt")
        if val_stats["loss"] < best_validation:
            best_validation = val_stats["loss"]
            atomic_torch_save(checkpoint, output_root / "best_model.pt")
        write_csv_rows(
            output_root / "history.csv",
            history,
            fieldnames=list(history[0]),
        )
    if tensorboard_writer is not None:
        tensorboard_writer.close()


if __name__ == "__main__":
    main()
