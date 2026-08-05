"""Train deterministic direct-BEV coarse LiDAR reconstruction independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    write_csv_rows,
)
from PFS.training_utils import resolve_device, seed_everything
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
    return parser.parse_args()


def _split_paths(data_root: Path, split: str, limit: int | None) -> list[Path]:
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Required dataset split is missing: {split_root}")
    paths = sorted(split_root.rglob("*.npz"))
    if limit is not None:
        if limit < 1:
            raise ValueError("Split sample limits must be positive")
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No NPZ samples found under {split_root}")
    return paths


def _move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "faulty_bev",
        "radar_bev",
        "reconstruction_mask",
        "healthy_context_mask",
        "halo_mask",
        "clean_bev",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape_log(inputs: dict, outputs: dict) -> dict:
    names = (
        "erased_lidar_bev",
        "replacement_raw",
        "coarse_lidar_bev",
        "reconstruction_mask",
        "healthy_context_mask",
        "halo_mask",
        "local_input",
        "local_bottleneck",
        "query_tokens",
        "context_tokens",
        "attention_context",
        "fused_bottleneck",
        "global_context_map",
    )
    result = {key: list(value.shape) for key, value in inputs.items()}
    result.update({key: list(outputs[key].shape) for key in names})
    return result


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
):
    training = optimizer is not None
    model.train(training)
    sums = {}
    samples = 0
    logged_shapes = None
    for batch_index, batch in enumerate(loader):
        inputs = _move_batch(batch, device)
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
                    return_attention_weights=return_attention,
                )
                losses = loss_fn(outputs, inputs["clean_bev"])
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
    data_root = Path(args.data_root)
    train_paths = _split_paths(data_root, "train", args.limit_train_samples)
    val_paths = _split_paths(data_root, "val", args.limit_val_samples)
    dataset_options = {
        "radar_root": Path(args.radar_root),
        "resize_hw": (320, 320),
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
            "loss": loss_config.__dict__,
            "selector": selector_config.__dict__,
            "training": training,
            "args": vars(args),
        },
    )
    print(f"Training samples: {len(train_dataset)}; validation: {len(val_dataset)}")
    print(f"Device: {device}; AMP: {use_amp}")
    history = []
    best_validation = float("inf")
    for epoch in range(1, epochs + 1):
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
            )
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = train_seconds + validation_seconds
        if epoch == 1:
            atomic_write_json(
                output_root / "debug_batch_shapes.json",
                {"train": train_shapes, "validation": val_shapes},
            )
            print("Debug tensor shapes:", json.dumps(val_shapes, indent=2))
        row = {
            "epoch": epoch,
            "runtime/train_seconds": train_seconds,
            "runtime/validation_seconds": validation_seconds,
            "runtime/epoch_seconds": epoch_seconds,
        }
        row.update({f"train/{key}": value for key, value in train_stats.items()})
        row.update({f"val/{key}": value for key, value in val_stats.items()})
        history.append(row)
        print(
            f"epoch {epoch:03d}: train/reconstruction_loss="
            f"{train_stats['reconstruction_loss']:.6f} "
            f"val/reconstruction_loss={val_stats['reconstruction_loss']:.6f} "
            f"val/improvement={val_stats['reconstruction_improvement']:.6f} "
            f"val/relative_improvement={val_stats['relative_improvement']:.3%} "
            f"outside_change={val_stats['outside_mask_max_change']:.3e} "
            f"train_time={train_seconds:.1f}s "
            f"val_time={validation_seconds:.1f}s "
            f"epoch_time={epoch_seconds:.1f}s"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": model_config.to_dict(),
            "loss_config": loss_config.__dict__,
            "history": history,
        }
        atomic_torch_save(checkpoint, output_root / "last_checkpoint.pt")
        if val_stats["reconstruction_loss"] < best_validation:
            best_validation = val_stats["reconstruction_loss"]
            atomic_torch_save(checkpoint, output_root / "best_model.pt")
        write_csv_rows(
            output_root / "history.csv",
            history,
            fieldnames=list(history[0]),
        )


if __name__ == "__main__":
    main()
