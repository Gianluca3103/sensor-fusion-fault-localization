from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import atomic_torch_save, atomic_write_json, write_csv_rows
from models.reconstruction_head.coarse_reconstructor import CoarseLiDARRadarReconstructor, coarse_parameter_breakdown
from models.reconstruction_head.datasets import Stage1RadarReconstructionDataset, collate_stage1_batch
from models.reconstruction_head.losses import coarse_reconstruction_loss, masked_feature_loss, healthy_region_change
from PFS.training_utils import capture_rng_state, resolve_device, seed_everything
from training.stage1_common import build_optimizer_scheduler, gpu_peak_memory_mb, prepare_stage1_paths


def run_epoch(model, loader, device, optimizer, scaler, args, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "feature": 0.0, "healthy_change": 0.0, "masked_mae": 0.0, "masked_mse": 0.0}
    samples = 0
    start = time.perf_counter()
    for batch in tqdm(loader, desc="train" if train else "validation", leave=False):
        lidar_corrupt = batch["lidar_corrupt"].to(device, non_blocking=True)
        lidar_clean = batch["lidar_clean"].to(device, non_blocking=True)
        radar = batch["radar"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        occupancy = batch["occupancy"].to(device, non_blocking=True)
        if lidar_corrupt.shape != lidar_clean.shape:
            raise ValueError("F_L_clean and F_L_corrupt must have identical shapes")
        if radar.shape[-2:] != lidar_corrupt.shape[-2:]:
            raise ValueError("Radar and LiDAR spatial dimensions must match in Stage I dataset")
        if mask.shape[1] != 1:
            raise ValueError("M_GT must have one channel")
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(lidar_corrupt, radar, mask, occupancy)
                loss, diagnostics = coarse_reconstruction_loss(
                    outputs,
                    lidar_corrupt,
                    lidar_clean,
                    mask,
                    feature_weight=args.feature_loss_weight,
                    occupancy_weight=args.occupancy_loss_weight,
                    offset_weight=args.offset_loss_weight,
                    feature_loss_mode=args.feature_loss_mode,
                )
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
        with torch.no_grad():
            error = mask * (outputs["coarse_features"] - lidar_clean)
            batch_size = lidar_corrupt.shape[0]
            totals["loss"] += float(loss.detach()) * batch_size
            totals["feature"] += float(diagnostics["feature"]) * batch_size
            totals["healthy_change"] += float(healthy_region_change(outputs["coarse_features"], lidar_corrupt, mask)) * batch_size
            totals["masked_mae"] += float(error.abs().sum() / (mask.sum().clamp_min(1e-6) * lidar_corrupt.shape[1])) * batch_size
            totals["masked_mse"] += float(error.square().sum() / (mask.sum().clamp_min(1e-6) * lidar_corrupt.shape[1])) * batch_size
            samples += batch_size
    elapsed = time.perf_counter() - start
    return {key: value / max(samples, 1) for key, value in totals.items()} | {
        "seconds_per_sample": elapsed / max(samples, 1),
        "peak_gpu_memory_mb": gpu_peak_memory_mb(device),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val, args, history):
    atomic_torch_save(
        {
            "stage": "stage1_coarse",
            "epoch": epoch,
            "best_val": best_val,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
            "history": history,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Stage I-A deterministic LiDAR/radar coarse reconstructor.")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--normalization", choices=["batch", "group", "none"], default="batch")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--feature-loss-mode", choices=["l1", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--occupancy-loss-weight", type=float, default=0.0)
    parser.add_argument("--offset-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-patches", action="store_true")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--halo-radius", type=int, default=12)
    parser.add_argument("--no-full-frame-fallback", action="store_false", dest="full_frame_fallback")
    parser.add_argument("--max-radar-delta-ms", type=float, default=None)
    parser.add_argument("--include-faults", nargs="*", default=None)
    parser.add_argument("--exclude-faults", nargs="*", default=None)
    parser.add_argument("--debug-overfit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_paths, val_paths, train_counts, val_counts = prepare_stage1_paths(args)
    resize_hw = (args.resize_height, args.resize_width)
    dataset_kwargs = {
        "radar_root": Path(args.radar_root),
        "resize_hw": resize_hw,
        "mask_threshold": args.mask_threshold,
        "use_patches": args.use_patches,
        "patch_size": args.patch_size,
        "halo_radius": args.halo_radius,
        "full_frame_fallback": args.full_frame_fallback,
    }
    train_dataset = Stage1RadarReconstructionDataset(train_paths, **dataset_kwargs)
    val_dataset = Stage1RadarReconstructionDataset(val_paths, **dataset_kwargs)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_stage1_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    first = train_dataset[0]
    args.lidar_channels = int(first["lidar_corrupt"].shape[0])
    args.radar_channels = int(first["radar"].shape[0])
    args.output_channels = int(first["lidar_clean"].shape[0])
    model = CoarseLiDARRadarReconstructor(
        lidar_channels=args.lidar_channels,
        radar_channels=args.radar_channels,
        output_channels=args.output_channels,
        base_channels=args.base_channels,
        levels=args.levels,
        normalization=args.normalization,
        dropout=args.dropout,
        use_occupancy=True,
    ).to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "training_config.json", vars(args))
    print(f"Device: {device} | train={len(train_dataset)} val={len(val_dataset)}")
    print("Train fault counts:", json.dumps(train_counts, sort_keys=True))
    print("Val fault counts:", json.dumps(val_counts, sort_keys=True))
    print("Input channels:", first["lidar_corrupt"].shape[0], "Radar channels:", first["radar"].shape[0])
    print("Parameters:", json.dumps(coarse_parameter_breakdown(model), indent=2))
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_stats = run_epoch(model, train_loader, device, optimizer, scaler, args, True)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, device, optimizer, scaler, args, False)
        if args.scheduler == "plateau":
            scheduler.step(val_stats["loss"])
        else:
            scheduler.step()
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)
        print(
            f"epoch {epoch:03d}: train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"mae={row['val_masked_mae']:.6f} mse={row['val_masked_mse']:.6f} "
            f"healthy={row['val_healthy_change']:.8f} gpu_mb={row['val_peak_gpu_memory_mb']:.1f}"
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(checkpoint_dir / "best_model.pt", model, optimizer, scheduler, scaler, epoch, best_val, args, history)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", model, optimizer, scheduler, scaler, epoch, best_val, args, history)
        write_csv_rows(output_root / "history.csv", history, fieldnames=list(dict.fromkeys(k for row in history for k in row)))
        if args.debug_overfit and epoch >= min(args.epochs, 10):
            break


if __name__ == "__main__":
    main()
