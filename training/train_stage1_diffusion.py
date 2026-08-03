from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import atomic_torch_save, atomic_write_json, write_csv_rows
from models.reconstruction_head.coarse_reconstructor import CoarseLiDARRadarReconstructor
from models.reconstruction_head.datasets import Stage1RadarReconstructionDataset, collate_stage1_batch
from models.reconstruction_head.diffusion_scheduler import DiffusionSchedule
from models.reconstruction_head.losses import diffusion_noise_loss, residual_l1_loss
from models.reconstruction_head.residual_diffusion_unet import ResidualDiffusionUNet
from PFS.training_utils import capture_rng_state, resolve_device, seed_everything
from training.stage1_common import build_optimizer_scheduler, gpu_peak_memory_mb, prepare_stage1_paths


def load_frozen_coarse(path: Path, device: torch.device) -> CoarseLiDARRadarReconstructor:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = CoarseLiDARRadarReconstructor(
        lidar_channels=int(args.get("lidar_channels", 3)),
        radar_channels=int(args.get("radar_channels", 4)),
        output_channels=int(args.get("lidar_channels", 3)),
        base_channels=int(args.get("base_channels", 16)),
        levels=int(args.get("levels", 4)),
        normalization=args.get("normalization", "batch"),
        dropout=float(args.get("dropout", 0.0)),
        use_occupancy=True,
    ).to(device)
    # Checkpoint may not have explicit channel args; infer from first conv if needed by non-strict load is not safe,
    # so keep the default repository representation of 3 LiDAR and 4 radar channels.
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def run_epoch(coarse_model, diffusion_model, schedule, loader, device, optimizer, scaler, args, train: bool):
    diffusion_model.train(train)
    totals = {"loss": 0.0, "noise": 0.0, "residual_l1": 0.0}
    samples = 0
    start = time.perf_counter()
    for batch in tqdm(loader, desc="train" if train else "validation", leave=False):
        lidar_corrupt = batch["lidar_corrupt"].to(device, non_blocking=True)
        lidar_clean = batch["lidar_clean"].to(device, non_blocking=True)
        radar = batch["radar"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        occupancy = batch["occupancy"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            coarse_output = coarse_model(lidar_corrupt, radar, mask, occupancy)
            coarse_features = coarse_output["coarse_features"].detach()
            conditioning = coarse_output["conditioning_features"].detach()
            residual_target = (mask * (lidar_clean - coarse_features)).detach()
        timestep = torch.randint(0, schedule.num_train_timesteps, (lidar_corrupt.shape[0],), device=device)
        noise = torch.randn_like(residual_target) * mask
        noisy = schedule.add_noise(residual_target, timestep, noise, mask)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                noise_prediction = diffusion_model(noisy, coarse_features, conditioning, mask, timestep)
                noise_loss = diffusion_noise_loss(noise, noise_prediction, mask)
                residual_loss = noise_loss.new_zeros(())
                if args.residual_loss_weight > 0:
                    pred_residual = schedule.reconstruct_x0(noisy, timestep, noise_prediction)
                    residual_loss = residual_l1_loss(pred_residual, residual_target, mask)
                loss = noise_loss + args.residual_loss_weight * residual_loss
            if train:
                scaler.scale(loss).backward()
                for parameter in coarse_model.parameters():
                    if parameter.grad is not None:
                        raise AssertionError("Frozen coarse model received gradients during diffusion training")
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), args.grad_clip, error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
        batch_size = lidar_corrupt.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        totals["noise"] += float(noise_loss.detach()) * batch_size
        totals["residual_l1"] += float(residual_loss.detach()) * batch_size
        samples += batch_size
    elapsed = time.perf_counter() - start
    return {key: value / max(samples, 1) for key, value in totals.items()} | {
        "seconds_per_sample": elapsed / max(samples, 1),
        "peak_gpu_memory_mb": gpu_peak_memory_mb(device),
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val, args, history):
    atomic_torch_save(
        {
            "stage": "stage1_diffusion",
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
    parser = argparse.ArgumentParser(description="Train Stage I-B residual diffusion denoiser.")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--coarse-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--normalization", choices=["batch", "group", "none"], default="batch")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--residual-loss-weight", type=float, default=0.0)
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
    dataset_kwargs = {
        "radar_root": Path(args.radar_root),
        "resize_hw": (args.resize_height, args.resize_width),
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
    coarse = load_frozen_coarse(Path(args.coarse_checkpoint), device)
    with torch.no_grad():
        probe = coarse(
            first["lidar_corrupt"][None].to(device),
            first["radar"][None].to(device),
            first["mask"][None].to(device),
            first["occupancy"][None].to(device),
        )
    diffusion = ResidualDiffusionUNet(
        residual_channels=first["lidar_corrupt"].shape[0],
        coarse_channels=first["lidar_corrupt"].shape[0],
        conditioning_channels=probe["conditioning_features"].shape[1],
        base_channels=args.base_channels,
        levels=args.levels,
        normalization=args.normalization,
        dropout=args.dropout,
    ).to(device)
    schedule = DiffusionSchedule(num_train_timesteps=args.num_train_timesteps).to(device)
    optimizer, scheduler = build_optimizer_scheduler(diffusion, args)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "training_config.json", vars(args))
    print(f"Device: {device} | train={len(train_dataset)} val={len(val_dataset)}")
    print("Train fault counts:", json.dumps(train_counts, sort_keys=True))
    print("Val fault counts:", json.dumps(val_counts, sort_keys=True))
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_stats = run_epoch(coarse, diffusion, schedule, train_loader, device, optimizer, scaler, args, True)
        with torch.no_grad():
            val_stats = run_epoch(coarse, diffusion, schedule, val_loader, device, optimizer, scaler, args, False)
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
            f"noise={row['val_noise']:.6f} residual_l1={row['val_residual_l1']:.6f} "
            f"gpu_mb={row['val_peak_gpu_memory_mb']:.1f}"
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(checkpoint_dir / "best_model.pt", diffusion, optimizer, scheduler, scaler, epoch, best_val, args, history)
        save_checkpoint(checkpoint_dir / "last_checkpoint.pt", diffusion, optimizer, scheduler, scaler, epoch, best_val, args, history)
        write_csv_rows(output_root / "history.csv", history, fieldnames=list(dict.fromkeys(k for row in history for k in row)))
        if args.debug_overfit and epoch >= min(args.epochs, 10):
            break


if __name__ == "__main__":
    main()

