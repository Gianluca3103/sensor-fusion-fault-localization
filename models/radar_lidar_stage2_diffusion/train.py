"""Train sparse 3D diffusion from frozen Stage I, radar, and faulty LiDAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from models.radar_lidar_stage2_residual.targets import make_residual_targets
from models.radar_lidar_stage2_residual.train import ResidualMetrics, _loader, _to_device
from scripts.visualize_stage1_confidence_cloud import load_encoder

from .config import DiffusionConfig
from .losses import training_state, denoising_loss
from .model import ResidualSparseDiffusion


def _batch_count(batch: dict) -> int:
    return batch["radar"].shape[0]


def _context_target(model, stage1, batch, config):
    with torch.no_grad():
        evidence = stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)
    context = model.prepare(evidence, stage1.config.grid, batch["faulty_lidar"],
                            batch["faulty_lidar_valid"], batch["fault_region"])
    target = make_residual_targets(context.domain, batch["clean_lidar"],
                                   batch["clean_lidar_valid"], context.coverage,
                                   free_ray_tolerance_m=config.residual.free_ray_tolerance_m)
    return context, target


def _denoise(model, context, target, batches, *, generator=None, fixed_timestep=None):
    clean = training_state(target, generator=generator)
    device = clean.device
    frame_t = (torch.randint(model.config.timesteps, (batches,), device=device)
               if fixed_timestep is None else
               torch.full((batches,), fixed_timestep, dtype=torch.long, device=device))
    site_t = frame_t[context.domain.coordinates[:, 0].long()]
    noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
    noisy = model.schedule.mix(clean, noise, site_t)
    predicted, counts = model.predict_clean(context, noisy, frame_t)
    return denoising_loss(predicted, clean, target, model.config), counts


def train_epoch(model, stage1, loader, optimizer, config, device, grad_accum, epoch, epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    names = ("total", "occupancy_denoise", "offset_denoise", "classification")
    totals = {key: torch.zeros((), device=device) for key in names}
    bar = tqdm(loader, desc=f"Sparse diffusion {epoch:03d}/{epochs:03d}",
               dynamic_ncols=True, leave=False)
    for step, raw in enumerate(bar, 1):
        batch = _to_device(raw, device)
        context, target = _context_target(model, stage1, batch, config)
        losses, counts = _denoise(model, context, target, _batch_count(batch))
        (losses["total"] / grad_accum).backward()
        if step % grad_accum == 0 or step == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for key in names:
            totals[key] += losses[key].detach()
        if step % 10 == 0 or step == len(loader):
            bar.set_postfix(loss=f"{float(totals['total']/step):.4f}",
                            candidates=counts["input"])
    return {key: float(value / max(len(loader), 1)) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, stage1, loader, config, device):
    model.eval()
    metrics = ResidualMetrics()
    names = ("total", "occupancy_denoise", "offset_denoise", "classification")
    totals = {key: torch.zeros((), device=device) for key in names}
    generator = torch.Generator(device=device).manual_seed(config.sample_seed)
    for raw in tqdm(loader, desc="Sparse diffusion validation", dynamic_ncols=True, leave=False):
        batch = _to_device(raw, device)
        context, target = _context_target(model, stage1, batch, config)
        losses, _ = _denoise(model, context, target, _batch_count(batch), generator=generator,
                             fixed_timestep=config.timesteps // 2)
        for key in names:
            totals[key] += losses[key]
        output = model.sample(context, frame_seeds=[config.sample_seed + int(frame)
                                                    for frame in batch["frame_id"]])
        metrics.add(output, target, config.residual.occupancy_threshold,
                    batch["clean_lidar"], batch["clean_lidar_valid"],
                    batch["faulty_lidar"], batch["faulty_lidar_valid"])
    report = metrics.summary()
    report["loss"] = {key: float(value / max(len(loader), 1)) for key, value in totals.items()}
    report["sampling_steps"] = config.timesteps
    return report


def _save(path, model, optimizer, epoch, config, args, report, best):
    state = {"model_type": "stage2_residual_sparse_diffusion_v1", "epoch": epoch,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "config": config.as_dict(), "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
             "radar_variant": args.radar_variant,
             "fault_samples_root": str(Path(args.fault_samples_root).resolve()),
             "fault_pattern": args.fault_pattern, "validation": report,
             "best_score": best, "diffusion_enabled": True}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--fault-samples-root", required=True)
    parser.add_argument("--fault-pattern", default="*")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/radar_lidar_stage2_diffusion.json")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=.0002)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum_steps, args.validate_every) < 1:
        parser.error("Epochs, batch size, accumulation and validation interval must be positive")
    config = DiffusionConfig.from_json(args.config)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume and (root / "metrics.jsonl").exists():
        raise ValueError("Output root contains a previous run; use --resume or a new output root")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    stage1, stage1_saved = load_encoder(Path(args.stage1_checkpoint), args.device)
    if not stage1.config.surface_region_enabled:
        raise ValueError("This diffusion run requires the learned-region Stage-I checkpoint")
    trained_variant = stage1_saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != args.radar_variant:
        raise ValueError("Stage-I checkpoint was trained with a different radar stack")
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    model = ResidualSparseDiffusion(stage1.config.channels, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    start, best = 1, -float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("model_type") != "stage2_residual_sparse_diffusion_v1"
                or saved["config"] != config.as_dict()
                or saved["stage1_checkpoint"] != str(Path(args.stage1_checkpoint).resolve())
                or saved["radar_variant"] != args.radar_variant
                or saved["fault_samples_root"] != str(Path(args.fault_samples_root).resolve())
                or saved["fault_pattern"] != args.fault_pattern):
            raise ValueError("Diffusion resume checkpoint and inputs differ")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = int(saved["epoch"]) + 1, float(saved["best_score"])
    training = _loader(args, "train", args.train_limit, True)
    validation = _loader(args, "val", args.val_limit, False)
    print(f"Sparse diffusion: train={len(training.dataset)} val={len(validation.dataset)} "
          f"timesteps={config.timesteps} Stage-I frozen=True grid={stage1.config.grid.size_xyz}", flush=True)
    for epoch in range(start, args.epochs + 1):
        train = train_epoch(model, stage1, training, optimizer, config, args.device,
                            args.grad_accum_steps, epoch, args.epochs)
        report = None
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            report = evaluate(model, stage1, validation, config, args.device)
            score = report["global_missing_f1_0.2m"]
            if score > best:
                best = score
                _save(root / "best_global_missing.ckpt", model, optimizer, epoch,
                      config, args, report, best)
        _save(root / "last.ckpt", model, optimizer, epoch, config, args, report, best)
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "train": train, "validation": report,
                                     "best_global_missing_f1_0.2m": best}) + "\n")
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | denoise {train['total']:.4f} "
              f"occupancy {train['occupancy_denoise']:.4f} "
              f"offset {train['offset_denoise']:.4f}", flush=True)
        if report:
            print(f"  val addition F1@20cm {report['addition_f1_0.2m']:.3f} "
                  f"global missing F1@20cm {report['global_missing_f1_0.2m']:.3f} "
                  f"predicted {report['predicted_sites']} "
                  f"unknown fraction {report['unknown_prediction_fraction']:.3f}", flush=True)


if __name__ == "__main__":
    main()
