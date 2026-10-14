"""Train Stage-II bottleneck diffusion with frozen, radar-only Stage-I features."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
from tqdm import tqdm

from models.radar_lidar_stage1.bottleneck_support import interface_version
from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from models.radar_lidar_stage2.voxel_target import VoxelTargets
from models.radar_lidar_stage2_residual.coverage import FaultyCoverage
from models.radar_lidar_stage2_residual.targets import ResidualTargets, make_residual_targets
from models.radar_lidar_stage2_residual.train import _loader
from models.radar_lidar_stage2_diffusion.cache import cached_loader, file_sha256, provenance
from scripts.visualize_stage1_confidence_cloud import load_encoder

from .config import BottleneckDiffusionConfig
from .losses import reconstruction_loss, training_state
from .metrics import ReconstructionMetrics
from .model import BottleneckSparseDiffusion


MODEL_TYPE = "stage2_bottleneck_sparse_diffusion_v2"
LOSS_NAMES = ("total", "diffusion", "noise_positive", "noise_free",
              "noise_observed", "occupancy", "geometry", "preservation")


def load_frozen_stage1(path: str | Path, device: str,
                       radar_variant: str | None = None):
    """Load only the deployed radar student and enforce its checkpoint interface."""
    stage1, saved = load_encoder(Path(path), device)
    if (not stage1.config.bottleneck_support_only or
            stage1.config.support_seed_mode != "radar_s2" or
            saved.get("stage1_interface") != interface_version(stage1.config)):
        raise ValueError("Stage II v2 requires a trained radar-seed bottleneck Stage-I checkpoint")
    if radar_variant is not None and saved.get("data", {}).get("radar_variant") != radar_variant:
        raise ValueError("Stage-I checkpoint radar variant differs from Stage II")
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    return stage1, saved


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def autocast(device: str, amp: str):
    if amp == "off":
        return nullcontext()
    if amp != "bf16" or torch.device(device).type != "cuda":
        raise ValueError("BFloat16 AMP requires CUDA")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def protect_target(target: ResidualTargets, coverage: FaultyCoverage) -> ResidualTargets:
    """Train protected sites as no-addition, as they are clamped at sampling."""
    if len(target.addition) != len(coverage.may_add) or bool(
            (target.addition & ~coverage.may_add).any()):
        raise ValueError("Clean addition target escaped the fault repair region")
    return ResidualTargets(target.addition, ~coverage.may_add,
                           target.known_free, target.offsets_normalized, target.clean)


def prepare_context(model, stage1, batch, config):
    if "cache" in batch:
        data = batch["cache"]
        domain = CandidateDomain(data["coordinates"], data["confidence"],
                                 data["grid"], data["counts"])
        coverage = FaultyCoverage(data["coverage_features"], data["measured_voxel"],
                                  data["measured_ray"], data["in_fault_region"])
        return model.prepare_cached(domain, coverage, data["sources"],
                                    data["indices"], data["source_centers"],
                                    data["source_batches"], len(batch["frame_id"]))
    with torch.no_grad():
        evidence = stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)
    return model.prepare(evidence, stage1.config.grid,
                         batch["faulty_lidar"], batch["faulty_lidar_valid"],
                         batch["fault_region"])


def make_target(context, batch, config):
    if "cache" in batch:
        data = batch["cache"]
        clean = VoxelTargets(data["clean_occupied"], data["clean_known_free"],
                             data["clean_offsets"], data["clean_centroids"],
                             data["clean_point_count"], data["clean_points_in_grid"],
                             data["clean_points_in_candidates"], data["point_centroid_errors"])
        target = ResidualTargets(data["target_addition"], data["target_observed"],
                                 data["target_known_free"], clean.offsets_normalized, clean)
    else:
        target = make_residual_targets(context.domain, batch["clean_lidar"],
                                       batch["clean_lidar_valid"], context.coverage,
                                       free_ray_tolerance_m=config.residual.free_ray_tolerance_m)
    return protect_target(target, context.coverage)


def context_and_target(model, stage1, batch, config):
    context = prepare_context(model, stage1, batch, config)
    target = make_target(context, batch, config)
    return context, target


def denoise_batch(model, context, target, *, amp="off", fixed_timestep=None,
                  generator=None):
    state = training_state(target, generator=generator)
    batches = context.batch_size
    frame_t = (torch.randint(model.config.timesteps, (batches,), device=state.device,
                             generator=generator)
               if fixed_timestep is None else
               torch.full((batches,), fixed_timestep, device=state.device, dtype=torch.long))
    site_t = frame_t[context.domain.coordinates[:, 0].long()]
    noise = torch.randn(state.shape, device=state.device, dtype=state.dtype,
                        generator=generator)
    noisy = model.schedule.mix(state, noise, site_t)
    with autocast(str(state.device), amp):
        predicted, logits, offsets, counts = model.predict_noise(context, noisy, frame_t)
    losses = reconstruction_loss(predicted.float(), noise, logits.float(), offsets.float(),
                                 target, context.coverage, model.config)
    return losses, counts


def train_epoch(model, stage1, loader, optimizer, config, device, accumulation,
                epoch, epochs, amp):
    model.train()
    stage1.eval()
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(len(LOSS_NAMES), device=device, dtype=torch.float64)
    bar = tqdm(loader, desc=f"Bottleneck diffusion {epoch:03d}/{epochs:03d}",
               unit="batch", dynamic_ncols=True)
    for index, raw in enumerate(bar):
        batch = to_device(raw, device)
        context, target = context_and_target(model, stage1, batch, config)
        losses, counts = denoise_batch(model, context, target, amp=amp)
        if not bool(torch.isfinite(losses["total"])):
            raise FloatingPointError(f"Non-finite Stage-II loss at epoch {epoch}, batch {index + 1}")
        group_start = index - index % accumulation
        group_size = min(accumulation, len(loader) - group_start)
        (losses["total"] / group_size).backward()
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        sums += torch.stack([losses[name].detach().double() for name in LOSS_NAMES])
        if index % 20 == 0:
            bar.set_postfix(loss=f"{(sums[0] / (index + 1)).item():.4f}",
                            sites=counts["input"])
    return dict(zip(LOSS_NAMES, (sums / max(len(loader), 1)).cpu().tolist()))


@torch.no_grad()
def evaluate(model, stage1, loader, config, device, *, amp="off",
             sampling_steps=None, sampling_method=None):
    model.eval()
    stage1.eval()
    metrics = ReconstructionMetrics()
    sums = torch.zeros(len(LOSS_NAMES), device=device, dtype=torch.float64)
    preparation_elapsed = 0.0
    sampling_elapsed = 0.0
    peak_bytes = 0
    cached = False
    frames_evaluated = 0
    generator = torch.Generator(device=device).manual_seed(config.sample_seed)
    for index, raw in enumerate(tqdm(loader, desc="Bottleneck diffusion validation",
                                     unit="frame", dynamic_ncols=True)):
        batch = to_device(raw, device)
        frames_evaluated += len(batch["frame_id"])
        cached = "cache" in batch
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        preparation_start = time.perf_counter()
        context = prepare_context(model, stage1, batch, config)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
        preparation_elapsed += time.perf_counter() - preparation_start
        target = make_target(context, batch, config)
        losses, _ = denoise_batch(model, context, target, amp=amp,
                                  fixed_timestep=config.timesteps // 2,
                                  generator=generator)
        sums += torch.stack([losses[name].detach().double() for name in LOSS_NAMES])
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with autocast(device, amp):
            output = model.sample(context,
                                  frame_seeds=[config.sample_seed + int(frame)
                                               for frame in batch["frame_id"]],
                                  sampling_steps=sampling_steps,
                                  method=sampling_method)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
        sampling_elapsed += time.perf_counter() - start
        metrics.add(output, target, batch["clean_lidar"], batch["clean_lidar_valid"],
                    batch["faulty_lidar"], batch["faulty_lidar_valid"],
                    batch.get("fault_kind", ["unknown"] * len(batch["frame_id"])),
                    config.residual.occupancy_threshold)
    report = metrics.summary()
    report["frames"] = frames_evaluated
    report["loss"] = dict(zip(LOSS_NAMES, (sums / max(len(loader), 1)).cpu().tolist()))
    denominator = max(frames_evaluated, 1)
    report["sampling"] = {"method": sampling_method or config.sampling_method,
                          "steps": sampling_steps or config.timesteps,
                          "seconds_per_frame": sampling_elapsed / denominator,
                          "conditioning_seconds_per_frame": preparation_elapsed / denominator,
                          "full_pipeline_seconds_per_frame": (
                              (preparation_elapsed + sampling_elapsed) / denominator
                              if not cached else None),
                          "stage1_encoder_included_in_timing": not cached,
                          "peak_gpu_allocated_gib": peak_bytes / 2**30}
    return report


def save_checkpoint(path, model, optimizer, epoch, config, args, stage1_hash,
                    validation, best):
    state = {"model_type": MODEL_TYPE, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "epoch": epoch,
             "config": config.as_dict(),
             "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
             "stage1_sha256": stage1_hash, "radar_variant": args.radar_variant,
             "fault_samples_root": str(Path(args.fault_samples_root).resolve()),
             "fault_pattern": args.fault_pattern, "amp": args.amp,
             "validation": validation, "best_score": best}
    temporary = Path(str(path) + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--fault-samples-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preprocess-cache-root")
    parser.add_argument("--config", default="configs/radar_lidar_stage2_bottleneck.json")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--fault-pattern", default="*")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--validation-sampling-steps", type=int)
    parser.add_argument("--validation-sampling-method", choices=("ddpm", "ddim"))
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--resume")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum_steps, args.validate_every) < 1:
        parser.error("Epochs, batch size, accumulation and validation interval must be positive")
    if args.amp != "off" and torch.device(args.device).type != "cuda":
        parser.error("bf16 AMP requires CUDA")
    config = BottleneckDiffusionConfig.from_json(args.config)
    method = args.validation_sampling_method or config.sampling_method
    try:
        from .schedule import NoiseSchedule
        NoiseSchedule(config.timesteps, config.beta_schedule).sampling_timesteps(
            args.validation_sampling_steps, method)
    except ValueError as error:
        parser.error(str(error))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if not args.resume and (root / "metrics.jsonl").exists():
        raise ValueError("Output root already contains metrics; use a new run or --resume")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    stage1, _ = load_frozen_stage1(args.stage1_checkpoint, args.device,
                                   args.radar_variant)
    stage1_hash = file_sha256(args.stage1_checkpoint)
    model = BottleneckSparseDiffusion(stage1.config.conditioning_channels, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    start_epoch, best = 1, -float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("model_type") != MODEL_TYPE or saved["config"] != config.as_dict()
                or saved["stage1_sha256"] != stage1_hash
                or saved["radar_variant"] != args.radar_variant
                or saved["fault_samples_root"] != str(Path(args.fault_samples_root).resolve())
                or saved["fault_pattern"] != args.fault_pattern):
            raise ValueError("Stage-II resume checkpoint or conditioning data differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best = int(saved["epoch"]) + 1, float(saved["best_score"])
    if args.preprocess_cache_root:
        expected = provenance(stage1_checkpoint=args.stage1_checkpoint,
                              vod_root=args.vod_root, fault_samples_root=args.fault_samples_root,
                              radar_variant=args.radar_variant,
                              fault_pattern=args.fault_pattern, config=config)
        training = cached_loader(args, "train", args.train_limit, True, expected)
        validation = cached_loader(args, "val", args.val_limit, False, expected)
    else:
        training = _loader(args, "train", args.train_limit, True)
        validation = _loader(args, "val", args.val_limit, False)
    print(f"Stage II v2: train={len(training.dataset)} val={len(validation.dataset)} "
          f"Stage I frozen=True fusion={config.fusion_mode} "
          f"timesteps={config.timesteps} cache={bool(args.preprocess_cache_root)}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train = train_epoch(model, stage1, training, optimizer, config, args.device,
                            args.grad_accum_steps, epoch, args.epochs, args.amp)
        report = None
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            report = evaluate(model, stage1, validation, config, args.device,
                              amp=args.amp, sampling_steps=args.validation_sampling_steps,
                              sampling_method=args.validation_sampling_method)
            score = report["global_missing_f1_0.2m"]
            if score > best:
                best = score
                save_checkpoint(root / "best_global_missing.ckpt", model, optimizer,
                                epoch, config, args, stage1_hash, report, best)
        save_checkpoint(root / "last.ckpt", model, optimizer, epoch, config,
                        args, stage1_hash, report, best)
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "train": train,
                                     "validation": report, "best_score": best}) + "\n")
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | total {train['total']:.4f} "
              f"diffusion {train['diffusion']:.4f} occupancy {train['occupancy']:.4f} "
              f"geometry {train['geometry']:.4f} preservation {train['preservation']:.4f}",
              flush=True)
        if report:
            print(f"  val missing F1@20cm {report['global_missing_f1_0.2m']:.3f} "
                  f"voxel IoU {report['voxel_repair']['iou_on_observed']:.3f} "
                  f"predicted {report['predicted_sites']}", flush=True)


if __name__ == "__main__":
    main()
