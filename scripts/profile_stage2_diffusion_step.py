"""Profile one real Stage-II diffusion training batch on the target GPU.

Run while the training process is stopped. This measures the actual Stage-I
encoder, Stage-II preparation, one denoising pass, backward, and optimizer.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import torch

from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from models.radar_lidar_stage2_diffusion.model import ResidualSparseDiffusion
from models.radar_lidar_stage2_diffusion.train import _denoise
from models.radar_lidar_stage2_residual.data import PairedFaultDataset, collate_paired
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from scripts.visualize_stage1_confidence_cloud import load_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--fault-samples-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--config", default="configs/radar_lidar_stage2_diffusion.json")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--fault-pattern", default="*")
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--amp", choices=("off", "bf16"), default="bf16")
    parser.add_argument("--split-backward", action="store_true",
                        help="Time denoiser and trainable conditioning gradients separately")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1 or args.start_index < 0:
        parser.error("Batch size, repeats and start index must be valid")
    device = torch.device(args.device)
    if args.amp == "bf16" and device.type != "cuda":
        parser.error("BF16 profiling requires CUDA")

    data = PairedFaultDataset(args.vod_root, args.fault_samples_root, "train",
                              radar_variant=args.radar_variant, fault_pattern=args.fault_pattern)
    if args.start_index + args.batch_size > len(data):
        parser.error("Requested batch extends beyond the training set")
    config = DiffusionConfig.from_json(args.config)
    stage1, saved = load_encoder(Path(args.stage1_checkpoint), str(device))
    trained_variant = saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != args.radar_variant:
        raise ValueError("Stage-I checkpoint was trained with a different radar stack")
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    model = ResidualSparseDiffusion(stage1.config.conditioning_channels, config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=.01)

    ticks: dict[str, list[float]] = {}

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(name, operation):
        sync()
        start = time.perf_counter()
        result = operation()
        sync()
        ticks.setdefault(name, []).append(time.perf_counter() - start)
        return result

    raw = timed("load_and_collate", lambda: collate_paired(
        [data[i] for i in range(args.start_index, args.start_index + args.batch_size)]))
    batch = timed("transfer_to_device", lambda: {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records = []
    for _ in range(args.repeats):
        with torch.no_grad():
            evidence = timed("stage1_encoder", lambda: stage1(
                batch["radar"], batch["radar_valid"], defer_diagnostics=True))
        context = timed("candidate_coverage_and_conditioning", lambda: model.prepare(
            evidence, stage1.config.grid, batch["faulty_lidar"],
            batch["faulty_lidar_valid"], batch["fault_region"]))
        target = timed("clean_targets", lambda: make_residual_targets(
            context.domain, batch["clean_lidar"], batch["clean_lidar_valid"],
            context.coverage, free_ray_tolerance_m=config.residual.free_ray_tolerance_m))
        optimizer.zero_grad(set_to_none=True)
        if args.split_backward:
            # The detached leaf keeps the exact forward values and receives
            # the gradient that will be propagated through the conditioner.
            condition_leaf = context.condition.detach().requires_grad_(True)
            denoiser_context = replace(context, condition=condition_leaf)
        else:
            denoiser_context = context
        losses, counts = timed("denoiser_forward_and_loss", lambda: _denoise(
            model, denoiser_context, target, args.batch_size, amp=args.amp))
        if not bool(torch.isfinite(losses["total"])):
            raise FloatingPointError("The profiled diffusion loss is non-finite")
        if args.split_backward:
            timed("denoiser_backward", lambda: losses["total"].backward())
            if condition_leaf.grad is None:
                raise AssertionError("The denoiser did not propagate to Stage-II conditioning")
            timed("conditioning_backward", lambda: context.condition.backward(condition_leaf.grad))
        else:
            timed("backward", lambda: losses["total"].backward())
        timed("optimizer_step", optimizer.step)
        records.append({"candidate_sites": len(context.domain.coordinates),
                        "unet_input_sites": counts["input"],
                        "may_add_sites": int(context.coverage.may_add.sum()),
                        "supervised_sites": int(target.supervised.sum()),
                        "region_sites": int(context.coverage.in_fault_region.sum())})
    print(json.dumps({"frames": batch["frame_id"], "batch_size": args.batch_size,
                      "repeats": args.repeats, "amp": args.amp,
                      "split_backward": args.split_backward,
                      "seconds_median": {key: statistics.median(values)
                                         for key, values in ticks.items()},
                      "seconds_each": ticks, "iterations": records,
                      "cuda_peak_allocated_gib": (torch.cuda.max_memory_allocated(device) / 2**30
                                                  if device.type == "cuda" else None)}, indent=2))


if __name__ == "__main__":
    main()
