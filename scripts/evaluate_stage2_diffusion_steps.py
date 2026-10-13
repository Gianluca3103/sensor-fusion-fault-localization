"""Compare DDIM sampling lengths on the same Stage-II validation frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import torch

from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from models.radar_lidar_stage2_diffusion.model import ResidualSparseDiffusion
from models.radar_lidar_stage2_diffusion.train import _loader, evaluate
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from scripts.visualize_stage1_confidence_cloud import load_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--fault-samples-root", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--fault-pattern", default="*")
    parser.add_argument("--sampling-steps", type=int, nargs="+", default=[3, 16])
    parser.add_argument("--val-limit", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("off", "bf16"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.val_limit < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("Validation limit and batch size must be positive; workers nonnegative")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (saved.get("model_type") != "stage2_residual_sparse_diffusion_v1"
            or saved.get("diffusion_enabled") is not True):
        raise ValueError("Expected a sparse diffusion Stage-II checkpoint")
    if args.fault_pattern != saved["fault_pattern"]:
        raise ValueError("Fault pattern differs from training")
    config = DiffusionConfig.from_checkpoint_dict(saved["config"])
    for steps in args.sampling_steps:
        if not 1 <= steps <= config.timesteps:
            parser.error(f"Sampling steps must be between 1 and {config.timesteps}")
    amp = args.amp or saved.get("amp", "off")
    if amp == "bf16" and torch.device(args.device).type != "cuda":
        parser.error("bf16 requires CUDA")

    stage1_path = args.stage1_checkpoint or Path(saved["stage1_checkpoint"])
    stage1, stage1_saved = load_encoder(stage1_path, args.device)
    variant = saved["radar_variant"]
    trained_variant = stage1_saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != variant:
        raise ValueError("Stage-I radar variant differs from Stage II")
    stage1.eval()
    model = ResidualSparseDiffusion(stage1.config.channels, config).to(args.device).eval()
    model.load_state_dict(saved["model"])
    loader_args = SimpleNamespace(
        vod_root=args.vod_root, fault_samples_root=args.fault_samples_root,
        radar_variant=variant, fault_pattern=args.fault_pattern,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    loader = _loader(loader_args, "val", args.val_limit, False)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(saved["epoch"]),
        "trained_timesteps": config.timesteps,
        "validation_frames": len(loader.dataset),
        "stage1_checkpoint": str(stage1_path.resolve()),
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for steps in args.sampling_steps:
        start = time.perf_counter()
        with torch.no_grad():
            report = evaluate(model, stage1, loader, config, args.device, amp,
                              sampling_steps=steps)
        report["elapsed_seconds"] = time.perf_counter() - start
        summary["results"].append(report)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{steps} steps | global missing F1@20cm "
              f"{report['global_missing_f1_0.2m']:.4f} | addition F1@20cm "
              f"{report['addition_f1_0.2m']:.4f} | predicted {report['predicted_sites']} "
              f"| {report['elapsed_seconds']:.1f}s", flush=True)
    print(f"Saved comparison: {args.output}", flush=True)


if __name__ == "__main__":
    main()
