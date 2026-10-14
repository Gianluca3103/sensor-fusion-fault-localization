"""Evaluate the full frozen-Stage-I and Stage-II reconstruction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.radar_lidar_stage2_diffusion.cache import cached_loader, file_sha256, provenance
from models.radar_lidar_stage2_residual.train import _loader

from .config import BottleneckDiffusionConfig
from .model import BottleneckSparseDiffusion
from .train import MODEL_TYPE, evaluate, load_frozen_stage1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--fault-samples-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint")
    parser.add_argument("--preprocess-cache-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--sampling-method", choices=("ddpm", "ddim"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("model_type") != MODEL_TYPE:
        parser.error("Checkpoint is not Stage-II bottleneck diffusion v2")
    config = BottleneckDiffusionConfig.from_dict(saved["config"])
    stage1_path = args.stage1_checkpoint or saved["stage1_checkpoint"]
    if file_sha256(stage1_path) != saved["stage1_sha256"]:
        parser.error("Stage-I checkpoint content differs from the trained conditioner")
    if Path(args.fault_samples_root).resolve() != Path(saved["fault_samples_root"]).resolve():
        parser.error("Fault samples differ from the training run")
    stage1, _ = load_frozen_stage1(stage1_path, args.device, saved["radar_variant"])
    model = BottleneckSparseDiffusion(stage1.config.conditioning_channels, config).to(args.device)
    model.load_state_dict(saved["model"])
    model.eval()
    args.radar_variant = saved["radar_variant"]
    args.fault_pattern = saved["fault_pattern"]
    args.train_limit = 0
    if args.preprocess_cache_root:
        expected = provenance(stage1_checkpoint=stage1_path, vod_root=args.vod_root,
                              fault_samples_root=args.fault_samples_root,
                              radar_variant=args.radar_variant,
                              fault_pattern=args.fault_pattern, config=config)
        loader = cached_loader(args, "val", args.val_limit, False, expected)
    else:
        loader = _loader(args, "val", args.val_limit, False)
    report = evaluate(model, stage1, loader, config, args.device,
                      amp=saved.get("amp", "off"), sampling_steps=args.sampling_steps,
                      sampling_method=args.sampling_method)
    report["stage2_epoch"] = saved["epoch"]
    report["stage1_checkpoint"] = str(Path(stage1_path).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
