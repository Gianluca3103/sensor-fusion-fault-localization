"""Prebuild fixed bottleneck Stage-I geometry before resuming training.

Only coordinates, neighbor maps, distances, and clean-scan supervision are
written. Encoder features and predicted support scores remain live outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_stage1.bottleneck_support import BottleneckSupportStage1, interface_version
from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.static_geometry_cache import Stage1StaticGeometryCache
from models.radar_lidar_stage1.train import config_from_dict


def prebuild_split(model, cache, dataset, split: str, *, batch_size: int,
                   num_workers: int, device: str):
    total = len(dataset)
    pending = [frame for frame in dataset.frames
               if not cache.has_entry(split, frame.frame_id)]
    reused = total - len(pending)
    dataset.frames = pending
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_stage1,
                        pin_memory=device.startswith("cuda"))
    built = 0
    with torch.no_grad():
        for raw in tqdm(loader, desc=f"Stage-I cache {split}", unit="batch",
                        dynamic_ncols=True):
            batch = {key: value.to(device, non_blocking=True)
                     if isinstance(value, torch.Tensor) else value
                     for key, value in raw.items()}
            output, radar_levels, _ = model.radar_only.encode(
                batch["radar"], batch["radar_valid"],
                defer_diagnostics=True, check_finite=False)
            lidar_sites, _ = model.lidar_point(
                batch["clean_lidar"], batch["clean_lidar_valid"],
                defer_diagnostics=True, check_finite=False)
            lidar_levels, _ = model.lidar_backbone(
                lidar_sites, defer_diagnostics=True)
            _, _, hits, builds = cache.get_batch(
                batch["frame_id"], split, radar_levels[-1], lidar_levels[-1],
                output.confidence_region, batch["clean_lidar"],
                batch["clean_lidar_valid"])
            if hits or builds != len(batch["frame_id"]):
                raise RuntimeError("Stage-I cache build did not cover the requested batch")
            built += builds
    if built != len(pending):
        raise RuntimeError(f"Stage-I cache {split} is incomplete: {built}/{len(pending)}")
    print(f"{split}: {total} frames, {reused} reused, {built} built", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers cannot be negative")
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("train limit must be positive")
    if args.val_limit is not None and args.val_limit < 1:
        parser.error("validation limit must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    if checkpoint.get("stage1_interface") != interface_version(config):
        parser.error("checkpoint is not a matching bottleneck-support Stage-I checkpoint")
    radar_variant = checkpoint.get("data", {}).get("radar_variant")
    if not radar_variant:
        parser.error("checkpoint has no radar variant provenance")
    datasets = {split: VoDStage1Dataset(args.vod_root, split, radar_variant=radar_variant)
                for split in ("train", "val")}
    cache = Stage1StaticGeometryCache(
        args.cache_root, config,
        [frame for dataset in datasets.values() for frame in dataset.frames])
    model = BottleneckSupportStage1(config).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    print(f"Stage-I cache: checkpoint epoch {checkpoint['epoch']}, "
          f"radar={radar_variant}, root={args.cache_root}", flush=True)
    for split, dataset in datasets.items():
        limit = args.train_limit if split == "train" else args.val_limit
        if limit is not None:
            dataset.frames = dataset.frames[:limit]
        prebuild_split(model, cache, dataset, split,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers, device=args.device)


if __name__ == "__main__":
    main()
