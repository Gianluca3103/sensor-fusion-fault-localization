"""Train the deterministic Stage-II model from frozen deployed radar-only Stage I."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from scripts.visualize_stage1_confidence_cloud import load_encoder
from .config import Stage2Config
from .losses import reconstruction_loss
from .metrics import Stage2MetricAccumulator
from .reconstruction_model import RadarLidarStage2
from .voxel_target import make_targets


def _batch_to(batch: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _loader(root, split, radar_variant, batch_size, workers, limit, shuffle):
    dataset = VoDStage1Dataset(root, split, radar_variant=radar_variant)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, collate_fn=collate_stage1,
                      pin_memory=True, persistent_workers=workers > 0)


def _frozen_features(stage1, batch):
    with torch.no_grad():
        return stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)


def evaluate(model, stage1, loader, config, device):
    model.eval()
    metrics = Stage2MetricAccumulator()
    loss_sum = {"total": 0., "occupancy": 0., "offset": 0.}
    count = 0
    with torch.no_grad():
        for raw in tqdm(loader, desc="Stage II validation", dynamic_ncols=True, leave=False):
            batch = _batch_to(raw, device)
            evidence = _frozen_features(stage1, batch)
            output = model(evidence, stage1.config.grid)
            target = make_targets(output.domain, batch["clean_lidar"], batch["clean_lidar_valid"],
                                  free_ray_tolerance_m=config.free_ray_tolerance_m)
            losses = reconstruction_loss(output, target, config)
            for name in loss_sum:
                loss_sum[name] += float(losses[name])
            metrics.add(output, target, batch["clean_lidar"], batch["clean_lidar_valid"],
                        occupancy_threshold=config.occupancy_threshold)
            count += 1
    report = metrics.summary()
    report["loss"] = {key: value/max(count,1) for key,value in loss_sum.items()}
    return report


def _checkpoint(path, model, optimizer, epoch, config, stage1_path, radar_variant,
                report, best_score):
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "config": config.as_dict(), "stage1_checkpoint": str(Path(stage1_path).resolve()),
             "radar_variant": radar_variant, "validation": report, "best_score": best_score,
             "diffusion_enabled": False}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default="configs/radar_lidar_stage2_small.json")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1 or args.validate_every < 1:
        parser.error("Epoch, batch, accumulation and validation intervals must be positive")
    config = Stage2Config.from_json(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage1, saved_stage1 = load_encoder(Path(args.stage1_checkpoint), args.device)
    trained_variant = saved_stage1.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != args.radar_variant:
        raise ValueError(f"Stage-I checkpoint used {trained_variant}, not {args.radar_variant}")
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    if len(stage1.config.channels) != 4:
        raise ValueError("Stage-I checkpoint must provide S1–S4 radar-only features")
    model = RadarLidarStage2(stage1.config.channels, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    start_epoch, best_score = 1, -float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["config"] != config.as_dict() or state["stage1_checkpoint"] != str(Path(args.stage1_checkpoint).resolve()):
            raise ValueError("Resume configuration or frozen Stage-I checkpoint differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
    train_loader = _loader(args.vod_root, "train", args.radar_variant,
                           args.batch_size, args.num_workers, args.train_limit, True)
    val_loader = _loader(args.vod_root, "val", args.radar_variant,
                         args.batch_size, args.num_workers, args.val_limit, False)
    print(f"Stage II: train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"Stage-I frozen=True diffusion=False grid={stage1.config.grid.size_xyz}", flush=True)
    for epoch in range(start_epoch, args.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {name: torch.zeros((), device=args.device) for name in
                  ("total", "occupancy", "offset", "positive_sites", "known_free_sites", "unknown_sites")}
        bar = tqdm(train_loader, desc=f"Stage II {epoch:03d}/{args.epochs:03d}",
                   dynamic_ncols=True, leave=False)
        for step, raw in enumerate(bar, 1):
            batch = _batch_to(raw, args.device)
            evidence = _frozen_features(stage1, batch)
            output = model(evidence, stage1.config.grid)
            target = make_targets(output.domain, batch["clean_lidar"], batch["clean_lidar_valid"],
                                  free_ray_tolerance_m=config.free_ray_tolerance_m)
            losses = reconstruction_loss(output, target, config)
            (losses["total"] / args.grad_accum_steps).backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for name in totals:
                totals[name] += losses[name].detach()
            if step % 10 == 0 or step == len(train_loader):
                bar.set_postfix(loss=f"{float(totals['total']/step):.4f}",
                                candidates=output.metadata["expanded_candidate_sites"])
        train = {name: float(value/max(len(train_loader),1)) for name,value in totals.items()}
        report = None
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            report = evaluate(model, stage1, val_loader, config, args.device)
            metric = (report["occupancy"]["iou"] if config.checkpoint_metric == "occupancy_iou" else
                      report["geometry"]["0.2m" if config.checkpoint_metric == "geom_f1_0.2m" else "0.5m"]["f1"])
            if metric > best_score:
                best_score = metric
                _checkpoint(output_root / "best_geom.ckpt", model, optimizer, epoch, config,
                            args.stage1_checkpoint, args.radar_variant, report, best_score)
        _checkpoint(output_root / "last.ckpt", model, optimizer, epoch, config,
                    args.stage1_checkpoint, args.radar_variant, report, best_score)
        record = {"epoch": epoch, "train": train, "validation": report, "best_score": best_score}
        with (output_root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train {train['total']:.4f} "
              f"occ {train['occupancy']:.4f} offset {train['offset']:.4f}", flush=True)
        if report:
            occ, geom = report["occupancy"], report["geometry"]
            print(f"  val occ P/R/F1/IoU {occ['precision']:.3f}/{occ['recall']:.3f}/"
                  f"{occ['f1']:.3f}/{occ['iou']:.3f} | point F1 .1/.2/.5m "
                  f"{geom['0.1m']['f1']:.3f}/{geom['0.2m']['f1']:.3f}/{geom['0.5m']['f1']:.3f} "
                  f"| coverage {report['candidate_point_coverage']:.3f} "
                  f"| sites {report['candidate_sites']} predicted {report['predicted_sites']}", flush=True)


if __name__ == "__main__":
    main()
