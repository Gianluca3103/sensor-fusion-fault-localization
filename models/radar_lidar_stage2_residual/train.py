"""Train additive Stage II with frozen radar-only Stage I and paired faulty LiDAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage2.voxel_target import decode_centroids
from scripts.visualize_stage1_confidence_cloud import load_encoder
from .config import ResidualStage2Config
from .data import PairedFaultDataset, collate_paired
from .losses import residual_loss
from .model import ResidualRadarLidarStage2
from .targets import make_residual_targets


def _loader(args, split, limit, shuffle):
    dataset = PairedFaultDataset(args.vod_root, args.fault_samples_root, split,
                                 radar_variant=args.radar_variant, fault_pattern=args.fault_pattern)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, collate_fn=collate_paired,
                      pin_memory=True, persistent_workers=args.num_workers > 0)


def _to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


class ResidualMetrics:
    def __init__(self):
        self.count = dict(addition_sites=0, predicted_sites=0, voxel_tp=0,
                          known_free_fp=0, unknown_pred=0, observed_pred=0,
                          represented_clean_sites=0, candidate_sites=0,
                          point_pred=0, point_target=0, point_match_pred=0, point_match_target=0,
                          missing_clean_points=0, global_match_pred=0, global_match_target=0)

    def add(self, output, target, threshold, clean, clean_valid, faulty, faulty_valid):
        row = self.count
        selected = (output.addition_probability > threshold) & output.coverage.may_add
        row["candidate_sites"] += len(selected)
        row["addition_sites"] += int(target.addition.sum())
        row["represented_clean_sites"] += int(target.clean.occupied.sum())
        row["predicted_sites"] += int(selected.sum())
        row["voxel_tp"] += int((selected & target.addition).sum())
        row["known_free_fp"] += int((selected & target.known_free).sum())
        row["unknown_pred"] += int((selected & ~target.supervised).sum())
        row["observed_pred"] += int((selected & target.observed).sum())
        xyz = decode_centroids(output.domain, output.predicted_offsets).detach().cpu().numpy()
        true_xyz = target.clean.clean_centroid_xyz.detach().cpu().numpy()
        ids = output.domain.coordinates[:, 0].detach().cpu().numpy()
        p_mask = selected.detach().cpu().numpy()
        t_mask = target.addition.detach().cpu().numpy()
        clean_np = clean.detach().cpu().numpy()
        clean_mask = clean_valid.detach().cpu().numpy()
        faulty_np = faulty.detach().cpu().numpy()
        faulty_mask = faulty_valid.detach().cpu().numpy()
        for batch in range(clean.shape[0]):
            p = xyz[(ids == batch) & p_mask]
            t = true_xyz[(ids == batch) & t_mask]
            row["point_pred"] += len(p)
            row["point_target"] += len(t)
            if len(p) and len(t):
                row["point_match_pred"] += int((cKDTree(t).query(p, workers=1)[0] <= .2).sum())
                row["point_match_target"] += int((cKDTree(p).query(t, workers=1)[0] <= .2).sum())
            # This global denominator includes missing clean returns outside
            # the radar candidate domain. Clean is used for validation only.
            all_clean = clean_np[batch, clean_mask[batch], :3]
            surviving = faulty_np[batch, faulty_mask[batch], :3]
            if len(surviving) and len(all_clean):
                missing = all_clean[cKDTree(surviving).query(all_clean, workers=1)[0] > .05]
            else:
                missing = all_clean
            row["missing_clean_points"] += len(missing)
            if len(p) and len(missing):
                row["global_match_pred"] += int((cKDTree(missing).query(p, workers=1)[0] <= .2).sum())
                row["global_match_target"] += int((cKDTree(p).query(missing, workers=1)[0] <= .2).sum())

    def summary(self):
        row = dict(self.count)
        def ratio(a, b):
            return row[a] / row[b] if row[b] else 0.
        precision = ratio("point_match_pred", "point_pred")
        recall = ratio("point_match_target", "point_target")
        row["addition_precision_0.2m"] = precision
        row["addition_recall_0.2m"] = recall
        row["addition_f1_0.2m"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        row["voxel_addition_precision"] = ratio("voxel_tp", "predicted_sites")
        row["voxel_addition_recall"] = ratio("voxel_tp", "addition_sites")
        row["unknown_prediction_fraction"] = ratio("unknown_pred", "predicted_sites")
        row["observed_prediction_fraction"] = ratio("observed_pred", "predicted_sites")
        global_precision = ratio("global_match_pred", "point_pred")
        global_recall = ratio("global_match_target", "missing_clean_points")
        row["global_missing_precision_0.2m"] = global_precision
        row["global_missing_recall_0.2m"] = global_recall
        row["global_missing_f1_0.2m"] = (2 * global_precision * global_recall /
                                           (global_precision + global_recall)
                                           if global_precision + global_recall else 0.)
        return row


def _run_epoch(model, stage1, loader, config, device, optimizer=None, grad_accum=1, epoch=0, epochs=0):
    training = optimizer is not None
    model.train(training)
    totals = {key: torch.zeros((), device=device) for key in ("total", "occupancy", "offset")}
    metrics = ResidualMetrics() if not training else None
    if training:
        optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=(f"Residual II {epoch:03d}/{epochs:03d}" if training else "Residual II validation"),
               dynamic_ncols=True, leave=False)
    for step, raw in enumerate(bar, 1):
        batch = _to_device(raw, device)
        with torch.no_grad():
            evidence = stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)
        with torch.set_grad_enabled(training):
            output = model(evidence, stage1.config.grid,
                           batch["faulty_lidar"], batch["faulty_lidar_valid"], batch["fault_region"])
            target = make_residual_targets(output.domain, batch["clean_lidar"],
                                           batch["clean_lidar_valid"], output.coverage,
                                           free_ray_tolerance_m=config.free_ray_tolerance_m)
            losses = residual_loss(output, target, config)
            if training:
                (losses["total"] / grad_accum).backward()
                if step % grad_accum == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += losses[key].detach()
        if metrics is not None:
            metrics.add(output, target, config.occupancy_threshold,
                        batch["clean_lidar"], batch["clean_lidar_valid"],
                        batch["faulty_lidar"], batch["faulty_lidar_valid"])
        if training and (step % 10 == 0 or step == len(loader)):
            bar.set_postfix(loss=f"{float(totals['total']/step):.4f}",
                            additions=int(target.addition.sum()), candidates=len(output.domain.coordinates))
    report = {"loss": {key: float(value / max(len(loader), 1)) for key, value in totals.items()}}
    if metrics is not None:
        report.update(metrics.summary())
    return report


def _save(path, model, optimizer, epoch, config, args, report, best):
    state = {"model_type": "stage2_residual_faulty_conditioned_v1", "epoch": epoch,
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "config": config.as_dict(), "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
             "radar_variant": args.radar_variant, "fault_samples_root": str(Path(args.fault_samples_root).resolve()),
             "fault_pattern": args.fault_pattern, "validation": report, "best_score": best,
             "diffusion_enabled": False}
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
    parser.add_argument("--config", default="configs/radar_lidar_stage2_residual.json")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=.0002)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.grad_accum_steps, args.validate_every) < 1:
        parser.error("Epochs, batch size, accumulation and validation interval must be positive")
    config = ResidualStage2Config.from_json(args.config)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    stage1, stage1_saved = load_encoder(Path(args.stage1_checkpoint), args.device)
    trained_variant = stage1_saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != args.radar_variant:
        raise ValueError("Stage-I checkpoint was trained with a different radar stack")
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    model = ResidualRadarLidarStage2(stage1.config.channels, config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    start, best = 1, -float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("model_type") != "stage2_residual_faulty_conditioned_v1"
                or saved["config"] != config.as_dict()
                or saved["stage1_checkpoint"] != str(Path(args.stage1_checkpoint).resolve())
                or saved["radar_variant"] != args.radar_variant
                or saved["fault_samples_root"] != str(Path(args.fault_samples_root).resolve())
                or saved["fault_pattern"] != args.fault_pattern):
            raise ValueError("Residual resume checkpoint and training inputs differ")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = int(saved["epoch"]) + 1, float(saved["best_score"])
    train = _loader(args, "train", args.train_limit, True)
    val = _loader(args, "val", args.val_limit, False)
    print(f"Residual Stage II: train={len(train.dataset)} val={len(val.dataset)} "
          f"Stage-I frozen=True diffusion=False grid={stage1.config.grid.size_xyz}", flush=True)
    for epoch in range(start, args.epochs + 1):
        training = _run_epoch(model, stage1, train, config, args.device, optimizer,
                              args.grad_accum_steps, epoch, args.epochs)
        validation = None
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            validation = _run_epoch(model, stage1, val, config, args.device)
            score = validation["global_missing_f1_0.2m"]
            if score > best:
                best = score
                _save(root / "best_global_missing.ckpt", model, optimizer, epoch, config, args, validation, best)
        _save(root / "last.ckpt", model, optimizer, epoch, config, args, validation, best)
        with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "train": training, "validation": validation,
                                     "best_global_missing_f1_0.2m": best}) + "\n")
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | train {training['loss']['total']:.4f}", flush=True)
        if validation:
            print(f"  val addition 20cm P/R/F1 "
                  f"{validation['addition_precision_0.2m']:.3f}/"
                  f"{validation['addition_recall_0.2m']:.3f}/"
                  f"{validation['addition_f1_0.2m']:.3f} | global missing F1 "
                  f"{validation['global_missing_f1_0.2m']:.3f} | predicted "
                  f"{validation['predicted_sites']} | unknown "
                  f"{validation['unknown_prediction_fraction']:.3f} | observed "
                  f"{validation['observed_prediction_fraction']:.3f}", flush=True)


if __name__ == "__main__":
    main()
