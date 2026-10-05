"""Jointly train radar-to-LiDAR relation features and absolute-depth diffusion.

A pretrained clean-LiDAR teacher and a paired cross-attention module provide
training-only feature targets. The diffusion condition always comes from radar
and faulty LiDAR. No first-return blueprint or bounded depth correction is
trained in this run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.two_stage_reconstruction_head.cross_modal_data import (
    CrossModalVoDDataset, collate_cross_modal,
)
from models.two_stage_reconstruction_head.cross_modal_encoders import EncoderGrid
from models.two_stage_reconstruction_head.diffusion_process.joint_relation_diffusion import (
    PairedRadarLidarAttention, RadarRelationDiffusion, RadarRelationEncoder,
    joint_relation_loss,
)
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.ray_depth_attention import (
    CleanLidarRelationshipTeacher,
)
from models.two_stage_reconstruction_head.ray_depth_queries import ray_tile_indices
from scripts.train_radar_gated_ray_diffusion import choose_radar_tile, _paths
from scripts.training_progress import record_epoch


@dataclass(frozen=True)
class JointSettings:
    epochs: int
    batch_size: int
    grad_accum_steps: int
    tile_rows: int
    tile_cols: int
    width: int
    hidden: int
    timesteps: int
    learning_rate: float
    alignment_weight: float
    paired_weight: float
    validate_every: int
    seed: int


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _one_batch(
    batch: dict, *, geometry: RangeGeometry, grid: EncoderGrid,
    relation_model: RadarRelationEncoder,
    paired_model: PairedRadarLidarAttention,
    teacher: CleanLidarRelationshipTeacher,
    diffusion: RadarRelationDiffusion, settings: JointSettings,
    rng: random.Random,
) -> dict[str, torch.Tensor]:
    rows_per_scene, cols_per_scene = [], []
    for scene in range(batch["radar"].shape[0]):
        row_start, col_start = choose_radar_tile(
            geometry, batch["radar"][scene:scene + 1],
            batch["radar_valid"][scene:scene + 1], grid,
            settings.tile_rows, settings.tile_cols, rng)
        rows, cols = ray_tile_indices(
            geometry, row_start=row_start,
            row_stop=row_start + settings.tile_rows,
            col_start=col_start, col_stop=col_start + settings.tile_cols,
            device=batch["radar"].device)
        rows_per_scene.append(rows)
        cols_per_scene.append(cols)
    rows = torch.cat(rows_per_scene, dim=0)
    cols = torch.cat(cols_per_scene, dim=0)
    # This path is identical at train and inference: no clean-LiDAR tensor is
    # passed to the relation encoder or diffusion's condition builder.
    relation = relation_model(batch["radar"], batch["radar_valid"], rows, cols)
    diffusion_loss = diffusion.training_loss(
        relation, batch["observed_lidar"], batch["observed_lidar_valid"],
        batch["clean_lidar"], batch["clean_lidar_valid"],
        (settings.tile_rows, settings.tile_cols))
    relation_loss = joint_relation_loss(
        relation, paired_model, teacher,
        batch["clean_lidar"], batch["clean_lidar_valid"],
        (settings.tile_rows, settings.tile_cols),
        alignment_weight=settings.alignment_weight,
        paired_weight=settings.paired_weight)
    return {"loss": diffusion_loss["loss"] + relation_loss["loss"],
            "diffusion": diffusion_loss["loss"],
            "alignment": relation_loss["alignment"],
            "paired": relation_loss["paired"],
            "supported": diffusion_loss["supported_rays"],
            "clean_hits": diffusion_loss["clean_hits"],
            "predicted_hits": diffusion_loss["predicted_hits"],
            "true_hits": diffusion_loss["true_hits"],
            "depth_error_sum_m": diffusion_loss["depth_error_sum_m"],
            "aligned_rays": relation_loss["aligned_rays"],
            "paired_rays": relation_loss["paired_rays"]}


def _run_epoch(
    loader: DataLoader, *, geometry: RangeGeometry, grid: EncoderGrid,
    relation_model: RadarRelationEncoder,
    paired_model: PairedRadarLidarAttention,
    teacher: CleanLidarRelationshipTeacher,
    diffusion: RadarRelationDiffusion,
    settings: JointSettings, optimizer: torch.optim.Optimizer | None,
    device: torch.device, epoch: int, seed: int, label: str,
) -> dict[str, float]:
    training = optimizer is not None
    relation_model.train(training)
    paired_model.train(training)
    diffusion.train(training)
    teacher.eval()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in (
        "loss", "diffusion", "alignment", "paired", "supported",
        "clean_hits", "predicted_hits", "true_hits", "depth_error_sum_m",
        "aligned_rays", "paired_rays")}
    rng = random.Random(seed)
    parameters = (list(relation_model.parameters()) +
                  list(paired_model.parameters()) +
                  list(diffusion.parameters()))
    progress = tqdm(loader, desc=f"{label} {epoch}/{settings.epochs}",
                    leave=False)
    for index, raw in enumerate(progress, start=1):
        batch = _device_batch(raw, device)
        with torch.set_grad_enabled(training):
            result = _one_batch(
                batch, geometry=geometry, grid=grid,
                relation_model=relation_model, paired_model=paired_model,
                teacher=teacher, diffusion=diffusion, settings=settings,
                rng=rng)
            if training:
                group_start = ((index - 1) // settings.grad_accum_steps)
                group_start *= settings.grad_accum_steps
                group_size = min(settings.grad_accum_steps,
                                 len(loader) - group_start)
                (result["loss"] / group_size).backward()
                if (index % settings.grad_accum_steps == 0 or
                        index == len(loader)):
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += float(result[key].detach())
        progress.set_postfix(loss=f"{totals['loss'] / index:.3f}",
                             support=f"{totals['supported'] / index:.1f}")
    average = {key: totals[key] / len(loader)
               for key in ("loss", "diffusion", "alignment", "paired")}
    average.update({key: totals[key] for key in totals if key not in average})
    precision = totals["true_hits"] / max(totals["predicted_hits"], 1)
    recall = totals["true_hits"] / max(totals["clean_hits"], 1)
    average["return_precision"] = precision
    average["return_recall"] = recall
    average["return_f1"] = 2 * precision * recall / max(precision + recall, 1e-8)
    average["depth_mae_m"] = (totals["depth_error_sum_m"] /
                              max(totals["clean_hits"], 1))
    return average


def _summary(message: dict, total_epochs: int) -> str:
    epoch, train = message["epoch"], message["train"]
    line = (f"Epoch {epoch:02d}/{total_epochs} | train loss {train['loss']:.4f}"
            f" (diffusion {train['diffusion']:.4f},"
            f" relation {train['alignment']:.4f})")
    if "val" in message:
        val = message["val"]
        line += (f"\n  val loss {val['loss']:.4f} | return P/R/F1 "
                 f"{val['return_precision']:.1%}/{val['return_recall']:.1%}/"
                 f"{val['return_f1']:.1%} | noisy-depth MAE "
                 f"{val['depth_mae_m']:.2f} m | clean hits "
                 f"{int(val['clean_hits']):,} | paired rays "
                 f"{int(val['paired_rays']):,}")
        if not val["paired_rays"]:
            line += "\n  WARNING: no clean/radar feature pairs in validation tiles"
    if "train_eval" in message:
        audit = message["train_eval"]
        line += (f"\n  frozen train loss {audit['loss']:.4f} | return F1 "
                 f"{audit['return_f1']:.1%}")
    return line


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True,
                        help="teacher_best_checkpoint.pt from clean teacher training")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--radar-height-filter",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Keep off: this gate derives radar input from faulty LiDAR")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--tile-rows", type=int, default=4)
    parser.add_argument("--tile-cols", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--paired-weight", type=float, default=0.1)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--audit-train-at-end",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.grad_accum_steps < 1 or
            args.tile_rows < 1 or args.tile_cols < 4 or args.width < 8 or
            args.width % 8 or args.hidden < 8 or args.hidden % 8 or
            args.timesteps < 2 or args.learning_rate <= 0 or
            args.alignment_weight < 0 or args.paired_weight < 0 or
            args.validate_every < 1 or args.num_workers < 0 or
            (args.train_limit is not None and args.train_limit < 1) or
            (args.val_limit is not None and args.val_limit < 1)):
        parser.error("Invalid model, optimizer, tile, or sample-limit setting")
    return args


def main() -> None:
    args = _arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    geometry = RangeGeometry.from_json(args.geometry)
    if (args.tile_rows > geometry.shape[0] or
            args.tile_cols > geometry.shape[1] or
            args.tile_rows * args.tile_cols * 5 > 2048):
        raise ValueError("Tile exceeds calibrated scan or attention candidate limit")
    settings = JointSettings(
        args.epochs, args.batch_size, args.grad_accum_steps,
        args.tile_rows, args.tile_cols, args.width, args.hidden,
        args.timesteps, args.learning_rate, args.alignment_weight,
        args.paired_weight, args.validate_every, args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    grid = EncoderGrid()
    relation_model = RadarRelationEncoder(
        geometry, grid, width=args.width, history_scans=20).to(device)
    paired_model = PairedRadarLidarAttention(args.width).to(device)
    teacher = CleanLidarRelationshipTeacher(
        geometry, grid, width=args.width, history_scans=20).to(device)
    diffusion = RadarRelationDiffusion(
        geometry, relation_width=args.width, hidden=args.hidden,
        timesteps=args.timesteps).to(device)
    teacher_saved = torch.load(
        args.teacher_checkpoint, map_location=device, weights_only=False)
    if (teacher_saved.get("stage") != "clean_relationship_teacher" or
            teacher_saved.get("geometry_parameters") != asdict(geometry) or
            teacher_saved.get("settings", {}).get("width") != args.width):
        raise ValueError("Clean teacher checkpoint disagrees with geometry or width")
    teacher.load_state_dict(teacher_saved["teacher"])
    teacher.eval().requires_grad_(False)
    training = CrossModalVoDDataset(
        _paths(args.samples_root, "train", args.train_limit), args.vod_root,
        radar_variant=args.radar_variant, include_clean=True,
        radar_height_filter=args.radar_height_filter)
    validation = CrossModalVoDDataset(
        _paths(args.samples_root, "val", args.val_limit), args.vod_root,
        radar_variant=args.radar_variant, include_clean=True,
        radar_height_filter=args.radar_height_filter)
    train_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_cross_modal)
    val_loader = DataLoader(
        validation, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_cross_modal)
    audit_loader = DataLoader(
        training, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_cross_modal)
    optimizer = torch.optim.AdamW(
        list(relation_model.parameters()) + list(paired_model.parameters()) +
        list(diffusion.parameters()), lr=args.learning_rate, weight_decay=1e-4)
    start_epoch, best_val = 1, float("inf")
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (saved.get("stage") != "joint_radar_relation_diffusion_v1" or
                saved.get("geometry_parameters") != asdict(geometry) or
                saved.get("radar_variant") != args.radar_variant or
                saved.get("radar_height_filter") != args.radar_height_filter or
                saved.get("teacher_checkpoint") !=
                str(args.teacher_checkpoint.resolve())):
            raise ValueError("Resume checkpoint belongs to a different model or input setup")
        for key in ("tile_rows", "tile_cols", "width", "hidden", "timesteps",
                    "alignment_weight", "paired_weight"):
            if saved["settings"][key] != getattr(settings, key):
                raise ValueError(f"Resume checkpoint disagrees on {key}")
        old_effective_batch = (saved["settings"]["batch_size"] *
                               saved["settings"]["grad_accum_steps"])
        new_effective_batch = settings.batch_size * settings.grad_accum_steps
        if old_effective_batch != new_effective_batch:
            raise ValueError("Resume checkpoint disagrees on effective batch size")
        relation_model.load_state_dict(saved["relation"])
        paired_model.load_state_dict(saved["paired"])
        diffusion.load_state_dict(saved["diffusion"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best_val = float(saved["best_val_loss"])
        if start_epoch > args.epochs:
            raise ValueError("Resume checkpoint already reached requested epochs")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "training_config.json").write_text(json.dumps({
        "stage": "joint_radar_relation_diffusion_v1",
        "settings": asdict(settings),
        "geometry": str(args.geometry.resolve()),
        "geometry_parameters": asdict(geometry),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_epoch": teacher_saved["epoch"],
        "radar_variant": args.radar_variant,
        "radar_height_filter": args.radar_height_filter,
        "train_examples": len(training), "val_examples": len(validation),
    }, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs + 1):
        train = _run_epoch(
            train_loader, geometry=geometry, grid=grid,
            relation_model=relation_model, paired_model=paired_model,
            teacher=teacher, diffusion=diffusion, settings=settings,
            optimizer=optimizer, device=device, epoch=epoch,
            seed=args.seed + epoch, label="joint train")
        message = {"epoch": epoch, "train": train}
        improved = False
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            val = _run_epoch(
                val_loader, geometry=geometry, grid=grid,
                relation_model=relation_model, paired_model=paired_model,
                teacher=teacher, diffusion=diffusion, settings=settings,
                optimizer=None, device=device, epoch=epoch,
                seed=args.seed + 10000, label="joint val")
            message["val"] = val
            improved = val["loss"] < best_val
            if improved:
                best_val = val["loss"]
            if epoch == args.epochs and args.audit_train_at_end:
                message["train_eval"] = _run_epoch(
                    audit_loader, geometry=geometry, grid=grid,
                    relation_model=relation_model, paired_model=paired_model,
                    teacher=teacher, diffusion=diffusion, settings=settings,
                    optimizer=None, device=device, epoch=epoch,
                    seed=args.seed + 10000, label="frozen train audit")
        record_epoch(args.output_root, message, _summary(message, args.epochs))
        checkpoint = {
            "stage": "joint_radar_relation_diffusion_v1", "epoch": epoch,
            "relation": relation_model.state_dict(),
            "paired": paired_model.state_dict(),
            "diffusion": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "settings": asdict(settings),
            "geometry_parameters": asdict(geometry),
            "radar_variant": args.radar_variant,
            "radar_height_filter": args.radar_height_filter,
            "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
            "best_val_loss": best_val, "validation": message.get("val"),
            "train_eval": message.get("train_eval"),
        }
        temporary = args.output_root / "last_checkpoint.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.output_root / "last_checkpoint.pt")
        if improved:
            temporary = args.output_root / "best_checkpoint.tmp"
            torch.save(checkpoint, temporary)
            os.replace(temporary, args.output_root / "best_checkpoint.pt")


if __name__ == "__main__":
    main()
