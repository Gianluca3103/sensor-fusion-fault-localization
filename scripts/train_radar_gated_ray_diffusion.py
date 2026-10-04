"""Train radar-gated range-view diffusion from full-scan VoD fault artifacts.

Tiles are chosen around measured radar returns using train inputs only. Clean
LiDAR supplies loss targets; it never chooses a tile, proposal, or edit mask.
The trained checkpoint is a new architecture and cannot load old range-view
or SVEFusion detector weights.
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
from models.two_stage_reconstruction_head.diffusion_process.ray_view_diffusion import (
    RadarGatedRayDiffusion, calibrate_reliability_threshold,
    project_lidar_tile,
)
from models.two_stage_reconstruction_head.range_view.geometry import (
    RangeGeometry, angular_indices,
)
from models.two_stage_reconstruction_head.ray_depth_attention import (
    RayDepthBlueprintModel,
)
from models.two_stage_reconstruction_head.ray_depth_queries import ray_tile_indices
from models.two_stage_reconstruction_head.ray_depth_training import (
    ray_depth_blueprint_loss,
)


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    batch_size: int
    tile_rows: int
    tile_cols: int
    width: int
    hidden: int
    timesteps: int
    max_correction_m: float
    learning_rate: float
    blueprint_loss_weight: float
    target_support_precision: float
    min_calibration_predictions: int
    seed: int


def choose_radar_tile(
    geometry: RangeGeometry, radar: torch.Tensor,
    radar_valid: torch.Tensor, grid: EncoderGrid,
    tile_rows: int, tile_cols: int, rng: random.Random,
) -> tuple[int, int]:
    """Choose one tile around a measured, in-grid radar return if possible."""
    height, width = geometry.shape
    if not (1 <= tile_rows <= height and 4 <= tile_cols <= width):
        raise ValueError("Tile dimensions must fit the calibrated scan")
    xyz = radar[0, radar_valid[0], :3].detach().cpu().numpy()
    if len(xyz):
        minimum, maximum = np.asarray(grid.minimum_xyz), np.asarray(grid.maximum_xyz)
        inside_grid = np.all((xyz >= minimum) & (xyz < maximum), axis=1)
        row, col, _range, angular_valid = angular_indices(
            xyz, geometry, require_beam_match=False)
        candidates = np.flatnonzero(inside_grid & angular_valid)
    else:
        candidates = np.empty(0, dtype=np.int64)
    if len(candidates):
        # Sample spatial regions, not individual returns: a dense road patch
        # should not crowd out a small radar-supported object during training.
        local = np.floor((xyz[candidates] - minimum) /
                         np.asarray(grid.voxel_size_xyz)).astype(np.int64)
        _, representatives = np.unique(local, axis=0, return_index=True)
        chosen = int(candidates[representatives[rng.randrange(len(representatives))]])
        centre_row, centre_col = int(row[chosen]), int(col[chosen])
    else:
        centre_row, centre_col = rng.randrange(height), rng.randrange(width)
    return (max(0, min(centre_row - tile_rows // 2, height - tile_rows)),
            max(0, min(centre_col - tile_cols // 2, width - tile_cols)))


def _paths(root: Path, split: str, limit: int | None) -> list[Path]:
    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Missing {split} samples: {split_root}")
    paths = sorted(split_root.rglob("*.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No {split} NPZ artifacts under {split_root}")
    return paths


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _one_batch(
    batch: dict, *, geometry: RangeGeometry, grid: EncoderGrid,
    blueprint_model: RayDepthBlueprintModel,
    diffusion: RadarGatedRayDiffusion, settings: TrainingSettings,
    rng: random.Random, calibrate: bool,
) -> dict[str, torch.Tensor]:
    row_start, col_start = choose_radar_tile(
        geometry, batch["radar"], batch["radar_valid"], grid,
        settings.tile_rows, settings.tile_cols, rng,
    )
    rows, cols = ray_tile_indices(
        geometry, row_start=row_start, row_stop=row_start + settings.tile_rows,
        col_start=col_start, col_stop=col_start + settings.tile_cols,
        batch_size=len(batch["radar"]), device=batch["radar"].device,
    )
    # The encoder has a separate teacher branch, but the clean scan is omitted
    # from this forward call. The training targets below read it afterwards.
    blueprint = blueprint_model(
        batch["radar"], batch["radar_valid"],
        batch["observed_lidar"], batch["observed_lidar_valid"], rows, cols,
    )
    blueprint_losses = ray_depth_blueprint_loss(
        blueprint, geometry, batch["clean_lidar"], batch["clean_lidar_valid"],
    )
    diffusion_losses = diffusion.training_loss(
        blueprint, batch["observed_lidar"], batch["observed_lidar_valid"],
        batch["clean_lidar"], batch["clean_lidar_valid"],
        (settings.tile_rows, settings.tile_cols),
    )
    total = (settings.blueprint_loss_weight * blueprint_losses["loss"]
             + diffusion_losses["loss"])
    result = {
        "loss": total,
        "blueprint": blueprint_losses["loss"],
        "coverage": blueprint_losses["coverage"],
        "diffusion": diffusion_losses["loss"],
        "supported": diffusion_losses["radar_supported_rays"],
        "correctable": diffusion_losses["correctable_rays"],
    }
    if calibrate:
        condition = diffusion.prepare_condition(
            blueprint, batch["observed_lidar"], batch["observed_lidar_valid"],
            (settings.tile_rows, settings.tile_cols),
        )
        clean_depth, _, clean_hit = project_lidar_tile(
            geometry, rows, cols, batch["clean_lidar"], batch["clean_lidar_valid"])
        clean_depth = clean_depth.reshape_as(condition.base_depth_m)
        clean_hit = clean_hit.reshape_as(condition.proposal_mask)
        supported = condition.proposal_mask
        correctable = (clean_hit &
            ((clean_depth - condition.base_depth_m).abs() <= settings.max_correction_m))
        result["calibration_scores"] = condition.reliability_logits.sigmoid()[supported].detach()
        result["calibration_labels"] = correctable[supported].detach()
    return result


def _run_epoch(loader: DataLoader, *, geometry: RangeGeometry,
               grid: EncoderGrid, blueprint_model: RayDepthBlueprintModel,
               diffusion: RadarGatedRayDiffusion, settings: TrainingSettings,
               optimizer: torch.optim.Optimizer | None,
               device: torch.device, seed: int,
               epoch: int, label: str) -> dict[str, float]:
    training = optimizer is not None
    blueprint_model.train(training)
    diffusion.train(training)
    rng = random.Random(seed)
    totals = {key: 0.0 for key in
              ("loss", "blueprint", "coverage", "diffusion", "supported", "correctable")}
    calibration_scores: list[torch.Tensor] = []
    calibration_labels: list[torch.Tensor] = []
    progress = tqdm(loader, desc=f"{label} {epoch}/{settings.epochs}", leave=False)
    for count, raw_batch in enumerate(progress, start=1):
        batch = _device_batch(raw_batch, device)
        with torch.set_grad_enabled(training):
            metrics = _one_batch(
                batch, geometry=geometry, grid=grid,
                blueprint_model=blueprint_model, diffusion=diffusion,
                settings=settings, rng=rng, calibrate=not training,
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(blueprint_model.parameters()) + list(diffusion.parameters()), 1.0)
                optimizer.step()
        for key in totals:
            totals[key] += float(metrics[key].detach())
        if not training:
            calibration_scores.append(metrics["calibration_scores"].cpu())
            calibration_labels.append(metrics["calibration_labels"].cpu())
        progress.set_postfix(loss=f"{totals['loss'] / count:.3f}",
                             support=f"{totals['supported'] / count:.1f}")
    average = {key: value / max(len(loader), 1) for key, value in totals.items()}
    if not training:
        calibration = calibrate_reliability_threshold(
            torch.cat(calibration_scores), torch.cat(calibration_labels),
            minimum_precision=settings.target_support_precision,
            minimum_predictions=settings.min_calibration_predictions,
        )
        average.update({f"calibration_{key}": float(value)
                        for key, value in calibration.items()})
    return average


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True,
                        help="Full-scan fault samples with train/ and val/ directories")
    parser.add_argument("--vod-root", type=Path, required=True,
                        help="VoD public root containing verified 20-scan radar")
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
                        help="Resume model and optimizer from a last_checkpoint.pt")
    parser.add_argument("--pretrained-blueprint", type=Path,
                        help="Initialize from train_ray_depth_blueprint.py")
    parser.add_argument("--freeze-blueprint-epochs", type=int, default=0,
                        help="Train diffusion alone for this many initial epochs")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--radar-height-filter", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Keep radar only between the observed faulty LiDAR's per-frame Z extrema")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=4)
    parser.add_argument("--tile-cols", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--max-correction-m", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--blueprint-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-support-precision", type=float, default=0.9,
                        help="Minimum empirical depth-correctness precision for the inference gate")
    parser.add_argument("--min-calibration-predictions", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.tile_rows < 1 or
            args.tile_cols < 4 or args.width < 8 or args.width % 8 or
            args.hidden < 8 or args.hidden % 8 or args.timesteps < 2 or
            args.max_correction_m <= 0 or args.learning_rate <= 0 or
            args.blueprint_loss_weight < 0 or args.validate_every < 1 or
            not 0 < args.target_support_precision <= 1 or
            args.min_calibration_predictions < 1 or
            args.freeze_blueprint_epochs < 0 or
            args.num_workers < 0 or
            (args.train_limit is not None and args.train_limit < 1) or
            (args.val_limit is not None and args.val_limit < 1)):
        parser.error("Invalid model, tile, optimizer, or dataset limit setting")
    if args.resume is not None and args.pretrained_blueprint is not None:
        parser.error("Use --resume or --pretrained-blueprint, not both")
    if (args.freeze_blueprint_epochs and args.pretrained_blueprint is None and
            args.resume is None):
        parser.error("Frozen blueprint epochs require --pretrained-blueprint")
    return args


def main() -> None:
    args = _args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    geometry = RangeGeometry.from_json(args.geometry)
    if args.tile_rows > geometry.shape[0] or args.tile_cols > geometry.shape[1]:
        raise ValueError("Training tile exceeds calibrated range-view dimensions")
    # Five depth slots are produced by the default ray-depth proposal builder.
    if args.tile_rows * args.tile_cols * 5 > 2048:
        raise ValueError("Tile exceeds the blueprint's 2048-candidate limit")
    settings = TrainingSettings(
        args.epochs, args.batch_size, args.tile_rows, args.tile_cols,
        args.width, args.hidden, args.timesteps, args.max_correction_m,
        args.learning_rate, args.blueprint_loss_weight,
        args.target_support_precision, args.min_calibration_predictions,
        args.seed,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    grid = EncoderGrid()
    blueprint_model = RayDepthBlueprintModel(
        geometry, grid, width=settings.width, history_scans=20).to(device)
    diffusion = RadarGatedRayDiffusion(
        geometry, blueprint_width=settings.width, hidden=settings.hidden,
        timesteps=settings.timesteps,
        max_correction_m=settings.max_correction_m).to(device)
    pretrained_blueprint = None
    if args.pretrained_blueprint is not None:
        pretrained = torch.load(
            args.pretrained_blueprint, map_location=device, weights_only=False)
        if pretrained.get("stage") != "blueprint_pretraining":
            raise ValueError("Pretrained checkpoint is not a ray-depth blueprint")
        if pretrained.get("geometry_parameters") != asdict(geometry):
            raise ValueError("Pretrained blueprint uses different calibrated geometry")
        if pretrained["settings"]["width"] != settings.width:
            raise ValueError("Pretrained blueprint uses a different width")
        if (pretrained["radar_variant"] != args.radar_variant or
                pretrained["radar_height_filter"] != args.radar_height_filter):
            raise ValueError("Pretrained blueprint uses different radar inputs")
        blueprint_model.load_state_dict(pretrained["blueprint"])
        pretrained_blueprint = str(args.pretrained_blueprint.resolve())
    training = CrossModalVoDDataset(
        _paths(args.samples_root, "train", args.train_limit),
        args.vod_root, radar_variant=args.radar_variant, include_clean=True,
        radar_height_filter=args.radar_height_filter)
    validation = CrossModalVoDDataset(
        _paths(args.samples_root, "val", args.val_limit),
        args.vod_root, radar_variant=args.radar_variant, include_clean=True,
        radar_height_filter=args.radar_height_filter)
    train_loader = DataLoader(training, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, collate_fn=collate_cross_modal)
    val_loader = DataLoader(validation, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, collate_fn=collate_cross_modal)
    optimizer = torch.optim.AdamW(
        list(blueprint_model.parameters()) + list(diffusion.parameters()),
        lr=settings.learning_rate, weight_decay=1e-4)
    start_epoch = 1
    last_calibration = None
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (saved.get("freeze_blueprint_epochs", 0) !=
                args.freeze_blueprint_epochs):
            raise ValueError("Resume checkpoint disagrees on frozen blueprint epochs")
        if saved.get("geometry_parameters") != asdict(geometry):
            raise ValueError("Resume checkpoint uses different calibrated LiDAR geometry")
        saved_settings = saved["settings"]
        for key in ("tile_rows", "tile_cols", "width", "hidden", "timesteps",
                    "max_correction_m"):
            if saved_settings[key] != getattr(settings, key):
                raise ValueError(f"Resume checkpoint disagrees on {key}")
        if saved.get("radar_variant", "radar_20frames_verified") != args.radar_variant:
            raise ValueError("Resume checkpoint disagrees on radar_variant")
        if bool(saved.get("radar_height_filter", False)) != args.radar_height_filter:
            raise ValueError("Resume checkpoint disagrees on radar_height_filter")
        blueprint_model.load_state_dict(saved["blueprint"])
        diffusion.load_state_dict(saved["diffusion"])
        optimizer.load_state_dict(saved["optimizer"])
        last_calibration = saved.get("calibration")
        pretrained_blueprint = saved.get("pretrained_blueprint")
        start_epoch = int(saved["epoch"]) + 1
        if start_epoch > settings.epochs:
            raise ValueError("Resume checkpoint already reached the requested epoch count")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "training_config.json").write_text(json.dumps({
        "settings": asdict(settings), "geometry": str(args.geometry.resolve()),
        "geometry_parameters": asdict(geometry),
        "radar_variant": args.radar_variant,
        "radar_height_filter": args.radar_height_filter,
        "pretrained_blueprint": pretrained_blueprint,
        "freeze_blueprint_epochs": args.freeze_blueprint_epochs,
        "train_examples": len(training), "val_examples": len(validation),
    }, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, settings.epochs + 1):
        blueprint_model.requires_grad_(epoch > args.freeze_blueprint_epochs)
        train_metrics = _run_epoch(train_loader, geometry=geometry, grid=grid,
            blueprint_model=blueprint_model, diffusion=diffusion,
            settings=settings, optimizer=optimizer, device=device,
            seed=args.seed + epoch, epoch=epoch, label="train")
        message = {"epoch": epoch, "train": train_metrics}
        if epoch % args.validate_every == 0 or epoch == settings.epochs:
            val_metrics = _run_epoch(val_loader, geometry=geometry, grid=grid,
                blueprint_model=blueprint_model, diffusion=diffusion,
                settings=settings, optimizer=None, device=device,
                seed=args.seed + 10000, epoch=epoch, label="val")
            message["val"] = val_metrics
            last_calibration = {
                key.removeprefix("calibration_"): value
                for key, value in val_metrics.items()
                if key.startswith("calibration_")
            }
        print(json.dumps(message), flush=True)
        checkpoint = {
            "epoch": epoch, "blueprint": blueprint_model.state_dict(),
            "diffusion": diffusion.state_dict(), "optimizer": optimizer.state_dict(),
            "settings": asdict(settings), "geometry": str(args.geometry.resolve()),
            "geometry_parameters": asdict(geometry),
            "radar_variant": args.radar_variant,
            "radar_height_filter": args.radar_height_filter,
            "pretrained_blueprint": pretrained_blueprint,
            "freeze_blueprint_epochs": args.freeze_blueprint_epochs,
            "calibration": last_calibration,
        }
        temporary = args.output_root / "last_checkpoint.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.output_root / "last_checkpoint.pt")


if __name__ == "__main__":
    main()
