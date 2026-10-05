"""Train a clean geometry teacher, then a radar-only relationship blueprint.

The teacher attends to clean LiDAR on radar-derived ray-depth queries. The
deployed student receives radar alone; clean LiDAR is used only after its
forward pass as a supervised target and a detached teacher representation.
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
    project_lidar_tile,
)
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.ray_depth_attention import (
    CleanLidarRelationshipTeacher, RayDepthBlueprint, RayDepthBlueprintModel,
)
from models.two_stage_reconstruction_head.ray_depth_queries import (
    propose_ray_depth_queries, ray_tile_indices,
)
from models.two_stage_reconstruction_head.ray_depth_training import (
    clean_first_return_targets, ray_depth_blueprint_loss,
    relationship_feature_loss,
)
from scripts.train_radar_gated_ray_diffusion import choose_radar_tile, _paths
from scripts.training_progress import blueprint_summary, record_epoch


@dataclass(frozen=True)
class BlueprintSettings:
    epochs: int
    batch_size: int
    grad_accum_steps: int
    tile_rows: int
    tile_cols: int
    width: int
    learning_rate: float
    validate_every: int
    seed: int
    teacher_epochs: int
    distill_weight: float


def _ray_counts(
    blueprint: RayDepthBlueprint, geometry: RangeGeometry,
    observed: torch.Tensor, observed_valid: torch.Tensor,
    clean: torch.Tensor, clean_valid: torch.Tensor,
) -> dict[str, float]:
    """Score missing, radar-supported rays without using clean input for proposals."""
    rows, cols = blueprint.queries.rows, blueprint.queries.cols
    clean_depth, clean_hit = clean_first_return_targets(
        geometry, rows, cols, clean, clean_valid)
    _, _, observed_hit = project_lidar_tile(
        geometry, rows, cols, observed, observed_valid)
    slots = blueprint.queries.depths_m.shape[-1]
    radar_support = (
        blueprint.queries.valid & (blueprint.evidence_weights[..., 0] > 0)
    ).any(-1)
    eligible = radar_support & ~observed_hit
    selected = blueprint.first_return_logits.argmax(-1)
    predicted = eligible & (selected != slots)
    predicted_depth = (
        blueprint.queries.depths_m + blueprint.depth_residual_m
    ).gather(-1, selected.clamp(max=slots - 1)[..., None]).squeeze(-1)
    depth_error = (predicted_depth - clean_depth).abs()
    matched = predicted & clean_hit
    correct_3m = matched & (depth_error <= 3.0)
    return {
        "eligible": float(eligible.sum()),
        "clean_hits": float((eligible & clean_hit).sum()),
        "predicted_hits": float(predicted.sum()),
        "matched_hits": float(matched.sum()),
        "correct_3m": float(correct_3m.sum()),
        "depth_error_sum_m": float(depth_error[matched].sum()),
    }


def _run_epoch(
    loader: DataLoader, *, model: RayDepthBlueprintModel,
    teacher: CleanLidarRelationshipTeacher,
    geometry: RangeGeometry, grid: EncoderGrid, settings: BlueprintSettings,
    optimizer: torch.optim.Optimizer | None, device: torch.device,
    epoch: int, seed: int, label: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rng = random.Random(seed)
    totals = {key: 0.0 for key in (
        "eligible", "clean_hits", "predicted_hits", "matched_hits",
        "correct_3m", "depth_error_sum_m",
    )}
    loss_total = classification_total = depth_total = coverage_total = 0.0
    distillation_total = pair_total = 0.0
    progress = tqdm(loader, desc=f"{label} {epoch}/{settings.epochs}", leave=False)
    if training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(progress):
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        row_start, col_start = choose_radar_tile(
            geometry, batch["radar"], batch["radar_valid"], grid,
            settings.tile_rows, settings.tile_cols, rng,
        )
        rows, cols = ray_tile_indices(
            geometry, row_start=row_start,
            row_stop=row_start + settings.tile_rows,
            col_start=col_start, col_stop=col_start + settings.tile_cols,
            batch_size=len(batch["radar"]), device=device,
        )
        with torch.set_grad_enabled(training):
            blueprint = model(
                batch["radar"], batch["radar_valid"],
                batch["observed_lidar"], batch["observed_lidar_valid"],
                rows, cols,
            )
            losses = ray_depth_blueprint_loss(
                blueprint, geometry, batch["clean_lidar"],
                batch["clean_lidar_valid"],
            )
            with torch.no_grad():
                teacher_blueprint = teacher(
                    blueprint.queries, batch["clean_lidar"],
                    batch["clean_lidar_valid"],
                )
            distillation, pairs = relationship_feature_loss(
                blueprint, teacher_blueprint, geometry,
                batch["clean_lidar"], batch["clean_lidar_valid"],
            )
            total = losses["loss"] + settings.distill_weight * distillation
            if training:
                assert optimizer is not None
                window_start = (
                    batch_index // settings.grad_accum_steps
                ) * settings.grad_accum_steps
                window_size = min(settings.grad_accum_steps, len(loader) - window_start)
                (total / window_size).backward()
                if batch_index - window_start + 1 == window_size:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            counts = _ray_counts(
                blueprint, geometry, batch["observed_lidar"],
                batch["observed_lidar_valid"], batch["clean_lidar"],
                batch["clean_lidar_valid"],
            )
        for key, value in counts.items():
            totals[key] += value
        loss_total += float(total.detach())
        classification_total += float(losses["classification"].detach())
        depth_total += float(losses["depth"].detach())
        coverage_total += float(losses["coverage"])
        distillation_total += float(distillation.detach())
        pair_total += float(pairs)
        progress.set_postfix(loss=f"{loss_total / (progress.n + 1):.3f}")
    count = len(loader)
    predicted = totals["predicted_hits"]
    clean = totals["clean_hits"]
    matched = totals["matched_hits"]
    correct = totals["correct_3m"]
    precision = correct / max(predicted, 1)
    recall = correct / max(clean, 1)
    return {
        "loss": loss_total / count,
        "classification_loss": classification_total / count,
        "depth_loss": depth_total / count,
        "feature_distillation_loss": distillation_total / count,
        "teacher_feature_pairs": pair_total,
        "candidate_coverage": coverage_total / count,
        "radar_supported_missing_rays": totals["eligible"],
        "clean_hits": clean,
        "predicted_hits": predicted,
        "depth_correct_hits_3m": correct,
        "depth_precision_3m": precision,
        "depth_recall_3m": recall,
        "depth_f1_3m": 2 * precision * recall / max(precision + recall, 1e-12),
        "matched_depth_mae_m": totals["depth_error_sum_m"] / max(matched, 1),
    }


def _run_teacher_epoch(
    loader: DataLoader, *, teacher: CleanLidarRelationshipTeacher,
    geometry: RangeGeometry, grid: EncoderGrid, settings: BlueprintSettings,
    optimizer: torch.optim.Optimizer | None, device: torch.device,
    epoch: int, seed: int,
) -> dict[str, float]:
    training = optimizer is not None
    teacher.train(training)
    rng = random.Random(seed)
    totals = {key: 0.0 for key in ("loss", "classification", "depth", "coverage")}
    if training:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"clean teacher {epoch}/{settings.teacher_epochs}",
                    leave=False)
    for batch_index, batch in enumerate(progress):
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        row_start, col_start = choose_radar_tile(
            geometry, batch["radar"], batch["radar_valid"], grid,
            settings.tile_rows, settings.tile_cols, rng,
        )
        rows, cols = ray_tile_indices(
            geometry, row_start=row_start,
            row_stop=row_start + settings.tile_rows,
            col_start=col_start,
            col_stop=col_start + settings.tile_cols,
            batch_size=len(batch["radar"]), device=device,
        )
        empty_lidar = batch["clean_lidar"][:, :0]
        empty_valid = batch["clean_lidar_valid"][:, :0]
        queries = propose_ray_depth_queries(
            geometry, rows, cols, batch["radar"], batch["radar_valid"],
            empty_lidar, empty_valid,
            radar_slots=3, lidar_slots=0, uniform_slots=2,
        )
        with torch.set_grad_enabled(training):
            blueprint = teacher(
                queries, batch["clean_lidar"], batch["clean_lidar_valid"],
            )
            losses = ray_depth_blueprint_loss(
                blueprint, geometry, batch["clean_lidar"],
                batch["clean_lidar_valid"],
            )
            if training:
                window_start = (
                    batch_index // settings.grad_accum_steps
                ) * settings.grad_accum_steps
                window_size = min(settings.grad_accum_steps,
                                  len(loader) - window_start)
                (losses["loss"] / window_size).backward()
                if batch_index - window_start + 1 == window_size:
                    torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        for key in totals:
            totals[key] += float(losses[key].detach())
        progress.set_postfix(loss=f"{totals['loss'] / (progress.n + 1):.3f}")
    return {key: value / len(loader) for key, value in totals.items()}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-teacher", type=Path,
                        help="Continue interrupted clean-teacher pretraining")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--radar-height-filter", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Optimizer update after this many independent scene tiles")
    parser.add_argument("--tile-rows", type=int, default=4)
    parser.add_argument("--tile-cols", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=5,
                        help="Pretrain clean geometry teacher before radar-only student")
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.resume is not None and args.resume_teacher is not None:
        parser.error("Use --resume or --resume-teacher, not both")
    if (args.epochs < 1 or args.batch_size != 1 or
            args.grad_accum_steps < 1 or args.tile_rows < 1 or
            args.tile_cols < 4 or args.width < 8 or args.width % 4 or
            args.learning_rate <= 0 or args.validate_every < 1 or
            args.teacher_epochs < 1 or args.distill_weight < 0 or
            args.num_workers < 0 or
            (args.train_limit is not None and args.train_limit < 1) or
            (args.val_limit is not None and args.val_limit < 1)):
        parser.error(
            "Use batch size 1 with positive gradient accumulation and valid settings"
        )
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
        raise ValueError("Tile exceeds the calibrated scan or 2048-candidate limit")
    settings = BlueprintSettings(
        args.epochs, args.batch_size, args.grad_accum_steps,
        args.tile_rows, args.tile_cols,
        args.width, args.learning_rate, args.validate_every, args.seed,
        args.teacher_epochs, args.distill_weight,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    grid = EncoderGrid()
    model = RayDepthBlueprintModel(
        geometry, grid, width=settings.width, history_scans=20,
    ).to(device)
    teacher = CleanLidarRelationshipTeacher(
        geometry, grid, width=settings.width, history_scans=20,
    ).to(device)
    training = CrossModalVoDDataset(
        _paths(args.samples_root, "train", args.train_limit),
        args.vod_root, radar_variant=args.radar_variant, include_clean=True,
        radar_height_filter=args.radar_height_filter,
    )
    validation = CrossModalVoDDataset(
        _paths(args.samples_root, "val", args.val_limit),
        args.vod_root, radar_variant=args.radar_variant, include_clean=True,
        radar_height_filter=args.radar_height_filter,
    )
    train_loader = DataLoader(
        training, batch_size=settings.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_cross_modal,
    )
    val_loader = DataLoader(
        validation, batch_size=settings.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cross_modal,
    )
    train_eval_loader = DataLoader(
        training, batch_size=settings.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cross_modal,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=1e-4,
    )
    start_epoch, best_score, best_val_loss = 1, -1.0, float("inf")
    if args.resume is not None:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if (saved.get("stage") != "blueprint_pretraining" or
                saved.get("relationship_version") != "radar_only_clean_teacher_v1"):
            raise ValueError("Resume checkpoint is not this radar-only blueprint")
        if saved.get("geometry_parameters") != asdict(geometry):
            raise ValueError("Resume checkpoint uses different calibrated geometry")
        for key in ("tile_rows", "tile_cols", "width", "grad_accum_steps",
                    "teacher_epochs", "distill_weight"):
            if saved["settings"].get(key, 1) != getattr(settings, key):
                raise ValueError(f"Resume checkpoint disagrees on {key}")
        if (saved["radar_variant"] != args.radar_variant or
                saved["radar_height_filter"] != args.radar_height_filter):
            raise ValueError("Resume checkpoint disagrees on radar input settings")
        model.load_state_dict(saved["blueprint"])
        teacher.load_state_dict(saved["teacher"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved.get("best_score", -1.0))
        best_val_loss = float(saved.get("best_val_loss", float("inf")))
        if start_epoch > settings.epochs:
            raise ValueError("Resume checkpoint already reached the requested epoch count")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "training_config.json").write_text(json.dumps({
        "stage": "blueprint_pretraining",
        "relationship_version": "radar_only_clean_teacher_v1",
        "settings": asdict(settings),
        "geometry": str(args.geometry.resolve()),
        "geometry_parameters": asdict(geometry),
        "radar_variant": args.radar_variant,
        "radar_height_filter": args.radar_height_filter,
        "train_examples": len(training), "val_examples": len(validation),
    }, indent=2), encoding="utf-8")
    if args.resume is None:
        teacher_optimizer = torch.optim.AdamW(
            teacher.parameters(), lr=settings.learning_rate, weight_decay=1e-4,
        )
        teacher_best, teacher_start = float("inf"), 1
        if args.resume_teacher is not None:
            previous = torch.load(args.resume_teacher, map_location=device,
                                  weights_only=False)
            if (previous.get("stage") != "clean_relationship_teacher" or
                    previous.get("geometry_parameters") != asdict(geometry) or
                    previous.get("radar_variant") != args.radar_variant or
                    previous.get("radar_height_filter") != args.radar_height_filter):
                raise ValueError("Teacher checkpoint disagrees with this data setup")
            for key in ("tile_rows", "tile_cols", "width",
                        "grad_accum_steps", "teacher_epochs"):
                if previous["settings"][key] != getattr(settings, key):
                    raise ValueError(f"Teacher checkpoint disagrees on {key}")
            teacher.load_state_dict(previous["teacher"])
            teacher_optimizer.load_state_dict(previous["optimizer"])
            teacher_best = float(previous["best_val_loss"])
            teacher_start = int(previous["epoch"]) + 1
            if teacher_start > settings.teacher_epochs + 1:
                raise ValueError("Teacher checkpoint has an invalid epoch")
        for teacher_epoch in range(teacher_start, settings.teacher_epochs + 1):
            teacher_train = _run_teacher_epoch(
                train_loader, teacher=teacher, geometry=geometry, grid=grid,
                settings=settings, optimizer=teacher_optimizer, device=device,
                epoch=teacher_epoch, seed=settings.seed + teacher_epoch,
            )
            teacher_message = {"phase": "clean_teacher", "epoch": teacher_epoch,
                               "train": teacher_train}
            if (teacher_epoch % settings.validate_every == 0 or
                    teacher_epoch == settings.teacher_epochs):
                teacher_val = _run_teacher_epoch(
                    val_loader, teacher=teacher, geometry=geometry, grid=grid,
                    settings=settings, optimizer=None, device=device,
                    epoch=teacher_epoch, seed=settings.seed + 10000,
                )
                teacher_message["val"] = teacher_val
                if teacher_val["loss"] < teacher_best:
                    teacher_best = teacher_val["loss"]
                    temporary = args.output_root / "teacher_best_checkpoint.tmp"
                    torch.save({
                        "stage": "clean_relationship_teacher",
                        "epoch": teacher_epoch,
                        "teacher": teacher.state_dict(),
                        "geometry_parameters": asdict(geometry),
                        "settings": asdict(settings),
                        "validation": teacher_val,
                    }, temporary)
                    os.replace(temporary,
                               args.output_root / "teacher_best_checkpoint.pt")
            with (args.output_root / "teacher_metrics.jsonl").open(
                    "a", encoding="utf-8") as report:
                report.write(json.dumps(teacher_message) + "\n")
            print(f"Clean teacher {teacher_epoch}/{settings.teacher_epochs} | "
                  f"train loss {teacher_train['loss']:.4f}" +
                  (f" | val loss {teacher_message['val']['loss']:.4f}"
                   if "val" in teacher_message else ""), flush=True)
            temporary = args.output_root / "teacher_last_checkpoint.tmp"
            torch.save({
                "stage": "clean_relationship_teacher",
                "epoch": teacher_epoch,
                "teacher": teacher.state_dict(),
                "optimizer": teacher_optimizer.state_dict(),
                "best_val_loss": teacher_best,
                "geometry_parameters": asdict(geometry),
                "settings": asdict(settings),
                "radar_variant": args.radar_variant,
                "radar_height_filter": args.radar_height_filter,
            }, temporary)
            os.replace(temporary, args.output_root / "teacher_last_checkpoint.pt")
        best_teacher = torch.load(
            args.output_root / "teacher_best_checkpoint.pt",
            map_location=device, weights_only=False,
        )
        teacher.load_state_dict(best_teacher["teacher"])
    teacher.eval().requires_grad_(False)
    for epoch in range(start_epoch, settings.epochs + 1):
        train_metrics = _run_epoch(
            train_loader, model=model, teacher=teacher,
            geometry=geometry, grid=grid,
            settings=settings, optimizer=optimizer, device=device,
            epoch=epoch, seed=settings.seed + epoch, label="blueprint train",
        )
        message = {"epoch": epoch, "train": train_metrics}
        improved = False
        if epoch % settings.validate_every == 0 or epoch == settings.epochs:
            val_metrics = _run_epoch(
                val_loader, model=model, teacher=teacher,
                geometry=geometry, grid=grid,
                settings=settings, optimizer=None, device=device,
                epoch=epoch, seed=settings.seed + 10000, label="blueprint val",
            )
            message["val"] = val_metrics
            if epoch == settings.epochs:
                # Unlike online train loss, this uses the final frozen weights
                # and the same deterministic tile-selection seed as val.
                message["train_eval"] = _run_epoch(
                    train_eval_loader, model=model, teacher=teacher,
                    geometry=geometry, grid=grid,
                    settings=settings, optimizer=None, device=device,
                    epoch=epoch, seed=settings.seed + 10000,
                    label="blueprint train audit",
                )
            if not val_metrics["clean_hits"]:
                message["warning"] = (
                    "No clean returns on sampled radar-supported missing "
                    "validation rays; depth F1 is not informative"
                )
            score = val_metrics["depth_f1_3m"]
            val_loss = val_metrics["loss"]
            improved = (score > best_score or
                        (score == best_score and val_loss < best_val_loss))
            if improved:
                best_score, best_val_loss = score, val_loss
        record_epoch(args.output_root, message,
                     blueprint_summary(message, settings.epochs))
        checkpoint = {
            "stage": "blueprint_pretraining", "epoch": epoch,
            "relationship_version": "radar_only_clean_teacher_v1",
            "blueprint": model.state_dict(), "optimizer": optimizer.state_dict(),
            "teacher": teacher.state_dict(),
            "settings": asdict(settings),
            "geometry_parameters": asdict(geometry),
            "radar_variant": args.radar_variant,
            "radar_height_filter": args.radar_height_filter,
            "best_score": best_score,
            "best_val_loss": best_val_loss,
            "validation": message.get("val"),
            "train_eval": message.get("train_eval"),
        }
        temporary = args.output_root / "last_checkpoint.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, args.output_root / "last_checkpoint.pt")
        if improved:
            best_temporary = args.output_root / "best_checkpoint.tmp"
            torch.save(checkpoint, best_temporary)
            os.replace(best_temporary, args.output_root / "best_checkpoint.pt")


if __name__ == "__main__":
    main()
