from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel, parameter_breakdown
from PFS_Radar.datasets import RadarReliabilityDataset
from PFS_Radar.radar_data import filter_samples_with_radar_cache
from Fault_Localization_Model.heatmap_metrics import HeatmapMetricAccumulator
from Fault_Localization_Model.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    write_csv_rows,
)
from Fault_Localization_Model.sample_utils import (
    filter_paths_by_fault,
    require_disjoint_splits,
)
from PFS.training_utils import (
    capture_rng_state,
    require_checkpoint_args_match,
    require_checkpoint_semantics,
    resolve_device,
    restore_rng_state,
    seed_everything,
)

TRAINING_SEMANTICS_VERSION = 2


def validate_training_args(parser, args):
    finite_values = {
        "--dropout": args.dropout,
        "--learning-rate": args.learning_rate,
        "--min-learning-rate": args.min_learning_rate,
        "--weight-decay": args.weight_decay,
        "--stability-weight": args.stability_weight,
        "--pfs-reliability-weight": args.pfs_reliability_weight,
        "--grad-clip": args.grad_clip,
        "--metric-threshold": args.metric_threshold,
        "--localization-tolerance-m": args.localization_tolerance_m,
        "--target-fault-threshold": args.target_fault_threshold,
        "--localization-loss-weight": args.localization_loss_weight,
        "--false-positive-weight": args.false_positive_weight,
        "--min-delta": args.min_delta,
        "--bev-x-span-m": args.bev_x_span_m,
        "--bev-y-span-m": args.bev_y_span_m,
    }
    optional_finite_values = {
        "--max-radar-delta-ms": args.max_radar_delta_ms,
        "--radar-max-abs-velocity": args.radar_max_abs_velocity,
    }
    for name, value in finite_values.items():
        if not math.isfinite(float(value)):
            parser.error(f"{name} must be finite")
    for name, value in optional_finite_values.items():
        if value is not None and not math.isfinite(float(value)):
            parser.error(f"{name} must be finite when provided")
    positive_integers = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--base-channels": args.base_channels,
        "--resize-height": args.resize_height,
        "--resize-width": args.resize_width,
        "--grid-size": args.grid_size,
        "--metrics-every": args.metrics_every,
    }
    for name, value in positive_integers.items():
        if value < 1:
            parser.error(f"{name} must be at least 1")
    if args.grid_size > min(args.resize_height, args.resize_width):
        parser.error(
            "--grid-size cannot exceed the smaller resized input dimension"
        )
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.metric_grid_size is not None and args.metric_grid_size < 1:
        parser.error("--metric-grid-size must be positive or omitted")
    if (
        args.metric_grid_size is not None
        and args.metric_grid_size > min(args.resize_height, args.resize_width)
    ):
        parser.error(
            "--metric-grid-size cannot exceed the smaller resized input dimension"
        )
    if args.radar_frame_count is not None and args.radar_frame_count < 1:
        parser.error("--radar-frame-count must be positive or omitted")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0,1)")
    if args.learning_rate <= 0.0 or args.min_learning_rate <= 0.0:
        parser.error("Learning rates must be positive")
    if args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must lie between 0 and --epochs")
    non_negative = {
        "--weight-decay": args.weight_decay,
        "--stability-weight": args.stability_weight,
        "--pfs-reliability-weight": args.pfs_reliability_weight,
        "--grad-clip": args.grad_clip,
        "--early-stop-patience": args.early_stop_patience,
        "--localization-tolerance-m": args.localization_tolerance_m,
        "--localization-loss-weight": args.localization_loss_weight,
    }
    for name, value in non_negative.items():
        if value < 0.0:
            parser.error(f"{name} must be non-negative")
    if not 0.0 < args.metric_threshold < 1.0:
        parser.error("--metric-threshold must lie strictly between 0 and 1")
    if not 0.0 <= args.target_fault_threshold < 1.0:
        parser.error("--target-fault-threshold must lie in [0,1)")
    if not 0.0 <= args.false_positive_weight <= 1.0:
        parser.error("--false-positive-weight must lie in [0,1]")
    if args.min_delta < 0.0:
        parser.error("--min-delta must be non-negative")
    if args.bev_x_span_m <= 0.0 or args.bev_y_span_m <= 0.0:
        parser.error("BEV spans must be positive")
    if args.max_radar_delta_ms is not None and args.max_radar_delta_ms < 0.0:
        parser.error("--max-radar-delta-ms must be non-negative")
    if (
        args.radar_max_abs_velocity is not None
        and args.radar_max_abs_velocity <= 0.0
    ):
        parser.error("--radar-max-abs-velocity must be positive")
    if args.require_full_radar_stack and args.radar_frame_count is None:
        parser.error(
            "--require-full-radar-stack requires --radar-frame-count"
        )
    if args.seed < 0:
        parser.error("--seed must be non-negative")


def make_scheduler(optimizer, epochs, warmup_epochs, base_lr, min_lr):
    minimum_factor = min_lr / max(base_lr, 1e-12)

    def schedule(epoch_index):
        epoch = epoch_index + 1
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return max(minimum_factor, epoch / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def euclidean_dilate(values, tolerance_m, x_cell_size_m, y_cell_size_m):
    """Maximum-filter values using a metric Euclidean neighborhood."""
    tolerance_m = max(float(tolerance_m), 0.0)
    x_cell_size_m = max(float(x_cell_size_m), 1e-9)
    y_cell_size_m = max(float(y_cell_size_m), 1e-9)
    row_radius = int(math.floor(tolerance_m / x_cell_size_m + 1e-9))
    col_radius = int(math.floor(tolerance_m / y_cell_size_m + 1e-9))
    padded = F.pad(values, (col_radius, col_radius, row_radius, row_radius))
    height, width = values.shape[-2:]
    output = torch.zeros_like(values)
    for row_offset in range(-row_radius, row_radius + 1):
        for col_offset in range(-col_radius, col_radius + 1):
            distance = math.hypot(
                row_offset * x_cell_size_m,
                col_offset * y_cell_size_m,
            )
            if distance > tolerance_m + 1e-9:
                continue
            row_start = row_radius + row_offset
            col_start = col_radius + col_offset
            shifted = padded[
                ...,
                row_start : row_start + height,
                col_start : col_start + width,
            ]
            output = torch.maximum(output, shifted)
    return output


def localization_surrogate_loss(
    logits,
    target,
    false_positive_weight=0.70,
    target_fault_threshold=0.0,
    tolerance_m=0.20,
    x_cell_size_m=0.20,
    y_cell_size_m=0.20,
):
    """Approximate tolerance-aware localization precision and recall."""
    probability = torch.sigmoid(logits)
    target_mask = (target > target_fault_threshold).to(probability.dtype)
    target_neighborhood = euclidean_dilate(
        target_mask,
        tolerance_m,
        x_cell_size_m,
        y_cell_size_m,
    )
    prediction_neighborhood = euclidean_dilate(
        probability,
        tolerance_m,
        x_cell_size_m,
        y_cell_size_m,
    )

    dimensions = tuple(range(1, probability.ndim))
    matched_prediction = (probability * target_neighborhood).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target_neighborhood)).sum(dim=dimensions)
    covered_target = (target_mask * prediction_neighborhood).sum(dim=dimensions)
    false_negative = (target_mask * (1.0 - prediction_neighborhood)).sum(dim=dimensions)

    precision_loss = false_positive / (matched_prediction + false_positive + 1e-6)
    recall_loss = false_negative / (covered_target + false_negative + 1e-6)
    empty_target = target_mask.sum(dim=dimensions) == 0
    precision_loss = torch.where(
        empty_target,
        probability.mean(dim=dimensions),
        precision_loss,
    )
    recall_loss = torch.where(empty_target, torch.zeros_like(recall_loss), recall_loss)
    false_positive_weight = float(np.clip(false_positive_weight, 0.0, 1.0))
    return (
        false_positive_weight * precision_loss
        + (1.0 - false_positive_weight) * recall_loss
    ).mean()


def stable_heatmap_loss_components(logits, target, grid_size):
    probability = torch.sigmoid(logits)
    weight = 1.0 + 3.0 * target
    pixel_l1 = torch.mean(weight * torch.abs(probability - target))
    prediction_grid = F.adaptive_avg_pool2d(
        probability, output_size=(grid_size, grid_size)
    )
    target_grid = F.adaptive_avg_pool2d(
        target, output_size=(grid_size, grid_size)
    )
    grid_l1 = F.smooth_l1_loss(prediction_grid, target_grid)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    total = 0.50 * pixel_l1 + 1.25 * grid_l1 + 0.25 * bce
    return total, pixel_l1, grid_l1, bce


def compute_loss(
    outputs,
    target,
    grid_size,
    stability_weight,
    pfs_weight,
    localization_weight,
    false_positive_weight,
    target_fault_threshold,
    localization_tolerance_m=0.20,
    x_cell_size_m=0.20,
    y_cell_size_m=0.20,
):
    logits = outputs["logits"]
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    heatmap_loss, pixel_l1, grid_l1, bce = stable_heatmap_loss_components(
        logits,
        target,
        grid_size,
    )
    localization_loss = localization_surrogate_loss(
        logits,
        target,
        false_positive_weight=false_positive_weight,
        target_fault_threshold=target_fault_threshold,
        tolerance_m=localization_tolerance_m,
        x_cell_size_m=x_cell_size_m,
        y_cell_size_m=y_cell_size_m,
    )
    stability_loss = F.smooth_l1_loss(outputs["stabilized_features"], outputs["clean_features"])
    reliability_target = 1.0 - F.adaptive_avg_pool2d(target, outputs["pfs_reliability"].shape[-2:])
    # Block 2 returns sigmoid probabilities. Probability-space BCE is not
    # autocast-safe, so evaluate this small auxiliary term in float32.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        pfs_loss = F.binary_cross_entropy(
            outputs["pfs_reliability"].float().clamp(1e-6, 1.0 - 1e-6),
            reliability_target.float(),
        )
    weighted_localization = localization_weight * localization_loss
    weighted_stability = stability_weight * stability_loss
    weighted_pfs = pfs_weight * pfs_loss
    total = heatmap_loss + weighted_localization + weighted_stability + weighted_pfs
    return (
        total,
        heatmap_loss,
        localization_loss,
        stability_loss,
        pfs_loss,
        pixel_l1,
        grid_l1,
        bce,
        weighted_localization,
        weighted_stability,
        weighted_pfs,
    )


def run_epoch(model, loader, device, optimizer, scaler, args, train, compute_metrics=True):
    model.train(train)
    totals = np.zeros(11, dtype=np.float64)
    samples = 0
    description = "train" if train else "validation"
    metric_accumulator = None
    if not train and compute_metrics:
        metric_accumulator = HeatmapMetricAccumulator(
            threshold=args.metric_threshold,
            target_threshold=args.target_fault_threshold,
            metric_grid_size=args.metric_grid_size,
            compute_chamfer=False,
            localization_tolerance_m=args.localization_tolerance_m,
        )
    for faulty, radar, clean, target, metadata_jsons in tqdm(loader, desc=description, leave=False):
        faulty, radar = faulty.to(device, non_blocking=True), radar.to(device, non_blocking=True)
        clean, target = clean.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(faulty, radar, clean_lidar_bev=clean, return_features=True)
                losses = compute_loss(
                    outputs,
                    target,
                    args.grid_size,
                    args.stability_weight,
                    args.pfs_reliability_weight,
                    args.localization_loss_weight,
                    args.false_positive_weight,
                    args.target_fault_threshold,
                    localization_tolerance_m=args.localization_tolerance_m,
                    x_cell_size_m=args.bev_x_span_m / target.shape[-2],
                    y_cell_size_m=args.bev_y_span_m / target.shape[-1],
                )
            loss_values = torch.stack([value.detach().float() for value in losses])
            if not torch.isfinite(loss_values).all():
                raise FloatingPointError(
                    f"Non-finite {description} loss detected: "
                    f"{loss_values.cpu().tolist()}"
                )
            if train:
                scaler.scale(losses[0]).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                        error_if_nonfinite=True,
                    )
                scaler.step(optimizer)
                scaler.update()
        if metric_accumulator is not None:
            metric_shape = (
                (args.metric_grid_size, args.metric_grid_size)
                if args.metric_grid_size is not None
                else target.shape[-2:]
            )
            metric_metadata = []
            for metadata_json in metadata_jsons:
                metadata = json.loads(metadata_json)
                x_range = metadata.get("x_range", [0.0, 64.0])
                y_range = metadata.get("y_range", [-32.0, 32.0])
                metadata["x_cell_size_m"] = (
                    float(x_range[1]) - float(x_range[0])
                ) / metric_shape[0]
                metadata["y_cell_size_m"] = (
                    float(y_range[1]) - float(y_range[0])
                ) / metric_shape[1]
                metric_metadata.append(metadata)
            metric_accumulator.update(
                outputs["logits"],
                target,
                metadata=metric_metadata,
                from_logits=True,
                update_groups=False,
            )
        batch_size = faulty.shape[0]
        totals += loss_values.cpu().numpy().astype(np.float64) * batch_size
        samples += batch_size
    metrics = metric_accumulator.compute() if metric_accumulator is not None else None
    return totals / max(samples, 1), metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val,
    early_stop_counter,
    args,
    history,
    best_localization_iou=float("-inf"),
):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "training_semantics_version": TRAINING_SEMANTICS_VERSION,
            "epoch": epoch,
            "best_val": best_val,
            "best_localization_iou": best_localization_iou,
            "early_stop_counter": early_stop_counter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
            "history": history,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train radar-conditioned PFS LiDAR fault localization.")
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Load model weights only and start a fresh optimizer, scheduler, and history.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--stability-weight", type=float, default=0.05)
    parser.add_argument("--pfs-reliability-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss decrease required to reset early stopping.",
    )
    parser.add_argument("--metric-threshold", type=float, default=0.15)
    parser.add_argument("--metric-grid-size", type=int, default=None)
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="Calculate expensive validation localization metrics every N epochs.",
    )
    parser.add_argument("--localization-tolerance-m", type=float, default=0.20)
    parser.add_argument("--target-fault-threshold", type=float, default=0.0)
    parser.add_argument("--localization-loss-weight", type=float, default=0.25)
    parser.add_argument("--false-positive-weight", type=float, default=0.70)
    parser.add_argument("--bev-x-span-m", type=float, default=64.0)
    parser.add_argument("--bev-y-span-m", type=float, default=64.0)
    parser.add_argument(
        "--radar-frame-count",
        type=int,
        default=None,
        help="Require cache entries built from this many causal radar frames.",
    )
    parser.add_argument(
        "--require-full-radar-stack",
        action="store_true",
        help="Reject cache entries with fewer frames than --radar-frame-count.",
    )
    parser.add_argument(
        "--max-radar-delta-ms",
        type=float,
        default=None,
        help="Reject cache entries whose nearest radar frame exceeds this offset.",
    )
    parser.add_argument(
        "--radar-max-abs-velocity",
        type=float,
        default=None,
        help="Require this radar velocity normalization limit in m/s.",
    )
    parser.add_argument("--include-faults", nargs="*", default=None)
    parser.add_argument("--exclude-faults", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    validate_training_args(parser, args)

    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_paths = sorted(Path(args.train_root).glob("*.npz"))
    val_paths = sorted(Path(args.val_root).glob("*.npz"))
    if not train_paths or not val_paths:
        raise FileNotFoundError("Both --train-root and --val-root must contain .npz files")
    train_paths, train_fault_counts = filter_paths_by_fault(
        train_paths,
        include_faults=args.include_faults,
        exclude_faults=args.exclude_faults,
        strict_fault_names=True,
    )
    val_paths, val_fault_counts = filter_paths_by_fault(
        val_paths,
        include_faults=args.include_faults,
        exclude_faults=args.exclude_faults,
        strict_fault_names=True,
    )
    print(f"Available train fault counts: {json.dumps(train_fault_counts, sort_keys=True)}")
    print(f"Available validation fault counts: {json.dumps(val_fault_counts, sort_keys=True)}")
    if args.include_faults:
        print(f"Including faults: {', '.join(args.include_faults)}")
    if args.exclude_faults:
        print(f"Excluding faults: {', '.join(args.exclude_faults)}")
    if not train_paths or not val_paths:
        raise FileNotFoundError("Fault filtering removed every train or validation sample")
    require_disjoint_splits({"train": train_paths, "validation": val_paths})
    cache_requirements = {
        "max_delta_ms": args.max_radar_delta_ms,
        "max_abs_velocity": args.radar_max_abs_velocity,
        "radar_frame_count": args.radar_frame_count,
        "require_full_stack": args.require_full_radar_stack,
    }
    train_paths, missing_train = filter_samples_with_radar_cache(
        train_paths,
        Path(args.radar_root),
        **cache_requirements,
    )
    val_paths, missing_val = filter_samples_with_radar_cache(
        val_paths,
        Path(args.radar_root),
        **cache_requirements,
    )
    if missing_train or missing_val:
        print(
            f"Skipping samples without aligned radar cache: "
            f"train={len(missing_train)} validation={len(missing_val)}"
        )
    if not train_paths or not val_paths:
        raise FileNotFoundError("No train/validation samples have aligned radar cache entries")
    resize_hw = (args.resize_height, args.resize_width)
    train_dataset = RadarReliabilityDataset(train_paths, Path(args.radar_root), resize_hw)
    val_dataset = RadarReliabilityDataset(val_paths, Path(args.radar_root), resize_hw)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = PFSRadarReliabilityModel(base_channels=args.base_channels, dropout=args.dropout).to(device)
    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint,
            map_location=device,
            weights_only=False,
        )
        source_args = initialization.get("args", {})
        for key in ("base_channels",):
            source_value = source_args.get(key)
            if source_value is not None and source_value != getattr(args, key):
                raise ValueError(
                    f"Initialization checkpoint {key}={source_value} does not "
                    f"match requested {key}={getattr(args, key)}"
                )
        model.load_state_dict(initialization["model_state_dict"])
        print(
            f"Initialized model weights from {args.init_checkpoint} "
            f"(source epoch {initialization.get('epoch', 'unknown')})"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.learning_rate, args.min_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val, best_localization_iou, early_stop_counter, history = (
        1,
        float("inf"),
        float("-inf"),
        0,
        [],
    )
    if args.resume:
        # This is a trusted checkpoint produced by this training script and
        # includes optimizer, scheduler, scaler, and history objects.
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        require_checkpoint_semantics(
            checkpoint,
            TRAINING_SEMANTICS_VERSION,
            "PFS-Radar",
        )
        saved_args = checkpoint.get("args", {})
        require_checkpoint_args_match(
            saved_args,
            args,
            (
                "base_channels",
                "dropout",
                "learning_rate",
                "min_learning_rate",
                "warmup_epochs",
                "weight_decay",
                "stability_weight",
                "pfs_reliability_weight",
                "localization_loss_weight",
                "false_positive_weight",
                "min_delta",
                "grid_size",
                "resize_height",
                "resize_width",
                "metric_grid_size",
                "metric_threshold",
                "localization_tolerance_m",
                "target_fault_threshold",
                "bev_x_span_m",
                "bev_y_span_m",
                "include_faults",
                "exclude_faults",
                "radar_frame_count",
                "require_full_radar_stack",
                "max_radar_delta_ms",
                "radar_max_abs_velocity",
            ),
        )
        required_state = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
        }
        missing_state = required_state - set(checkpoint)
        if missing_state:
            raise ValueError(
                f"Checkpoint {args.resume} cannot be resumed because it lacks: "
                + ", ".join(sorted(missing_state))
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))
        best_localization_iou = float(
            checkpoint.get("best_localization_iou", best_localization_iou)
        )
        early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
        history = list(checkpoint.get("history", []))
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"Resumed {args.resume} at epoch {start_epoch}", flush=True)

    print(f"Device: {device} | train: {len(train_dataset)} | validation: {len(val_dataset)}")
    print("Parameters:", json.dumps(parameter_breakdown(model), indent=2))
    atomic_write_json(output_root / "training_config.json", vars(args))
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_values, _ = run_epoch(model, train_loader, device, optimizer, scaler, args, train=True)
        calculate_metrics = epoch == 1 or epoch % args.metrics_every == 0
        with torch.no_grad():
            val_values, val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer,
                scaler,
                args,
                train=False,
                compute_metrics=calculate_metrics,
            )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "train_loss": train_values[0],
            "train_heatmap": train_values[1],
            "train_localization": train_values[2],
            "train_stability": train_values[3],
            "train_pfs_reliability": train_values[4],
            "train_heat_pixel_l1": train_values[5],
            "train_heat_grid_l1": train_values[6],
            "train_heat_bce": train_values[7],
            "train_weighted_localization": train_values[8],
            "train_weighted_stability": train_values[9],
            "train_weighted_pfs": train_values[10],
            "val_loss": val_values[0],
            "val_heatmap": val_values[1],
            "val_localization": val_values[2],
            "val_stability": val_values[3],
            "val_pfs_reliability": val_values[4],
            "val_heat_pixel_l1": val_values[5],
            "val_heat_grid_l1": val_values[6],
            "val_heat_bce": val_values[7],
            "val_weighted_localization": val_values[8],
            "val_weighted_stability": val_values[9],
            "val_weighted_pfs": val_values[10],
            "val_localization_iou": (
                val_metrics["localization_iou"] if val_metrics is not None else float("nan")
            ),
            "val_localization_precision": (
                val_metrics["localization_precision"] if val_metrics is not None else float("nan")
            ),
            "val_localization_recall": (
                val_metrics["localization_recall"] if val_metrics is not None else float("nan")
            ),
            "val_localization_f1": (
                val_metrics["localization_f1"] if val_metrics is not None else float("nan")
            ),
            "metric_threshold": args.metric_threshold,
            "localization_tolerance_m": args.localization_tolerance_m,
        }
        history.append(row)
        metric_message = (
            f"\n  localization@{args.localization_tolerance_m:.2f}m "
            f"(threshold={args.metric_threshold:.3f}): "
            f"iou={row['val_localization_iou']:.6f} "
            f"precision={row['val_localization_precision']:.6f} "
            f"recall={row['val_localization_recall']:.6f} "
            f"f1={row['val_localization_f1']:.6f}"
            if val_metrics is not None
            else f"\n  localization metrics skipped (every {args.metrics_every} epochs)"
        )
        print(
            f"epoch {epoch:03d}: train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"heat={row['val_heatmap']:.6f} loc_loss={row['val_localization']:.6f} "
            f"stable={row['val_stability']:.6f} pfs={row['val_pfs_reliability']:.6f} "
            f"lr={row['learning_rate']:.2e}"
            f"\n  val heat: 0.50*pixel={0.50 * row['val_heat_pixel_l1']:.6f} "
            f"+ 1.25*grid={1.25 * row['val_heat_grid_l1']:.6f} "
            f"+ 0.25*bce={0.25 * row['val_heat_bce']:.6f}"
            f"\n  val total: heat={row['val_heatmap']:.6f} "
            f"+ loc={row['val_weighted_localization']:.6f} "
            f"+ stable={row['val_weighted_stability']:.6f} "
            f"+ pfs={row['val_weighted_pfs']:.6f}"
            f"{metric_message}"
        )
        localization_improved = (
            val_metrics is not None
            and row["val_localization_iou"] > best_localization_iou
        )
        if localization_improved:
            best_localization_iou = row["val_localization_iou"]
        improved = row["val_loss"] < best_val - args.min_delta
        if improved:
            best_val = row["val_loss"]
            early_stop_counter = 0
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val,
                early_stop_counter,
                args,
                history,
                best_localization_iou=best_localization_iou,
            )
        else:
            early_stop_counter += 1
        if localization_improved:
            save_checkpoint(
                checkpoint_dir / "best_localization_iou.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val,
                early_stop_counter,
                args,
                history,
                best_localization_iou=best_localization_iou,
            )
        save_checkpoint(
            checkpoint_dir / "last_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val,
            early_stop_counter,
            args,
            history,
            best_localization_iou=best_localization_iou,
        )
        write_csv_rows(
            output_root / "history.csv",
            history,
            fieldnames=list(dict.fromkeys(key for item in history for key in item)),
        )
        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(
                f"Early stopping after {early_stop_counter} epochs without validation-loss improvement. "
                f"Best validation loss: {best_val:.6f}"
            )
            break


if __name__ == "__main__":
    main()
