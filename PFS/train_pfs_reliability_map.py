from pathlib import Path
import argparse
import math
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PFS.datasets import PFSReliabilityDataset, collate_reliability_batch
from PFS.pfs_model import MODEL_VARIANTS, build_reliability_model
from Fault_Localization_Model.heatmap_metrics import (
    HeatmapMetricAccumulator,
    save_group_metrics,
    save_spatial_error_map,
    save_threshold_sweep,
)
from Fault_Localization_Model.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    write_csv_rows,
)
from Fault_Localization_Model.sample_utils import require_disjoint_splits
from PFS.training_utils import (
    capture_rng_state,
    original_reliability_loss,
    require_checkpoint_args_match,
    require_checkpoint_semantics,
    resolve_device,
    restore_rng_state,
    save_curve,
    save_predictions,
    seed_everything,
    split_paths,
)
from Fault_Localization_Model.visualization_utils import (
    add_label_above,
    add_reliability_colorbar,
    blue_red_reliability,
    draw_cell_boundaries,
    save_image,
    side_by_side,
)


DEFAULT_DATASET_ROOT = FAULT_MODEL_DIR / "grid_reliability_change_marks"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "pfs_reliability_map"
TRAINING_SEMANTICS_VERSION = 2


def validate_training_args(parser, args):
    finite_values = {
        "--dropout": args.dropout,
        "--learning-rate": args.learning_rate,
        "--min-learning-rate": args.min_learning_rate,
        "--val-ratio": args.val_ratio,
        "--stability-weight": args.stability_weight,
        "--pfs-reliability-weight": args.pfs_reliability_weight,
        "--localization-loss-weight": args.localization_loss_weight,
        "--false-positive-weight": args.false_positive_weight,
        "--weight-decay": args.weight_decay,
        "--grad-clip": args.grad_clip,
        "--min-delta": args.min_delta,
        "--metric-threshold": args.metric_threshold,
        "--metric-x-cell-size": args.metric_x_cell_size,
        "--metric-y-cell-size": args.metric_y_cell_size,
        "--target-fault-threshold": args.target_fault_threshold,
        "--localization-tolerance-m": args.localization_tolerance_m,
    }
    for name, value in finite_values.items():
        if not math.isfinite(float(value)):
            parser.error(f"{name} must be finite")
    positive_integers = {
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--resize-height": args.resize_height,
        "--resize-width": args.resize_width,
        "--base-channels": args.base_channels,
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
    if args.max_val_images < 0 or args.metric_example_count < 0:
        parser.error("Image/example counts must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0,1)")
    if args.learning_rate <= 0.0 or args.min_learning_rate <= 0.0:
        parser.error("Learning rates must be positive")
    if args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate cannot exceed --learning-rate")
    if not 0 <= args.warmup_epochs <= args.epochs:
        parser.error("--warmup-epochs must lie between 0 and --epochs")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must lie strictly between 0 and 1")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    non_negative = {
        "--stability-weight": args.stability_weight,
        "--pfs-reliability-weight": args.pfs_reliability_weight,
        "--localization-loss-weight": args.localization_loss_weight,
        "--weight-decay": args.weight_decay,
        "--grad-clip": args.grad_clip,
        "--early-stop-patience": args.early_stop_patience,
        "--min-delta": args.min_delta,
    }
    for name, value in non_negative.items():
        if value < 0.0:
            parser.error(f"{name} must be non-negative")
    if not 0.0 < args.metric_threshold < 1.0:
        parser.error("--metric-threshold must lie strictly between 0 and 1")
    if any(not 0.0 < value < 1.0 for value in args.metric_thresholds):
        parser.error("--metric-thresholds values must lie strictly between 0 and 1")
    if len(set(args.metric_thresholds)) != len(args.metric_thresholds):
        parser.error("--metric-thresholds must not contain duplicates")
    if args.metric_x_cell_size <= 0.0 or args.metric_y_cell_size <= 0.0:
        parser.error("Metric cell sizes must be positive")
    if not 0.0 <= args.target_fault_threshold < 1.0:
        parser.error("--target-fault-threshold must lie in [0,1)")
    if args.localization_tolerance_m < 0.0:
        parser.error("--localization-tolerance-m must be non-negative")
    if not 0.0 <= args.false_positive_weight <= 1.0:
        parser.error("--false-positive-weight must lie in [0, 1]")
    if (
        args.disable_metrics
        and args.best_checkpoint_metric != "val_loss"
    ):
        parser.error(
            "--best-checkpoint-metric must be val_loss when --disable-metrics is used"
        )


def stable_heatmap_loss(logits, target, grid_size=100):
    pred = torch.sigmoid(logits)
    weight = 1.0 + 3.0 * target
    pixel_l1 = torch.mean(weight * torch.abs(pred - target))
    pred_grid = F.adaptive_avg_pool2d(pred, output_size=(grid_size, grid_size))
    target_grid = F.adaptive_avg_pool2d(target, output_size=(grid_size, grid_size))
    grid_l1 = F.smooth_l1_loss(pred_grid, target_grid)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    return 0.50 * pixel_l1 + 1.25 * grid_l1 + 0.25 * bce


def euclidean_dilate(values, tolerance_m, x_cell_size_m, y_cell_size_m):
    tolerance_m = max(float(tolerance_m), 0.0)
    x_cell_size_m = max(float(x_cell_size_m), 1e-9)
    y_cell_size_m = max(float(y_cell_size_m), 1e-9)
    row_radius = int(math.floor(tolerance_m / x_cell_size_m + 1e-9))
    col_radius = int(math.floor(tolerance_m / y_cell_size_m + 1e-9))
    if row_radius == 0 and col_radius == 0:
        return values
    padded = F.pad(
        values,
        (col_radius, col_radius, row_radius, row_radius),
        mode="constant",
        value=0.0,
    )
    output = torch.zeros_like(values)
    height, width = values.shape[-2:]
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


def pfs_training_loss(
    outputs,
    target,
    grid_size,
    stability_weight,
    pfs_reliability_weight,
    localization_weight,
    false_positive_weight,
    target_fault_threshold,
    localization_tolerance_m,
    loss_mode,
):
    logits = outputs["logits"]
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

    if loss_mode == "original":
        heatmap = original_reliability_loss(logits, target, grid_size=grid_size)
    else:
        heatmap = stable_heatmap_loss(logits, target, grid_size=grid_size)
    localization = localization_surrogate_loss(
        logits,
        target,
        false_positive_weight=false_positive_weight,
        target_fault_threshold=target_fault_threshold,
        tolerance_m=localization_tolerance_m,
        x_cell_size_m=64.0 / target.shape[-2],
        y_cell_size_m=64.0 / target.shape[-1],
    )
    stability = logits.new_tensor(0.0)
    if outputs["clean_features"] is not None:
        stability = F.smooth_l1_loss(outputs["stabilized_features"], outputs["clean_features"])

    pfs_reliability = logits.new_tensor(0.0)
    if outputs["pfs_reliability"] is not None:
        reliability_target = 1.0 - target
        reliability_target = F.interpolate(
            reliability_target,
            size=outputs["pfs_reliability"].shape[-2:],
            mode="area",
        )
        pfs_reliability = F.binary_cross_entropy(outputs["pfs_reliability"], reliability_target)
    total = (
        heatmap
        + localization_weight * localization
        + stability_weight * stability
        + pfs_reliability_weight * pfs_reliability
    )
    return total, {
        "heatmap_loss": float(heatmap.detach().cpu()),
        "localization_loss": float(localization.detach().cpu()),
        "stability_loss": float(stability.detach().cpu()),
        "pfs_reliability_loss": float(pfs_reliability.detach().cpu()),
    }


def metric_metadata_from_batch(batch, args):
    adjusted = []
    for item in batch["metadata"]:
        metadata = dict(item)
        x_range = metadata.get("x_range")
        y_range = metadata.get("y_range")
        metadata["x_cell_size_m"] = (
            (float(x_range[1]) - float(x_range[0])) / args.grid_size
            if x_range
            else args.metric_x_cell_size
        )
        metadata["y_cell_size_m"] = (
            (float(y_range[1]) - float(y_range[0])) / args.grid_size
            if y_range
            else args.metric_y_cell_size
        )
        adjusted.append(metadata)
    return adjusted


def save_error_examples(examples, output_dir, grid_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, example in enumerate(examples):
        meta = example["metadata"]
        target = example["target"]
        pred = example["prediction"]
        error = np.abs(pred - target)
        input_rgb = example["rgb"]

        pred_rgb = draw_cell_boundaries(blue_red_reliability(pred), grid_size=grid_size)
        target_rgb = draw_cell_boundaries(blue_red_reliability(target), grid_size=grid_size)
        error_rgb = draw_cell_boundaries(blue_red_reliability(error), grid_size=grid_size)

        if input_rgb.shape[:2] != pred_rgb.shape[:2]:
            from PIL import Image

            input_rgb = np.array(
                Image.fromarray(input_rgb, mode="RGB").resize(
                    (pred_rgb.shape[1], pred_rgb.shape[0]),
                    Image.Resampling.BILINEAR,
                )
            )
        label = f"{meta.get('fault', 'unknown')} S{meta.get('severity', '?')}"
        panel = side_by_side(
            [
                add_label_above(input_rgb, f"faulty BEV input: {label}"),
                add_reliability_colorbar(add_label_above(target_rgb, f"target unreliability: {label}")),
                add_reliability_colorbar(add_label_above(pred_rgb, f"predicted unreliability: {label}")),
                add_reliability_colorbar(add_label_above(error_rgb, f"absolute error: {label}")),
            ]
        )
        stem = f"{index:04d}_{meta.get('fault', 'unknown')}_s{meta.get('severity', 'x')}_{meta.get('timestamp', 'no_timestamp')}"
        save_image(output_dir / f"{stem}_prediction_target_error.png", panel)


def save_validation_artifacts(accumulator, sweep_rows, examples, output_dir, args, x_range=None, y_range=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_error = accumulator.mean_error_map()
    if mean_error is not None:
        save_spatial_error_map(mean_error, output_dir, x_range=x_range, y_range=y_range)
    save_group_metrics(accumulator.groups, output_dir)
    save_threshold_sweep(sweep_rows, output_dir)
    save_error_examples(examples, output_dir / "examples", args.grid_size)


def checkpoint_score(stats, metric_name):
    if metric_name not in {"val_loss", "val_f1", "val_iou", "val_brier_score"}:
        raise ValueError(f"Unknown checkpoint metric: {metric_name}")
    if metric_name == "val_loss":
        return stats["loss"]
    if metric_name == "val_f1":
        return stats.get("f1")
    if metric_name == "val_iou":
        return stats.get("iou")
    if metric_name == "val_brier_score":
        return stats.get("brier_score")


def checkpoint_improved(score, best_score, metric_name, min_delta):
    if metric_name in {"val_f1", "val_iou"}:
        return score > best_score + min_delta
    return score < best_score - min_delta


def run_epoch(model, loader, optimizer, device, train, args, compute_metrics=True):
    model.train(train)
    totals = {
        "loss": 0.0,
        "heatmap_loss": 0.0,
        "localization_loss": 0.0,
        "stability_loss": 0.0,
        "pfs_reliability_loss": 0.0,
    }
    count = 0
    metric_accumulator = None
    examples = []
    x_range = None
    y_range = None
    for batch in tqdm(loader, leave=False):
        x = batch["x"].to(device)
        clean = batch["clean"].to(device)
        y = batch["y"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            outputs = model(x, clean_bev=clean, return_features=True)
            loss, parts = pfs_training_loss(
                outputs,
                y,
                grid_size=args.grid_size,
                stability_weight=args.stability_weight,
                pfs_reliability_weight=args.pfs_reliability_weight,
                localization_weight=args.localization_loss_weight,
                false_positive_weight=args.false_positive_weight,
                target_fault_threshold=args.target_fault_threshold,
                localization_tolerance_m=args.localization_tolerance_m,
                loss_mode=args.loss_mode,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {'training' if train else 'validation'} loss detected"
                )
            if train:
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
        if not train and compute_metrics and not args.disable_metrics:
            logits = outputs["logits"]
            metric_metadata = metric_metadata_from_batch(batch, args)
            first_metadata = metric_metadata[0] if metric_metadata else {}
            x_cell = first_metadata.get(
                "x_cell_size_m", args.metric_x_cell_size
            )
            y_cell = first_metadata.get(
                "y_cell_size_m", args.metric_y_cell_size
            )
            if metric_accumulator is None:
                metric_accumulator = HeatmapMetricAccumulator(
                    threshold=args.metric_threshold,
                    target_threshold=args.target_fault_threshold,
                    metric_grid_size=args.grid_size,
                    x_cell_size_m=x_cell,
                    y_cell_size_m=y_cell,
                    boundary_chamfer=False,
                    compute_chamfer=False,
                    localization_tolerance_m=args.localization_tolerance_m,
                )
            metric_accumulator.update(
                logits,
                y,
                metadata=metric_metadata,
                from_logits=True,
                update_groups=False,
            )
            if len(examples) < args.metric_example_count:
                prob = torch.sigmoid(logits.detach())
                if prob.shape[-2:] != y.shape[-2:]:
                    prob = F.interpolate(prob, size=y.shape[-2:], mode="bilinear", align_corners=False)
                prob = F.adaptive_avg_pool2d(prob, output_size=(args.grid_size, args.grid_size)).cpu().numpy()
                target_grid = F.adaptive_avg_pool2d(y.detach(), output_size=(args.grid_size, args.grid_size)).cpu().numpy()
                for i in range(x.shape[0]):
                    if len(examples) >= args.metric_example_count:
                        break
                    examples.append(
                        {
                            "prediction": prob[i, 0],
                            "target": target_grid[i, 0],
                            "metadata": batch["metadata"][i],
                            "rgb": batch["rgb"][i],
                        }
                    )
            if x_range is None and batch["metadata"]:
                x_range = batch["metadata"][0].get("x_range")
                y_range = batch["metadata"][0].get("y_range")
        batch_size = x.shape[0]
        totals["loss"] += float(loss.item()) * batch_size
        for key, value in parts.items():
            totals[key] += value * batch_size
        count += batch_size
    stats = {key: value / max(count, 1) for key, value in totals.items()}
    if metric_accumulator is not None:
        stats.update(metric_accumulator.compute())
    artifacts = {
        "accumulator": metric_accumulator,
        "threshold_sweep": [],
        "examples": examples,
        "x_range": tuple(x_range) if x_range else None,
        "y_range": tuple(y_range) if y_range else None,
    }
    return stats, artifacts


def load_resume_checkpoint(path, model, optimizer, scheduler, device, current_args):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    require_checkpoint_semantics(
        checkpoint,
        TRAINING_SEMANTICS_VERSION,
        "PFS",
    )
    saved_args = checkpoint.get("args", {})
    require_checkpoint_args_match(
        saved_args,
        current_args,
        (
            "model_variant",
            "base_channels",
            "dropout",
            "learning_rate",
            "min_learning_rate",
            "warmup_epochs",
            "weight_decay",
            "grid_size",
            "stability_weight",
            "pfs_reliability_weight",
            "localization_loss_weight",
            "false_positive_weight",
            "loss_mode",
            "best_checkpoint_metric",
            "metric_threshold",
            "target_fault_threshold",
            "localization_tolerance_m",
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
            f"Checkpoint {path} cannot be resumed because it lacks: "
            + ", ".join(sorted(missing_state))
        )
    saved_metric = checkpoint.get(
        "best_checkpoint_metric",
        saved_args.get("best_checkpoint_metric"),
    )
    if saved_metric is None and current_args.best_checkpoint_metric != "val_loss":
        raise ValueError(
            "This legacy checkpoint does not record its selection metric. "
            "Resume it with --best-checkpoint-metric val_loss or initialize a new run."
        )
    if (
        saved_metric is not None
        and saved_metric != current_args.best_checkpoint_metric
    ):
        raise ValueError(
            f"Checkpoint selection metric {saved_metric!r} does not match requested "
            f"metric {current_args.best_checkpoint_metric!r}. Use the original metric "
            "for an exact resume."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_score = float(
        checkpoint.get(
            "best_checkpoint_score",
            checkpoint.get("best_val_loss", float("inf")),
        )
    )
    history = list(checkpoint.get("history", []))
    early_stop_counter = int(checkpoint.get("early_stop_counter", 0))
    restore_rng_state(checkpoint.get("rng_state"))
    return start_epoch, best_score, history, early_stop_counter


def build_checkpoint_payload(
    model,
    optimizer,
    scheduler,
    args,
    epoch,
    best_score,
    val_loss,
    history,
    early_stop_counter,
):
    return {
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "validation_loss": val_loss,
        "best_val_loss": (
            best_score
            if args.best_checkpoint_metric == "val_loss"
            else None
        ),
        "best_checkpoint_metric": args.best_checkpoint_metric,
        "best_checkpoint_score": best_score,
        "early_stop_counter": early_stop_counter,
        "history": history,
        "rng_state": capture_rng_state(),
        "architecture": type(model).__name__,
        "input": "faulty_rgb_bev",
        "training_clean_input": (
            "unused"
            if args.model_variant == "no-pfs"
            else "clean_rgb_bev used only for feature stabilization loss"
        ),
        "target": "fault_heatmap/unreliability; reliability=1-target",
    }


def build_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs, min_lr, base_lr):
    """Linear warmup followed by cosine annealing to min_lr."""
    warmup_epochs = max(0, int(warmup_epochs))
    epochs = max(1, int(epochs))
    min_factor = max(float(min_lr) / max(float(base_lr), 1e-12), 0.0)

    def lr_lambda(epoch_index):
        epoch_number = epoch_index + 1
        if warmup_epochs > 0 and epoch_number <= warmup_epochs:
            return max(min_factor, epoch_number / float(warmup_epochs))
        cosine_epochs = max(1, epochs - warmup_epochs)
        progress = min(max((epoch_number - warmup_epochs) / float(cosine_epochs), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train the reliability-map model on generated VoD BEV samples.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--train-root",
        default=None,
        help="Optional training dataset folder. If set with --val-root, disables random splitting of --dataset-root.",
    )
    parser.add_argument(
        "--val-root",
        default=None,
        help="Optional validation dataset folder generated from held-out scenes.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--model-variant",
        choices=sorted(MODEL_VARIANTS),
        default="pfs",
        help="Select full PFS, a PFS ablation, the LiDAR-only adaptation, or the no-PFS baseline.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--stability-weight", type=float, default=0.25)
    parser.add_argument("--pfs-reliability-weight", type=float, default=0.25)
    parser.add_argument(
        "--localization-loss-weight",
        type=float,
        default=0.0,
        help="Weight for tolerance-aware IoU-style localization surrogate loss.",
    )
    parser.add_argument("--false-positive-weight", type=float, default=0.70)
    parser.add_argument("--loss-mode", choices=["stable", "original"], default="stable")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--best-checkpoint-metric",
        choices=["val_loss", "val_f1", "val_iou", "val_brier_score"],
        default="val_loss",
        help="Metric used to select checkpoints. Defaults to old behavior: lowest validation loss.",
    )
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--metric-thresholds", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--metric-x-cell-size", type=float, default=0.64)
    parser.add_argument("--metric-y-cell-size", type=float, default=0.64)
    parser.add_argument("--target-fault-threshold", type=float, default=0.0)
    parser.add_argument("--localization-tolerance-m", type=float, default=0.20)
    parser.add_argument("--metric-example-count", type=int, default=5)
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="Compute expensive validation metrics every N epochs. Validation loss still runs every epoch.",
    )
    parser.add_argument("--threshold-sweep", action="store_true", default=False)
    parser.add_argument("--disable-threshold-sweep", action="store_false", dest="threshold_sweep")
    parser.add_argument("--boundary-chamfer", action="store_true")
    parser.add_argument("--disable-chamfer", action="store_true")
    parser.add_argument("--disable-metrics", action="store_true")
    parser.add_argument("--disable-validation-artifacts", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--disable-final-predictions", action="store_true")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint. Prefer checkpoints/last_checkpoint.pt for exact epoch resume.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    validate_training_args(parser, args)
    seed_everything(args.seed)

    output_root = Path(args.output_root)
    if args.train_root or args.val_root:
        if not args.train_root or not args.val_root:
            raise ValueError("--train-root and --val-root must be provided together.")
        train_root = Path(args.train_root)
        val_root = Path(args.val_root)
        train_paths = sorted(train_root.glob("*.npz"))
        val_paths = sorted(val_root.glob("*.npz"))
        if not train_paths:
            raise FileNotFoundError(f"No .npz files found in train root {train_root}")
        if not val_paths:
            raise FileNotFoundError(f"No .npz files found in validation root {val_root}")
    else:
        dataset_root = Path(args.dataset_root)
        paths = sorted(dataset_root.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {dataset_root}")
        train_paths, val_paths = split_paths(paths, args.val_ratio, args.seed)
    require_disjoint_splits({"train": train_paths, "validation": val_paths})
    resize_hw = (args.resize_height, args.resize_width)
    device = resolve_device(args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "training_config.json", vars(args))

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_reliability_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        PFSReliabilityDataset(train_paths, resize_hw),
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        PFSReliabilityDataset(val_paths, resize_hw),
        shuffle=False,
        **loader_options,
    )

    model = build_reliability_model(
        args.model_variant,
        in_channels=3,
        base_channels=args.base_channels,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_learning_rate,
        base_lr=args.learning_rate,
    )

    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_score = float("-inf") if args.best_checkpoint_metric in {"val_f1", "val_iou"} else float("inf")
    history = []
    epochs_without_improvement = 0
    if args.resume:
        start_epoch, best_score, history, epochs_without_improvement = (
            load_resume_checkpoint(
                Path(args.resume),
                model,
                optimizer,
                scheduler,
                device,
                args,
            )
        )
        print(
            f"Resumed from {args.resume} at epoch {start_epoch}; "
            f"best_{args.best_checkpoint_metric}={best_score:.6f}",
            flush=True,
        )

    curve_history = {
        "epoch": [row["epoch"] for row in history],
        "train_loss": [row["train_loss"] for row in history],
        "val_loss": [row["val_loss"] for row in history],
    }

    print(
        f"Training {args.model_variant} reliability model on "
        f"{len(train_paths)} train and {len(val_paths)} val samples.",
        flush=True,
    )
    latest_val_artifacts = None
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_stats, _ = run_epoch(model, train_loader, optimizer, device, train=True, args=args)
        compute_val_metrics = (
            not args.disable_metrics
            and args.metrics_every > 0
            and (epoch % args.metrics_every == 0 or epoch == args.epochs)
        )
        val_stats, val_artifacts = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            args=args,
            compute_metrics=compute_val_metrics,
        )
        if compute_val_metrics:
            latest_val_artifacts = val_artifacts
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "train_heatmap_loss": train_stats["heatmap_loss"],
            "val_heatmap_loss": val_stats["heatmap_loss"],
            "train_localization_loss": train_stats["localization_loss"],
            "val_localization_loss": val_stats["localization_loss"],
            "train_stability_loss": train_stats["stability_loss"],
            "val_stability_loss": val_stats["stability_loss"],
            "train_pfs_reliability_loss": train_stats["pfs_reliability_loss"],
            "val_pfs_reliability_loss": val_stats["pfs_reliability_loss"],
            "learning_rate": epoch_learning_rate,
        }
        for metric_key, metric_value in val_stats.items():
            if metric_key not in {
                "loss",
                "heatmap_loss",
                "localization_loss",
                "stability_loss",
                "pfs_reliability_loss",
            }:
                row[f"val_{metric_key}"] = metric_value
        history.append(row)
        curve_history["epoch"].append(epoch)
        curve_history["train_loss"].append(train_stats["loss"])
        curve_history["val_loss"].append(val_stats["loss"])

        print(
            "epoch "
            f"{epoch:03d}: train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} "
            f"heat={val_stats['heatmap_loss']:.6f} loc={val_stats['localization_loss']:.6f} "
            f"stable={val_stats['stability_loss']:.6f} "
            f"pfs_rel={val_stats['pfs_reliability_loss']:.6f} "
            f"iou={val_stats.get('iou', 0.0):.4f} f1={val_stats.get('f1', 0.0):.4f} "
            f"brier={val_stats.get('brier_score', 0.0):.5f} mae={val_stats.get('pixel_mae', 0.0):.5f} "
            f"lr={epoch_learning_rate:.2e}",
            flush=True,
        )
        score = checkpoint_score(val_stats, args.best_checkpoint_metric)
        if score is not None and checkpoint_improved(score, best_score, args.best_checkpoint_metric, args.min_delta):
            best_score = score
            epochs_without_improvement = 0
            atomic_torch_save(
                build_checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    best_score,
                    val_stats["loss"],
                    history,
                    epochs_without_improvement,
                ),
                checkpoint_dir / "best_model.pt",
            )
            if not args.disable_validation_artifacts and val_artifacts["accumulator"] is not None:
                save_validation_artifacts(
                    val_artifacts["accumulator"],
                    val_artifacts["threshold_sweep"],
                    val_artifacts["examples"],
                    output_root / "validation_metrics" / "best",
                    args,
                    x_range=val_artifacts["x_range"],
                    y_range=val_artifacts["y_range"],
                )
        elif score is not None:
            epochs_without_improvement += 1
        atomic_torch_save(
            build_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                args,
                epoch,
                best_score,
                val_stats["loss"],
                history,
                epochs_without_improvement,
            ),
            checkpoint_dir / "last_checkpoint.pt",
        )
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}: no validation improvement for "
                f"{epochs_without_improvement} epochs.",
                flush=True,
            )
            break

    if not args.disable_plots:
        save_curve(curve_history, output_root / "plots" / "training_curve.png")
    write_csv_rows(output_root / "training_history.csv", history)
    if not args.disable_validation_artifacts and latest_val_artifacts and latest_val_artifacts["accumulator"] is not None:
        save_validation_artifacts(
            latest_val_artifacts["accumulator"],
            latest_val_artifacts["threshold_sweep"],
            latest_val_artifacts["examples"],
            output_root / "validation_metrics" / "latest",
            args,
            x_range=latest_val_artifacts["x_range"],
            y_range=latest_val_artifacts["y_range"],
        )

    if not args.disable_final_predictions and args.max_val_images > 0:
        checkpoint_candidates = [checkpoint_dir / "best_model.pt"]
        if args.resume:
            resume_path = Path(args.resume)
            checkpoint_candidates.extend(
                [resume_path.parent / "best_model.pt", resume_path]
            )
        checkpoint_candidates.append(checkpoint_dir / "last_checkpoint.pt")
        prediction_checkpoint = next(
            (path for path in checkpoint_candidates if path.exists()),
            None,
        )
        if prediction_checkpoint is None:
            raise FileNotFoundError(
                "No checkpoint is available for final validation predictions."
            )
        checkpoint = torch.load(
            prediction_checkpoint,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        rows = save_predictions(
            model,
            val_loader,
            output_root,
            device,
            args.max_val_images,
            visual_grid_size=args.grid_size,
            localization_threshold=args.metric_threshold,
            localization_tolerance_m=args.localization_tolerance_m,
            target_fault_threshold=args.target_fault_threshold,
        )
        write_csv_rows(
            output_root / "val_predictions" / "prediction_metrics.csv",
            rows,
            fieldnames=list(rows[0]) if rows else None,
        )
    print(f"Saved PFS run: {output_root}", flush=True)


if __name__ == "__main__":
    main()
