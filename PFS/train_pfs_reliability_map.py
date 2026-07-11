from pathlib import Path
import argparse
import csv
import json
import math
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
if str(FAULT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FAULT_MODEL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pfs_model import PFSReliabilityModel
from heatmap_metrics import (
    HeatmapMetricAccumulator,
    save_group_metrics,
    save_spatial_error_map,
    save_threshold_sweep,
    threshold_sweep,
)
from train_reliability_map import (
    add_label_above,
    add_reliability_colorbar,
    blue_red_reliability,
    draw_cell_boundaries,
    reliability_loss,
    save_curve,
    save_image,
    save_predictions,
    side_by_side,
    split_paths,
)


DEFAULT_DATASET_ROOT = FAULT_MODEL_DIR / "grid_reliability_change_marks"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "pfs_reliability_map"


class PFSReliabilityDataset(Dataset):
    def __init__(self, paths, resize_hw):
        self.paths = list(paths)
        self.resize_hw = resize_hw
        if not self.paths:
            raise FileNotFoundError("No .npz reliability-map samples found.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with np.load(path, allow_pickle=False) as data:
            if "clean_rgb" not in data:
                raise KeyError(f"{path} does not contain clean_rgb. Regenerate the reliability-map dataset first.")
            faulty_rgb = data["faulty_rgb"].astype(np.float32) / 255.0
            clean_rgb = data["clean_rgb"].astype(np.float32) / 255.0
            target = data["fault_heatmap"].astype(np.float32)
            metadata = json.loads(str(data["metadata_json"]))

        x = torch.from_numpy(np.transpose(faulty_rgb, (2, 0, 1)))
        clean = torch.from_numpy(np.transpose(clean_rgb, (2, 0, 1)))
        y = torch.from_numpy(target).unsqueeze(0)
        if self.resize_hw:
            x = F.interpolate(x.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
            clean = F.interpolate(clean.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=self.resize_hw, mode="nearest").squeeze(0)
        return {
            "x": x,
            "clean": clean,
            "y": y,
            "rgb": (faulty_rgb * 255).astype(np.uint8),
            "path": str(path),
            "metadata": metadata,
        }


def collate(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "clean": torch.stack([item["clean"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "rgb": [item["rgb"] for item in batch],
        "path": [item["path"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def stable_heatmap_loss(logits, target, grid_size=100):
    pred = torch.sigmoid(logits)
    weight = 1.0 + 3.0 * target
    pixel_l1 = torch.mean(weight * torch.abs(pred - target))
    pred_grid = F.adaptive_avg_pool2d(pred, output_size=(grid_size, grid_size))
    target_grid = F.adaptive_avg_pool2d(target, output_size=(grid_size, grid_size))
    grid_l1 = F.smooth_l1_loss(pred_grid, target_grid)
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    return 0.50 * pixel_l1 + 1.25 * grid_l1 + 0.25 * bce


def pfs_training_loss(outputs, target, grid_size, stability_weight, pfs_reliability_weight, loss_mode):
    logits = outputs["logits"]
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

    if loss_mode == "original":
        heatmap = reliability_loss(logits, target, grid_size=grid_size)
    else:
        heatmap = stable_heatmap_loss(logits, target, grid_size=grid_size)
    stability = logits.new_tensor(0.0)
    if outputs["clean_features"] is not None:
        stability = F.smooth_l1_loss(outputs["stabilized_features"], outputs["clean_features"])

    reliability_target = 1.0 - target
    reliability_target = F.interpolate(
        reliability_target,
        size=outputs["pfs_reliability"].shape[-2:],
        mode="area",
    )
    pfs_reliability = F.binary_cross_entropy(outputs["pfs_reliability"], reliability_target)
    total = heatmap + stability_weight * stability + pfs_reliability_weight * pfs_reliability
    return total, {
        "heatmap_loss": float(heatmap.detach().cpu()),
        "stability_loss": float(stability.detach().cpu()),
        "pfs_reliability_loss": float(pfs_reliability.detach().cpu()),
    }


def write_metrics_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_cell_sizes_from_batch(batch, args):
    metadata = batch["metadata"][0] if batch["metadata"] else {}
    x_cell = metadata.get("x_cell_size_m")
    y_cell = metadata.get("y_cell_size_m")
    if x_cell is None:
        x_range = metadata.get("x_range")
        if x_range:
            x_cell = (float(x_range[1]) - float(x_range[0])) / args.grid_size
    if y_cell is None:
        y_range = metadata.get("y_range")
        if y_range:
            y_cell = (float(y_range[1]) - float(y_range[0])) / args.grid_size
    return float(x_cell or args.metric_x_cell_size), float(y_cell or args.metric_y_cell_size)


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
    totals = {"loss": 0.0, "heatmap_loss": 0.0, "stability_loss": 0.0, "pfs_reliability_loss": 0.0}
    count = 0
    metric_accumulator = None
    sweep_outputs = []
    sweep_targets = []
    examples = []
    x_range = None
    y_range = None
    for batch in tqdm(loader, leave=False):
        x = batch["x"].to(device)
        clean = batch["clean"].to(device)
        y = batch["y"].to(device)
        with torch.set_grad_enabled(train):
            outputs = model(x, clean_bev=clean, return_features=True)
            loss, parts = pfs_training_loss(
                outputs,
                y,
                grid_size=args.grid_size,
                stability_weight=args.stability_weight,
                pfs_reliability_weight=args.pfs_reliability_weight,
                loss_mode=args.loss_mode,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        if not train and compute_metrics and not args.disable_metrics:
            logits = outputs["logits"]
            x_cell, y_cell = metric_cell_sizes_from_batch(batch, args)
            if metric_accumulator is None:
                metric_accumulator = HeatmapMetricAccumulator(
                    threshold=args.metric_threshold,
                    metric_grid_size=args.grid_size,
                    x_cell_size_m=x_cell,
                    y_cell_size_m=y_cell,
                    boundary_chamfer=args.boundary_chamfer,
                )
            metric_accumulator.update(logits, y, metadata=batch["metadata"], from_logits=True)
            if args.threshold_sweep:
                prob = torch.sigmoid(logits.detach())
                if prob.shape[-2:] != y.shape[-2:]:
                    prob = F.interpolate(prob, size=y.shape[-2:], mode="bilinear", align_corners=False)
                prob = F.adaptive_avg_pool2d(prob, output_size=(args.grid_size, args.grid_size)).cpu()
                target_grid = F.adaptive_avg_pool2d(y.detach(), output_size=(args.grid_size, args.grid_size)).cpu()
                sweep_outputs.append(prob)
                sweep_targets.append(target_grid)
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
    sweep_rows = []
    if sweep_outputs and sweep_targets:
        sweep_rows = threshold_sweep(sweep_outputs, sweep_targets, args.metric_thresholds, args.grid_size)
    artifacts = {
        "accumulator": metric_accumulator,
        "threshold_sweep": sweep_rows,
        "examples": examples,
        "x_range": tuple(x_range) if x_range else None,
        "y_range": tuple(y_range) if y_range else None,
    }
    return stats, artifacts


def write_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = sorted({key for row in history for key in row.keys()})
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def load_resume_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float(checkpoint.get("best_val_loss", float("inf")))
    history = checkpoint.get("history", [])
    return start_epoch, best_val, history


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
    parser = argparse.ArgumentParser(description="Train a PFS-style model for Hercules BEV fault reliability maps.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--stability-weight", type=float, default=0.25)
    parser.add_argument("--pfs-reliability-weight", type=float, default=0.25)
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
    parser.add_argument("--metric-example-count", type=int, default=5)
    parser.add_argument(
        "--metrics-every",
        type=int,
        default=1,
        help="Compute expensive validation metrics every N epochs. Validation loss still runs every epoch.",
    )
    parser.add_argument("--threshold-sweep", action="store_true", default=True)
    parser.add_argument("--disable-threshold-sweep", action="store_false", dest="threshold_sweep")
    parser.add_argument("--boundary-chamfer", action="store_true")
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

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    paths = sorted(dataset_root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_root}")

    train_paths, val_paths = split_paths(paths, args.val_ratio, args.seed)
    resize_hw = (args.resize_height, args.resize_width)
    device = torch.device(args.device)

    train_loader = DataLoader(
        PFSReliabilityDataset(train_paths, resize_hw),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        PFSReliabilityDataset(val_paths, resize_hw),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    model = PFSReliabilityModel(in_channels=3, base_channels=args.base_channels, dropout=args.dropout).to(device)
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
    if args.resume:
        start_epoch, resumed_best, history = load_resume_checkpoint(Path(args.resume), model, optimizer, scheduler, device)
        if args.best_checkpoint_metric == "val_loss":
            best_score = resumed_best
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

    print(f"Training PFS reliability model on {len(train_paths)} train and {len(val_paths)} val samples.", flush=True)
    epochs_without_improvement = 0
    latest_val_artifacts = None
    for epoch in range(start_epoch, args.epochs + 1):
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
            "train_stability_loss": train_stats["stability_loss"],
            "val_stability_loss": val_stats["stability_loss"],
            "train_pfs_reliability_loss": train_stats["pfs_reliability_loss"],
            "val_pfs_reliability_loss": val_stats["pfs_reliability_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for metric_key, metric_value in val_stats.items():
            if metric_key not in {"loss", "heatmap_loss", "stability_loss", "pfs_reliability_loss"}:
                row[f"val_{metric_key}"] = metric_value
        history.append(row)
        curve_history["epoch"].append(epoch)
        curve_history["train_loss"].append(train_stats["loss"])
        curve_history["val_loss"].append(val_stats["loss"])

        print(
            "epoch "
            f"{epoch:03d}: train={train_stats['loss']:.6f} val={val_stats['loss']:.6f} "
            f"heat={val_stats['heatmap_loss']:.6f} stable={val_stats['stability_loss']:.6f} "
            f"pfs_rel={val_stats['pfs_reliability_loss']:.6f} "
            f"iou={val_stats.get('iou', 0.0):.4f} f1={val_stats.get('f1', 0.0):.4f} "
            f"brier={val_stats.get('brier_score', 0.0):.5f} mae={val_stats.get('pixel_mae', 0.0):.5f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        score = checkpoint_score(val_stats, args.best_checkpoint_metric)
        if score is not None and checkpoint_improved(score, best_score, args.best_checkpoint_metric, args.min_delta):
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_loss": val_stats["loss"],
                    "best_checkpoint_metric": args.best_checkpoint_metric,
                    "best_checkpoint_score": best_score,
                    "architecture": "PFSReliabilityModel",
                    "input": "faulty_rgb_bev",
                    "training_clean_input": "clean_rgb_bev used only for feature stabilization loss",
                    "target": "fault_heatmap/unreliability; reliability=1-target",
                },
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
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_val_loss": val_stats["loss"],
                "best_checkpoint_metric": args.best_checkpoint_metric,
                "best_checkpoint_score": best_score,
                "history": history,
                "architecture": "PFSReliabilityModel",
                "input": "faulty_rgb_bev",
                "training_clean_input": "clean_rgb_bev used only for feature stabilization loss",
                "target": "fault_heatmap/unreliability; reliability=1-target",
            },
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
    write_history(history, output_root / "training_history.csv")
    write_metrics_csv(history, output_root / "validation_metrics" / "epoch_metrics.csv")
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
        checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows = save_predictions(model, val_loader, output_root, device, args.max_val_images)
        if rows:
            with (output_root / "val_predictions" / "prediction_metrics.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Saved PFS run: {output_root}", flush=True)


if __name__ == "__main__":
    main()
