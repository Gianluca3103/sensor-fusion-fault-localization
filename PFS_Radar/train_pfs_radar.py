from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PFS_DIR = REPO_ROOT / "PFS"
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
for path in (REPO_ROOT, PFS_DIR, FAULT_MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel, parameter_breakdown
from PFS_Radar.radar_data import filter_samples_with_radar_cache, radar_cache_path
from heatmap_metrics import HeatmapMetricAccumulator
from train_pfs_reliability_map import stable_heatmap_loss


class RadarReliabilityDataset(Dataset):
    def __init__(self, paths, radar_root: Path, resize_hw=(320, 320)):
        self.paths = list(paths)
        self.radar_root = Path(radar_root)
        self.resize_hw = resize_hw
        if not self.paths:
            raise FileNotFoundError("No reliability-map samples were provided")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            faulty = data["faulty_rgb"].astype(np.float32) / 255.0
            clean = data["clean_rgb"].astype(np.float32) / 255.0
            target = data["fault_heatmap"].astype(np.float32)
            metadata = json.loads(str(data["metadata_json"]))
        cache_path = radar_cache_path(self.radar_root, metadata)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Radar cache missing for {path}: {cache_path}. Run prepare_radar_cache.py first."
            )
        with np.load(cache_path, allow_pickle=False) as radar_data:
            radar = radar_data["radar_bev"].astype(np.float32)

        lidar_tensor = torch.from_numpy(faulty.transpose(2, 0, 1))
        clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1))
        radar_tensor = torch.from_numpy(radar)
        target_tensor = torch.from_numpy(target).unsqueeze(0)
        if self.resize_hw:
            lidar_tensor = F.interpolate(
                lidar_tensor.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False
            ).squeeze(0)
            clean_tensor = F.interpolate(
                clean_tensor.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False
            ).squeeze(0)
            radar_tensor = F.interpolate(
                radar_tensor.unsqueeze(0), size=self.resize_hw, mode="bilinear", align_corners=False
            ).squeeze(0)
            target_tensor = F.interpolate(
                target_tensor.unsqueeze(0), size=self.resize_hw, mode="nearest"
            ).squeeze(0)
        return lidar_tensor, radar_tensor, clean_tensor, target_tensor, json.dumps(metadata)


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


def localization_surrogate_loss(
    logits,
    target,
    radius_cells=1,
    false_positive_weight=0.70,
    target_fault_threshold=0.0,
):
    """Approximate tolerance-aware localization precision and recall."""
    probability = torch.sigmoid(logits)
    target_mask = (target > target_fault_threshold).to(probability.dtype)
    kernel_size = 2 * max(int(radius_cells), 0) + 1
    if kernel_size > 1:
        target_neighborhood = F.max_pool2d(
            target_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        prediction_neighborhood = F.max_pool2d(
            probability, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
    else:
        target_neighborhood = target_mask
        prediction_neighborhood = probability

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


def compute_loss(
    outputs,
    target,
    grid_size,
    stability_weight,
    pfs_weight,
    localization_weight,
    false_positive_weight,
    localization_radius_cells,
    target_fault_threshold,
):
    logits = outputs["logits"]
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    heatmap_loss = stable_heatmap_loss(logits, target, grid_size=grid_size)
    localization_loss = localization_surrogate_loss(
        logits,
        target,
        radius_cells=localization_radius_cells,
        false_positive_weight=false_positive_weight,
        target_fault_threshold=target_fault_threshold,
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
    total = (
        heatmap_loss
        + localization_weight * localization_loss
        + stability_weight * stability_loss
        + pfs_weight * pfs_loss
    )
    return total, heatmap_loss, localization_loss, stability_loss, pfs_loss


def run_epoch(model, loader, device, optimizer, scaler, args, train, compute_metrics=True):
    model.train(train)
    totals = np.zeros(5, dtype=np.float64)
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
                    args.localization_radius_cells,
                    args.target_fault_threshold,
                )
            if train:
                scaler.scale(losses[0]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
        totals += np.asarray([float(loss.detach()) for loss in losses]) * batch_size
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
    torch.save(
        {
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
    parser.add_argument("--localization-radius-cells", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.metrics_every < 1:
        parser.error("--metrics-every must be at least 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_paths = sorted(Path(args.train_root).glob("*.npz"))
    val_paths = sorted(Path(args.val_root).glob("*.npz"))
    if not train_paths or not val_paths:
        raise FileNotFoundError("Both --train-root and --val-root must contain .npz files")
    train_paths, missing_train = filter_samples_with_radar_cache(
        train_paths, Path(args.radar_root)
    )
    val_paths, missing_val = filter_samples_with_radar_cache(val_paths, Path(args.radar_root))
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

    print(f"Device: {device} | train: {len(train_dataset)} | validation: {len(val_dataset)}")
    print("Parameters:", json.dumps(parameter_breakdown(model), indent=2))
    (output_root / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epochs + 1):
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
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_values[0],
            "train_heatmap": train_values[1],
            "train_localization": train_values[2],
            "train_stability": train_values[3],
            "train_pfs_reliability": train_values[4],
            "val_loss": val_values[0],
            "val_heatmap": val_values[1],
            "val_localization": val_values[2],
            "val_stability": val_values[3],
            "val_pfs_reliability": val_values[4],
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
            f"pfs={row['val_pfs_reliability']:.6f} "
            f"lr={row['learning_rate']:.2e}"
            f"{metric_message}"
        )
        localization_improved = (
            val_metrics is not None
            and row["val_localization_iou"] > best_localization_iou
        )
        if localization_improved:
            best_localization_iou = row["val_localization_iou"]
        improved = row["val_loss"] < best_val
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
        with (output_root / "history.csv").open("w", newline="", encoding="utf-8") as file:
            fieldnames = list(dict.fromkeys(key for item in history for key in item))
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(
                f"Early stopping after {early_stop_counter} epochs without validation-loss improvement. "
                f"Best validation loss: {best_val:.6f}"
            )
            break


if __name__ == "__main__":
    main()
