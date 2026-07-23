from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
for path in (REPO_ROOT, FAULT_MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heatmap_metrics import HeatmapMetricAccumulator, prepare_probability_target, save_group_metrics
from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel
from PFS_Radar.radar_data import filter_samples_with_radar_cache, radar_cache_path


class RadarEvaluationDataset(Dataset):
    def __init__(self, paths, radar_root: Path, resize_hw):
        self.paths = list(paths)
        self.radar_root = Path(radar_root)
        self.resize_hw = tuple(resize_hw)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with np.load(path, allow_pickle=False) as data:
            lidar = data["faulty_rgb"].astype(np.float32).transpose(2, 0, 1) / 255.0
            target = data["fault_heatmap"].astype(np.float32)[None]
            metadata_json = str(data["metadata_json"])
        metadata = json.loads(metadata_json)
        cache_path = radar_cache_path(self.radar_root, metadata)
        if not cache_path.exists():
            raise FileNotFoundError(f"Radar cache missing for {path}: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            radar = data["radar_bev"].astype(np.float32)

        lidar = F.interpolate(
            torch.from_numpy(lidar)[None],
            size=self.resize_hw,
            mode="bilinear",
            align_corners=False,
        )[0]
        radar = F.interpolate(
            torch.from_numpy(radar)[None],
            size=self.resize_hw,
            mode="bilinear",
            align_corners=False,
        )[0]
        target = F.interpolate(
            torch.from_numpy(target)[None],
            size=self.resize_hw,
            mode="nearest",
        )[0]
        return lidar, radar, target, metadata_json


def metadata_fault(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    return str(metadata.get("fault", ""))


def list_npz(root: Path, include_faults=None, exclude_faults=None):
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {root}")
    include = set(include_faults or [])
    exclude = set(exclude_faults or [])
    if not include and not exclude:
        return paths
    kept = [path for path in paths if (not include or metadata_fault(path) in include) and metadata_fault(path) not in exclude]
    if not kept:
        raise FileNotFoundError(f"No samples remain in {root} after fault filtering")
    print(f"Fault filter for {root}: kept {len(kept)} / {len(paths)} samples", flush=True)
    return kept


def build_loader(paths, radar_root, resize_hw, batch_size, num_workers, device):
    return DataLoader(
        RadarEvaluationDataset(paths, radar_root, resize_hw),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def load_model(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = PFSRadarReliabilityModel(
        base_channels=int(saved_args.get("base_channels", 16)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def epoch_loss_summary(checkpoint):
    epoch = int(checkpoint.get("epoch", 0))
    history = checkpoint.get("history", [])
    row = next((item for item in reversed(history) if int(item.get("epoch", -1)) == epoch), None)
    if row is None:
        return {"epoch": epoch, "train_loss": None, "val_loss": None}
    return {
        "epoch": epoch,
        "train_loss": float(row["train_loss"]),
        "val_loss": float(row["val_loss"]),
    }


def latest_loss_summary(checkpoint_path: Path):
    latest_path = checkpoint_path.parent / "last_checkpoint.pt"
    if not latest_path.exists():
        return None
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    return epoch_loss_summary(latest)


def format_losses(summary):
    if summary["train_loss"] is None or summary["val_loss"] is None:
        return "not stored in checkpoint"
    return f"train={summary['train_loss']:.6f} validation={summary['val_loss']:.6f}"


def metric_metadata(metadata_jsons, metric_shape):
    rows, cols = metric_shape
    output = []
    for metadata_json in metadata_jsons:
        metadata = json.loads(metadata_json)
        x_range = metadata.get("x_range", [0.0, 64.0])
        y_range = metadata.get("y_range", [-32.0, 32.0])
        metadata["x_cell_size_m"] = (float(x_range[1]) - float(x_range[0])) / rows
        metadata["y_cell_size_m"] = (float(y_range[1]) - float(y_range[0])) / cols
        output.append(metadata)
    return output


def collect_validation(model, loader, device, metric_grid_size, progress_every):
    probabilities, targets, metadata_rows = [], [], []
    with torch.inference_mode():
        for index, (lidar, radar, target, metadata_jsons) in enumerate(loader, start=1):
            logits = model(
                lidar.to(device, non_blocking=True),
                radar.to(device, non_blocking=True),
            )
            probability, target = prepare_probability_target(
                logits,
                target.to(device, non_blocking=True),
                from_logits=True,
                metric_grid_size=metric_grid_size,
            )
            probabilities.append(probability.cpu())
            targets.append(target.cpu())
            metadata_rows.extend(metric_metadata(metadata_jsons, probability.shape[-2:]))
            if index == 1 or index == len(loader) or index % progress_every == 0:
                print(f"[validation inference] batch {index}/{len(loader)}", flush=True)
    return torch.cat(probabilities), torch.cat(targets), metadata_rows


def sweep_thresholds(probability, target, metadata, thresholds, args):
    rows = []
    for index, threshold in enumerate(thresholds, start=1):
        print(f"[validation sweep] threshold {index}/{len(thresholds)}: {threshold:.4f}", flush=True)
        accumulator = HeatmapMetricAccumulator(
            threshold=threshold,
            metric_grid_size=None,
            compute_chamfer=False,
            localization_tolerance_m=args.localization_tolerance_m,
            target_threshold=args.target_fault_threshold,
        )
        accumulator.update(probability, target, metadata=metadata, from_logits=False, update_groups=False)
        row = {"threshold": float(threshold)}
        row.update(accumulator.compute())
        rows.append(row)
        print(
            f"  iou={row['iou']:.4f} f1={row['f1']:.4f} "
            f"loc_iou={row['localization_iou']:.4f} "
            f"precision={row['localization_precision']:.4f} "
            f"recall={row['localization_recall']:.4f}",
            flush=True,
        )
    return rows


def evaluate_test(model, loader, device, threshold, args):
    accumulator = HeatmapMetricAccumulator(
        threshold=threshold,
        metric_grid_size=args.grid_size,
        compute_chamfer=False,
        localization_tolerance_m=args.localization_tolerance_m,
        target_threshold=args.target_fault_threshold,
    )
    with torch.inference_mode():
        for index, (lidar, radar, target, metadata_jsons) in enumerate(loader, start=1):
            logits = model(
                lidar.to(device, non_blocking=True),
                radar.to(device, non_blocking=True),
            )
            shape = (args.grid_size, args.grid_size) if args.grid_size else target.shape[-2:]
            accumulator.update(
                logits,
                target.to(device, non_blocking=True),
                metadata=metric_metadata(metadata_jsons, shape),
                from_logits=True,
                update_groups=True,
            )
            if index == 1 or index == len(loader) or index % args.progress_every == 0:
                print(f"[test evaluation] batch {index}/{len(loader)}", flush=True)
    return accumulator


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(title, metrics):
    print(f"\n{title}")
    for key in (
        "threshold",
        "iou",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "localization_iou",
        "localization_precision",
        "localization_recall",
        "localization_f1",
        "localization_tolerance_m",
        "brier_score",
        "pixel_mae",
        "sample_count",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate PFS-Radar on validation data and evaluate test data once."
    )
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--grid-size", type=int, default=320)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--include-faults", nargs="*", default=None)
    parser.add_argument("--exclude-faults", nargs="*", default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--select-metric",
        choices=("iou", "f1", "balanced_accuracy", "localization_iou", "localization_f1"),
        default="localization_iou",
    )
    parser.add_argument("--localization-tolerance-m", type=float, default=0.20)
    parser.add_argument("--target-fault-threshold", type=float, default=0.0)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Select and save the validation threshold without evaluating test data.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if any(not 0.0 < threshold < 1.0 for threshold in args.thresholds):
        raise ValueError("Every prediction threshold must be strictly between 0 and 1")
    if not args.validation_only and not args.test_root:
        raise ValueError("--test-root is required unless --validation-only is used")

    device = torch.device(args.device)
    val_paths = list_npz(Path(args.val_root), args.include_faults, args.exclude_faults)
    val_paths, missing_val = filter_samples_with_radar_cache(val_paths, Path(args.radar_root))
    test_paths = []
    missing_test = []
    if not args.validation_only:
        test_paths = list_npz(Path(args.test_root), args.include_faults, args.exclude_faults)
        test_paths, missing_test = filter_samples_with_radar_cache(
            test_paths, Path(args.radar_root)
        )
    if missing_val or missing_test:
        print(
            f"Skipping samples without aligned radar cache: "
            f"validation={len(missing_val)} test={len(missing_test)}"
        )
    resize_hw = (args.resize_height, args.resize_width)
    val_loader = build_loader(
        val_paths, Path(args.radar_root), resize_hw, args.batch_size, args.num_workers, device
    )
    test_loader = None
    if not args.validation_only:
        test_loader = build_loader(
            test_paths,
            Path(args.radar_root),
            resize_hw,
            args.batch_size,
            args.num_workers,
            device,
        )
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model(checkpoint_path, device)
    evaluated_losses = epoch_loss_summary(checkpoint)
    latest_losses = latest_loss_summary(checkpoint_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Evaluated checkpoint losses: {format_losses(evaluated_losses)}")
    if latest_losses is not None and latest_losses["epoch"] != evaluated_losses["epoch"]:
        print(f"Latest completed epoch {latest_losses['epoch']} losses: {format_losses(latest_losses)}")
    print(f"Validation samples: {len(val_paths)}")
    if not args.validation_only:
        print(f"Test samples: {len(test_paths)}")
    print(f"Metric grid: {args.grid_size}x{args.grid_size}")
    print(f"Localization tolerance: {args.localization_tolerance_m:.3f} m")
    print(f"Selection metric: {args.select_metric}")

    probability, target, metadata = collect_validation(
        model, val_loader, device, args.grid_size, max(args.progress_every, 1)
    )
    sweep_rows = sweep_thresholds(probability, target, metadata, args.thresholds, args)
    selected = max(sweep_rows, key=lambda row: (row[args.select_metric], row["f1"], row["iou"]))
    threshold = float(selected["threshold"])
    print(f"\nSelected validation threshold: {threshold:.6f}", flush=True)

    write_csv(output_root / "validation_threshold_sweep.csv", sweep_rows)
    summary = {
        "selected_threshold": threshold,
        "selected_metric": args.select_metric,
        "validation_selected_metrics": selected,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "evaluated_checkpoint_losses": evaluated_losses,
        "latest_completed_losses": latest_losses,
        "validation_root": str(args.val_root),
        "metric_grid_size": args.grid_size,
        "localization_tolerance_m": args.localization_tolerance_m,
        "target_fault_threshold": args.target_fault_threshold,
        "include_faults": args.include_faults,
        "exclude_faults": args.exclude_faults,
        "validation_only": args.validation_only,
    }
    test_metrics = None
    if not args.validation_only:
        test_accumulator = evaluate_test(model, test_loader, device, threshold, args)
        test_metrics = test_accumulator.compute()
        test_metrics["threshold"] = threshold
        write_csv(output_root / "test_metrics.csv", [test_metrics])
        save_group_metrics(test_accumulator.groups, output_root / "test_group_metrics")
        summary["test_metrics"] = test_metrics
        summary["test_root"] = str(args.test_root)
    (output_root / "threshold_calibration_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_metrics("Validation metrics at selected threshold", selected)
    if test_metrics is not None:
        print_metrics("Frozen-threshold test metrics", test_metrics)
    print(f"\nSaved outputs to: {output_root}")


if __name__ == "__main__":
    main()
