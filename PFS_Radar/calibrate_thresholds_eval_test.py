from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.heatmap_metrics import (
    HeatmapMetricAccumulator,
    prepare_probability_target,
    save_group_metrics,
)
from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows
from PFS_Radar.datasets import (
    RadarEvaluationDataset,
)
from PFS_Radar.pfs_radar_model import load_model_checkpoint
from PFS_Radar.radar_data import (
    filter_samples_with_radar_cache,
    radar_cache_requirements_from_checkpoint,
)
from Fault_Localization_Model.sample_utils import (
    filter_paths_by_fault,
    require_disjoint_splits,
)
from PFS.training_utils import resolve_device


def list_npz(root: Path, include_faults=None, exclude_faults=None):
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {root}")
    if not include_faults and not exclude_faults:
        return paths
    kept, _ = filter_paths_by_fault(
        paths,
        include_faults,
        exclude_faults,
        strict_fault_names=True,
    )
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


def evaluate_validation_thresholds(model, loader, device, thresholds, args):
    """Run validation inference once while keeping memory bounded by one batch."""
    accumulators = [
        HeatmapMetricAccumulator(
            threshold=threshold,
            metric_grid_size=None,
            compute_chamfer=False,
            localization_tolerance_m=args.localization_tolerance_m,
            target_threshold=args.target_fault_threshold,
        )
        for threshold in thresholds
    ]
    with torch.inference_mode():
        for index, (lidar, radar, target, metadata_jsons) in enumerate(
            loader, start=1
        ):
            logits = model(
                lidar.to(device, non_blocking=True),
                radar.to(device, non_blocking=True),
            )
            probability, target = prepare_probability_target(
                logits,
                target.to(device, non_blocking=True),
                from_logits=True,
                metric_grid_size=args.grid_size,
            )
            metadata = metric_metadata(metadata_jsons, probability.shape[-2:])
            probability = probability.cpu()
            target = target.cpu()
            for accumulator in accumulators:
                accumulator.update(
                    probability,
                    target,
                    metadata=metadata,
                    from_logits=False,
                    update_groups=False,
                )
            if (
                index == 1
                or index == len(loader)
                or index % args.progress_every == 0
            ):
                print(
                    f"[validation calibration] batch {index}/{len(loader)}",
                    flush=True,
                )

    rows = []
    for index, (threshold, accumulator) in enumerate(
        zip(thresholds, accumulators), start=1
    ):
        row = {"threshold": float(threshold)}
        row.update(accumulator.compute())
        rows.append(row)
        print(
            f"[validation result] threshold {index}/{len(thresholds)} "
            f"({threshold:.4f}): "
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
        parser.error("Every prediction threshold must be strictly between 0 and 1")
    if len(set(args.thresholds)) != len(args.thresholds):
        parser.error("--thresholds must not contain duplicates")
    if not args.validation_only and not args.test_root:
        parser.error("--test-root is required unless --validation-only is used")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.resize_height < 1 or args.resize_width < 1 or args.grid_size < 1:
        parser.error("Resize dimensions and --grid-size must be positive")
    if args.grid_size > min(args.resize_height, args.resize_width):
        parser.error(
            "--grid-size cannot exceed the smaller resized input dimension"
        )
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    if (
        not math.isfinite(args.localization_tolerance_m)
        or args.localization_tolerance_m < 0.0
    ):
        parser.error("--localization-tolerance-m must be non-negative")
    if not 0.0 <= args.target_fault_threshold < 1.0:
        parser.error("--target-fault-threshold must lie in [0,1)")

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    cache_requirements = radar_cache_requirements_from_checkpoint(checkpoint)
    val_paths = list_npz(Path(args.val_root), args.include_faults, args.exclude_faults)
    val_paths, missing_val = filter_samples_with_radar_cache(
        val_paths,
        Path(args.radar_root),
        **cache_requirements,
    )
    test_paths = []
    missing_test = []
    if not args.validation_only:
        test_paths = list_npz(Path(args.test_root), args.include_faults, args.exclude_faults)
        test_paths, missing_test = filter_samples_with_radar_cache(
            test_paths,
            Path(args.radar_root),
            **cache_requirements,
        )
        require_disjoint_splits(
            {"validation": val_paths, "test": test_paths}
        )
    if missing_val or missing_test:
        print(
            f"Skipping samples without aligned radar cache: "
            f"validation={len(missing_val)} test={len(missing_test)}"
        )
    if not val_paths:
        raise FileNotFoundError("No validation samples have a compatible radar cache")
    if not args.validation_only and not test_paths:
        raise FileNotFoundError("No test samples have a compatible radar cache")
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

    sweep_rows = evaluate_validation_thresholds(
        model,
        val_loader,
        device,
        args.thresholds,
        args,
    )
    selected = max(sweep_rows, key=lambda row: (row[args.select_metric], row["f1"], row["iou"]))
    threshold = float(selected["threshold"])
    print(f"\nSelected validation threshold: {threshold:.6f}", flush=True)

    write_csv_rows(output_root / "validation_threshold_sweep.csv", sweep_rows)
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
        write_csv_rows(output_root / "test_metrics.csv", [test_metrics])
        save_group_metrics(test_accumulator.groups, output_root / "test_group_metrics")
        summary["test_metrics"] = test_metrics
        summary["test_root"] = str(args.test_root)
    atomic_write_json(
        output_root / "threshold_calibration_test_summary.json",
        summary,
    )
    print_metrics("Validation metrics at selected threshold", selected)
    if test_metrics is not None:
        print_metrics("Frozen-threshold test metrics", test_metrics)
    print(f"\nSaved outputs to: {output_root}")


if __name__ == "__main__":
    main()
