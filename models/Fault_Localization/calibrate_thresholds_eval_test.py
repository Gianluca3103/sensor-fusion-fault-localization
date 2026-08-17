from pathlib import Path
import argparse
import math
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.Fault_Localization.datasets import PFSReliabilityDataset, collate_reliability_batch  # noqa: E402
from Fault_Localization_Model.heatmap_metrics import (  # noqa: E402
    HeatmapMetricAccumulator,
    prepare_probability_target,
    save_group_metrics,
)
from Fault_Localization_Model.io_utils import atomic_write_json, write_csv_rows  # noqa: E402
from models.Fault_Localization.pfs_model import MODEL_VARIANTS, load_model_checkpoint  # noqa: E402
from Fault_Localization_Model.sample_utils import filter_paths_by_fault, require_disjoint_splits  # noqa: E402
from models.Fault_Localization.training_utils import resolve_device  # noqa: E402


DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "threshold_calibration_test_eval"


def list_npz(root: Path, include_faults=None, exclude_faults=None):
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {root}")
    include = {fault.strip() for fault in include_faults or [] if fault.strip()}
    exclude = {fault.strip() for fault in exclude_faults or [] if fault.strip()}
    if not include and not exclude:
        return paths

    kept, counts = filter_paths_by_fault(
        paths,
        include,
        exclude,
        strict_fault_names=True,
    )
    if not kept:
        raise FileNotFoundError(f"No .npz files remain in {root} after fault filtering.")
    print(f"Fault filter for {root}: kept {len(kept)} / {len(paths)} samples", flush=True)
    removed = {
        fault: count
        for fault, count in counts.items()
        if (include and fault not in include) or fault in exclude
    }
    if removed:
        print(f"  removed by fault: {removed}", flush=True)
    return kept


def build_loader(paths, resize_hw, batch_size, num_workers, device):
    return DataLoader(
        PFSReliabilityDataset(paths, resize_hw),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_reliability_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def metric_cell_metadata(metadata, metric_shape):
    rows, cols = metric_shape
    adjusted = []
    for item in metadata:
        meta = dict(item)
        x_range = meta.get("x_range")
        y_range = meta.get("y_range")
        if x_range is not None and y_range is not None:
            meta["x_cell_size_m"] = (float(x_range[1]) - float(x_range[0])) / float(rows)
            meta["y_cell_size_m"] = (float(y_range[1]) - float(y_range[0])) / float(cols)
        adjusted.append(meta)
    return adjusted


def evaluate_dataset(
    model,
    loader,
    device,
    threshold,
    metric_grid_size,
    boundary_chamfer,
    compute_chamfer,
    localization_tolerance_m,
    target_fault_threshold,
    label,
    progress_every,
):
    accumulator = HeatmapMetricAccumulator(
        threshold=threshold,
        metric_grid_size=metric_grid_size,
        boundary_chamfer=boundary_chamfer,
        compute_chamfer=compute_chamfer,
        localization_tolerance_m=localization_tolerance_m,
        target_threshold=target_fault_threshold,
    )
    model.eval()
    total_batches = len(loader)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            if metric_grid_size is None:
                metric_shape = y.shape[-2:]
            else:
                metric_shape = (metric_grid_size, metric_grid_size)
            metadata = metric_cell_metadata(batch["metadata"], metric_shape)
            accumulator.update(logits, y, metadata=metadata, from_logits=True, update_groups=True)
            if batch_index == 1 or batch_index == total_batches or batch_index % progress_every == 0:
                print(f"[{label}] evaluated batch {batch_index}/{total_batches}", flush=True)
    return accumulator


def evaluate_validation_thresholds(
    model,
    loader,
    device,
    thresholds,
    metric_grid_size,
    localization_tolerance_m,
    target_fault_threshold,
    progress_every,
):
    """Calibrate thresholds in one bounded-memory pass without artifact metrics."""
    accumulators = [
        HeatmapMetricAccumulator(
            threshold=threshold,
            metric_grid_size=None,
            compute_chamfer=False,
            localization_tolerance_m=localization_tolerance_m,
            target_threshold=target_fault_threshold,
        )
        for threshold in thresholds
    ]
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            logits = model(batch["x"].to(device, non_blocking=True))
            probability, target = prepare_probability_target(
                logits,
                batch["y"].to(device, non_blocking=True),
                from_logits=True,
                metric_grid_size=metric_grid_size,
            )
            metadata = metric_cell_metadata(
                batch["metadata"], probability.shape[-2:]
            )
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
                batch_index == 1
                or batch_index == len(loader)
                or batch_index % progress_every == 0
            ):
                print(
                    f"[validation calibration] batch {batch_index}/{len(loader)}",
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
            f"({threshold:.4f}): iou={row['iou']:.4f} "
            f"loc_iou={row['localization_iou']:.4f}",
            flush=True,
        )
    return rows


def select_threshold(rows, metric_name):
    if not rows:
        raise ValueError("No threshold rows available for selection.")
    missing = [row for row in rows if metric_name not in row]
    if missing:
        raise KeyError(f"Metric {metric_name!r} is not present in threshold sweep rows.")
    return max(rows, key=lambda row: (row[metric_name], row.get("f1", 0.0), row.get("iou", 0.0)))


def print_metrics(title, metrics):
    print(title)
    for key in [
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
        "chamfer_distance_m",
        "empty_mask_mismatch_rate",
        "sample_count",
        "faulty_sample_count",
    ]:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate fault-heatmap metric threshold on validation data, then evaluate held-out test data."
    )
    parser.add_argument("--val-root", required=True, help="Validation .npz folder used only for threshold calibration.")
    parser.add_argument("--test-root", required=True, help="Held-out test .npz folder used only after threshold selection.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Optional metric pooling size. Omit to evaluate at native resized resolution, usually 320x320.",
    )
    parser.add_argument("--thresholds", type=float, nargs="*", default=[x / 100 for x in range(5, 96, 5)])
    parser.add_argument("--include-faults", nargs="*", default=None, help="Evaluate only these fault names.")
    parser.add_argument("--exclude-faults", nargs="*", default=None, help="Exclude these fault names from val/test.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N batches.")
    parser.add_argument(
        "--select-metric",
        choices=[
            "f1",
            "iou",
            "balanced_accuracy",
            "faulty_only_f1",
            "faulty_only_iou",
            "localization_iou",
            "localization_precision",
            "localization_recall",
            "localization_f1",
        ],
        default="f1",
        help="Validation metric maximized to choose the frozen test threshold.",
    )
    parser.add_argument(
        "--localization-tolerance-m",
        type=float,
        default=0.20,
        help="Metric tolerance in meters for predicted-vs-ground-truth fault localization matches.",
    )
    parser.add_argument(
        "--target-fault-threshold",
        type=float,
        default=0.0,
        help=(
            "Fixed ideal fault-map cutoff. Ground-truth cells are faulty when target > cutoff; "
            "prediction threshold candidates do not change this mask."
        ),
    )
    parser.add_argument("--boundary-chamfer", action="store_true")
    parser.add_argument("--disable-chamfer", action="store_true", help="Skip Chamfer distance for faster high-resolution evaluation.")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--model-variant", choices=sorted(MODEL_VARIANTS), default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.thresholds or any(
        not 0.0 < threshold < 1.0 for threshold in args.thresholds
    ):
        parser.error(
            "All thresholds must be strictly between 0 and 1. "
            "A threshold of 0 marks every cell as faulty and gives invalid perfect metrics."
        )
    if len(set(args.thresholds)) != len(args.thresholds):
        parser.error("--thresholds must not contain duplicates")
    if not 0.0 <= args.target_fault_threshold < 1.0:
        parser.error("--target-fault-threshold must be in [0, 1).")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.resize_height < 1 or args.resize_width < 1:
        parser.error("Resize dimensions must be positive")
    if args.grid_size is not None and args.grid_size < 1:
        parser.error("--grid-size must be positive or omitted")
    if (
        args.grid_size is not None
        and args.grid_size > min(args.resize_height, args.resize_width)
    ):
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
    if args.base_channels is not None and args.base_channels < 1:
        parser.error("--base-channels must be positive")
    if args.dropout is not None and not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0,1)")

    val_root = Path(args.val_root)
    test_root = Path(args.test_root)
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    val_paths = list_npz(val_root, include_faults=args.include_faults, exclude_faults=args.exclude_faults)
    test_paths = list_npz(test_root, include_faults=args.include_faults, exclude_faults=args.exclude_faults)
    require_disjoint_splits({"validation": val_paths, "test": test_paths})
    resize_hw = (args.resize_height, args.resize_width)
    device = resolve_device(args.device)

    model, _, model_info = load_model_checkpoint(
        checkpoint_path,
        device,
        base_channels=args.base_channels,
        dropout=args.dropout,
        model_variant=args.model_variant,
    )
    val_loader = build_loader(
        val_paths, resize_hw, args.batch_size, args.num_workers, device
    )
    test_loader = build_loader(
        test_paths, resize_hw, args.batch_size, args.num_workers, device
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model variant: {model_info['model_variant']}")
    print(f"Validation samples: {len(val_paths)} from {val_root}")
    print(f"Test samples: {len(test_paths)} from {test_root}")
    print(f"Include faults: {args.include_faults}")
    print(f"Exclude faults: {args.exclude_faults}")
    print(f"Threshold candidates: {args.thresholds}")
    print(f"Selection metric: {args.select_metric}")
    metric_resolution = f"{args.grid_size}x{args.grid_size}" if args.grid_size is not None else f"{args.resize_height}x{args.resize_width}"
    print(f"Metric evaluation resolution: {metric_resolution}")
    print(f"Chamfer distance enabled: {not args.disable_chamfer}")
    print(f"Localization tolerance: {args.localization_tolerance_m:.3f} m")
    print(f"Fixed target fault threshold: > {args.target_fault_threshold:.6f}")

    print("[stage] Calibrating validation thresholds", flush=True)
    val_sweep_rows = evaluate_validation_thresholds(
        model,
        val_loader,
        device,
        args.thresholds,
        args.grid_size,
        localization_tolerance_m=args.localization_tolerance_m,
        target_fault_threshold=args.target_fault_threshold,
        progress_every=args.progress_every,
    )
    best_row = select_threshold(val_sweep_rows, args.select_metric)
    selected_threshold = float(best_row["threshold"])

    print(
        f"[stage] Evaluating validation set once at selected threshold "
        f"{selected_threshold:.6f}",
        flush=True,
    )
    val_accumulator = evaluate_dataset(
        model,
        val_loader,
        device,
        threshold=selected_threshold,
        metric_grid_size=args.grid_size,
        boundary_chamfer=args.boundary_chamfer,
        compute_chamfer=not args.disable_chamfer,
        localization_tolerance_m=args.localization_tolerance_m,
        target_fault_threshold=args.target_fault_threshold,
        label="grouped validation",
        progress_every=max(args.progress_every, 1),
    )
    validation_metrics = val_accumulator.compute()
    validation_metrics["threshold"] = selected_threshold

    print(
        f"[stage] Evaluating test set once at frozen validation threshold {selected_threshold:.6f}",
        flush=True,
    )
    test_accumulator = evaluate_dataset(
        model,
        test_loader,
        device,
        threshold=selected_threshold,
        metric_grid_size=args.grid_size,
        boundary_chamfer=args.boundary_chamfer,
        compute_chamfer=not args.disable_chamfer,
        localization_tolerance_m=args.localization_tolerance_m,
        target_fault_threshold=args.target_fault_threshold,
        label="grouped test",
        progress_every=max(args.progress_every, 1),
    )
    test_metrics = test_accumulator.compute()
    test_metrics["threshold"] = selected_threshold

    write_csv_rows(output_root / "validation_threshold_sweep.csv", val_sweep_rows)
    save_group_metrics(
        val_accumulator.groups,
        output_root / "validation_group_metrics",
    )
    save_group_metrics(test_accumulator.groups, output_root / "test_group_metrics")

    summary = {
        "selected_threshold": selected_threshold,
        "selected_metric": args.select_metric,
        "validation_selected_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "threshold_candidates": args.thresholds,
        "test_evaluation_protocol": "single evaluation at validation-selected threshold",
        "include_faults": args.include_faults,
        "exclude_faults": args.exclude_faults,
        "validation_root": str(val_root),
        "test_root": str(test_root),
        "checkpoint": str(checkpoint_path),
        "model": model_info,
        "resize_height": args.resize_height,
        "resize_width": args.resize_width,
        "metric_grid_size": args.grid_size,
        "metric_evaluation_resolution": metric_resolution,
        "boundary_chamfer": args.boundary_chamfer,
        "chamfer_enabled": not args.disable_chamfer,
        "localization_tolerance_m": args.localization_tolerance_m,
        "target_fault_threshold": args.target_fault_threshold,
    }
    atomic_write_json(
        output_root / "threshold_calibration_test_summary.json",
        summary,
    )
    write_csv_rows(output_root / "test_metrics.csv", [test_metrics])

    print()
    print("Final calibrated threshold parameters")
    print(f"  selected_threshold: {selected_threshold:.6f}")
    print(f"  selected_metric: {args.select_metric}")
    print(f"  metric_evaluation_resolution: {metric_resolution}")
    print(f"  threshold_candidates: {args.thresholds}")
    print(f"  target_fault_threshold: > {args.target_fault_threshold:.6f}")
    print()
    print_metrics("Validation metrics at selected threshold", validation_metrics)
    print()
    print_metrics("Frozen-threshold test metrics", test_metrics)
    print()
    print(f"Saved calibration/evaluation outputs to: {output_root}")


if __name__ == "__main__":
    main()
