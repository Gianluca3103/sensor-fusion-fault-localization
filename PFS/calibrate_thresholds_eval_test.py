from pathlib import Path
import argparse
import csv
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import torch
from torch.utils.data import DataLoader
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
if str(FAULT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(FAULT_MODEL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from heatmap_metrics import (  # noqa: E402
    HeatmapMetricAccumulator,
    prepare_probability_target,
    save_group_metrics,
)
from pfs_model import PFSReliabilityModel  # noqa: E402
from train_pfs_reliability_map import PFSReliabilityDataset, collate  # noqa: E402


DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "threshold_calibration_test_eval"


def metadata_fault(npz_path: Path) -> str:
    with np.load(npz_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
    return str(metadata.get("fault", ""))


def list_npz(root: Path, include_faults=None, exclude_faults=None):
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {root}")
    include = {fault.strip() for fault in include_faults or [] if fault.strip()}
    exclude = {fault.strip() for fault in exclude_faults or [] if fault.strip()}
    if not include and not exclude:
        return paths

    kept = []
    removed = {}
    for path in paths:
        fault = metadata_fault(path)
        should_keep = (not include or fault in include) and fault not in exclude
        if should_keep:
            kept.append(path)
        else:
            removed[fault] = removed.get(fault, 0) + 1
    if not kept:
        raise FileNotFoundError(f"No .npz files remain in {root} after fault filtering.")
    print(f"Fault filter for {root}: kept {len(kept)} / {len(paths)} samples", flush=True)
    if removed:
        print(f"  removed by fault: {removed}", flush=True)
    return kept


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(checkpoint_path: Path, device, base_channels=None, dropout=None):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_base_channels = base_channels or int(checkpoint_args.get("base_channels", 16))
    model_dropout = float(dropout if dropout is not None else checkpoint_args.get("dropout", 0.0))
    model = PFSReliabilityModel(in_channels=3, base_channels=model_base_channels, dropout=model_dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "base_channels": model_base_channels,
        "dropout": model_dropout,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metric": checkpoint.get("best_checkpoint_metric", "unknown"),
        "checkpoint_best_score": checkpoint.get("best_checkpoint_score"),
    }


def build_loader(paths, resize_hw, batch_size, num_workers):
    return DataLoader(
        PFSReliabilityDataset(paths, resize_hw),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
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


def collect_probabilities(model, loader, device, metric_grid_size, label, progress_every):
    outputs = []
    targets = []
    metadata_rows = []
    model.eval()
    total_batches = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            logits = model(x)
            prob, target = prepare_probability_target(
                logits,
                y,
                from_logits=True,
                metric_grid_size=metric_grid_size,
            )
            outputs.append(prob.cpu())
            targets.append(target.cpu())
            metadata_rows.extend(metric_cell_metadata(batch["metadata"], prob.shape[-2:]))
            if batch_index == 1 or batch_index == total_batches or batch_index % progress_every == 0:
                print(f"[{label}] collected batch {batch_index}/{total_batches}", flush=True)
    return outputs, targets, metadata_rows


def evaluate_dataset(model, loader, device, threshold, metric_grid_size, boundary_chamfer, compute_chamfer, label, progress_every):
    accumulator = HeatmapMetricAccumulator(
        threshold=threshold,
        metric_grid_size=metric_grid_size,
        boundary_chamfer=boundary_chamfer,
        compute_chamfer=compute_chamfer,
    )
    model.eval()
    total_batches = len(loader)
    with torch.no_grad():
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


def sweep_thresholds_with_progress(outputs, targets, metadata, thresholds, label, compute_chamfer):
    rows = []
    total = len(thresholds)
    output_tensor = torch.cat(outputs, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    for index, threshold in enumerate(thresholds, start=1):
        print(f"[{label}] threshold {index}/{total}: {threshold:.4f}", flush=True)
        accumulator = HeatmapMetricAccumulator(threshold=threshold, metric_grid_size=None, compute_chamfer=compute_chamfer)
        accumulator.update(output_tensor, target_tensor, metadata=metadata, from_logits=False, update_groups=False)
        row = {"threshold": float(threshold)}
        row.update(accumulator.compute())
        rows.append(row)
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
        choices=["f1", "iou", "balanced_accuracy", "faulty_only_f1", "faulty_only_iou"],
        default="f1",
        help="Validation metric maximized to choose the frozen test threshold.",
    )
    parser.add_argument("--boundary-chamfer", action="store_true")
    parser.add_argument("--disable-chamfer", action="store_true", help="Skip Chamfer distance for faster high-resolution evaluation.")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    val_root = Path(args.val_root)
    test_root = Path(args.test_root)
    checkpoint_path = Path(args.checkpoint)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    val_paths = list_npz(val_root, include_faults=args.include_faults, exclude_faults=args.exclude_faults)
    test_paths = list_npz(test_root, include_faults=args.include_faults, exclude_faults=args.exclude_faults)
    resize_hw = (args.resize_height, args.resize_width)
    device = torch.device(args.device)

    model, model_info = load_model(checkpoint_path, device, args.base_channels, args.dropout)
    val_loader = build_loader(val_paths, resize_hw, args.batch_size, args.num_workers)
    test_loader = build_loader(test_paths, resize_hw, args.batch_size, args.num_workers)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Validation samples: {len(val_paths)} from {val_root}")
    print(f"Test samples: {len(test_paths)} from {test_root}")
    print(f"Include faults: {args.include_faults}")
    print(f"Exclude faults: {args.exclude_faults}")
    print(f"Threshold candidates: {args.thresholds}")
    print(f"Selection metric: {args.select_metric}")
    metric_resolution = f"{args.grid_size}x{args.grid_size}" if args.grid_size is not None else f"{args.resize_height}x{args.resize_width}"
    print(f"Metric evaluation resolution: {metric_resolution}")
    print(f"Chamfer distance enabled: {not args.disable_chamfer}")

    print("[stage] Collecting validation probabilities", flush=True)
    val_outputs, val_targets, val_metadata = collect_probabilities(
        model, val_loader, device, args.grid_size, "validation", max(args.progress_every, 1)
    )
    print("[stage] Sweeping validation thresholds", flush=True)
    val_sweep_rows = sweep_thresholds_with_progress(
        val_outputs,
        val_targets,
        val_metadata,
        args.thresholds,
        "validation sweep",
        compute_chamfer=not args.disable_chamfer,
    )
    best_row = select_threshold(val_sweep_rows, args.select_metric)
    selected_threshold = float(best_row["threshold"])

    print("[stage] Collecting test probabilities for all-threshold test sweep", flush=True)
    test_outputs, test_targets, test_metadata = collect_probabilities(
        model, test_loader, device, args.grid_size, "test", max(args.progress_every, 1)
    )
    print("[stage] Sweeping test thresholds", flush=True)
    test_sweep_rows = sweep_thresholds_with_progress(
        test_outputs,
        test_targets,
        test_metadata,
        args.thresholds,
        "test sweep",
        compute_chamfer=not args.disable_chamfer,
    )

    print(f"[stage] Evaluating grouped test metrics at selected threshold {selected_threshold:.6f}", flush=True)
    test_accumulator = evaluate_dataset(
        model,
        test_loader,
        device,
        threshold=selected_threshold,
        metric_grid_size=args.grid_size,
        boundary_chamfer=args.boundary_chamfer,
        compute_chamfer=not args.disable_chamfer,
        label="grouped test",
        progress_every=max(args.progress_every, 1),
    )
    test_metrics = test_accumulator.compute()
    test_metrics["threshold"] = selected_threshold

    write_csv(output_root / "validation_threshold_sweep.csv", val_sweep_rows)
    write_csv(output_root / "test_threshold_sweep.csv", test_sweep_rows)
    save_group_metrics(test_accumulator.groups, output_root / "test_group_metrics")

    summary = {
        "selected_threshold": selected_threshold,
        "selected_metric": args.select_metric,
        "validation_selected_metrics": best_row,
        "test_metrics": test_metrics,
        "threshold_candidates": args.thresholds,
        "test_threshold_sweep": test_sweep_rows,
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
    }
    with (output_root / "threshold_calibration_test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    write_csv(output_root / "test_metrics.csv", [test_metrics])

    print()
    print("Final calibrated threshold parameters")
    print(f"  selected_threshold: {selected_threshold:.6f}")
    print(f"  selected_metric: {args.select_metric}")
    print(f"  metric_evaluation_resolution: {metric_resolution}")
    print(f"  threshold_candidates: {args.thresholds}")
    print()
    print_metrics("Validation metrics at selected threshold", best_row)
    print()
    print_metrics("Frozen-threshold test metrics", test_metrics)
    print()
    print(f"Saved calibration/evaluation outputs to: {output_root}")


if __name__ == "__main__":
    main()
