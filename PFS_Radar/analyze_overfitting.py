from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel
from PFS_Radar.radar_data import filter_samples_with_radar_cache
from PFS_Radar.train_pfs_radar import RadarReliabilityDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate PFS-Radar overfitting by fault type and BEV position."
    )
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--target-fault-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_paths(root, radar_root, maximum):
    paths = sorted(Path(root).glob("*.npz"))
    paths, missing = filter_samples_with_radar_cache(paths, Path(radar_root))
    if missing:
        print(f"Skipping {len(missing)} samples without radar cache under {root}")
    if maximum > 0:
        paths = paths[:maximum]
    if not paths:
        raise FileNotFoundError(f"No samples with radar cache found under {root}")
    return paths


def empty_group():
    return {
        "samples": 0,
        "pixels": 0,
        "absolute_error_sum": 0.0,
        "brier_sum": 0.0,
        "predicted_fault_cells": 0,
        "target_fault_cells": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def finish_group(values):
    tp, fp, fn = values["tp"], values["fp"], values["fn"]
    pixels = max(values["pixels"], 1)
    return {
        "samples": values["samples"],
        "mae": values["absolute_error_sum"] / pixels,
        "brier_score": values["brier_sum"] / pixels,
        "predicted_fault_fraction": values["predicted_fault_cells"] / pixels,
        "target_fault_fraction": values["target_fault_cells"] / pixels,
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


@torch.no_grad()
def evaluate_split(model, loader, device, threshold, target_threshold, name):
    spatial = None
    groups = defaultdict(empty_group)
    sample_count = 0

    for faulty, radar, _clean, target, metadata_jsons in tqdm(loader, desc=name):
        faulty = faulty.to(device, non_blocking=True)
        radar = radar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        probability = torch.sigmoid(model(faulty, radar))
        if probability.shape[-2:] != target.shape[-2:]:
            probability = torch.nn.functional.interpolate(
                probability, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        probability_np = probability.cpu().numpy()[:, 0]
        target_np = target.cpu().numpy()[:, 0]
        if spatial is None:
            shape = target_np.shape[-2:]
            spatial = {
                "absolute_error": np.zeros(shape, dtype=np.float64),
                "overprediction": np.zeros(shape, dtype=np.float64),
                "underprediction": np.zeros(shape, dtype=np.float64),
            }

        for prediction, truth, metadata_json in zip(
            probability_np, target_np, metadata_jsons
        ):
            metadata = json.loads(metadata_json)
            fault = str(metadata.get("fault", "unknown"))
            absolute_error = np.abs(prediction - truth)
            overprediction = np.maximum(prediction - truth, 0.0)
            underprediction = np.maximum(truth - prediction, 0.0)
            spatial["absolute_error"] += absolute_error
            spatial["overprediction"] += overprediction
            spatial["underprediction"] += underprediction

            predicted_mask = prediction >= threshold
            target_mask = truth > target_threshold
            values = groups[fault]
            values["samples"] += 1
            values["pixels"] += truth.size
            values["absolute_error_sum"] += float(absolute_error.sum())
            values["brier_sum"] += float(np.square(prediction - truth).sum())
            values["predicted_fault_cells"] += int(predicted_mask.sum())
            values["target_fault_cells"] += int(target_mask.sum())
            values["tp"] += int(np.logical_and(predicted_mask, target_mask).sum())
            values["fp"] += int(np.logical_and(predicted_mask, ~target_mask).sum())
            values["fn"] += int(np.logical_and(~predicted_mask, target_mask).sum())
            sample_count += 1

    for key in spatial:
        spatial[key] = (spatial[key] / max(sample_count, 1)).astype(np.float32)
    return spatial, {fault: finish_group(values) for fault, values in sorted(groups.items())}


def save_spatial_figure(train_maps, val_maps, output_path):
    names = ("absolute_error", "overprediction", "underprediction")
    labels = ("absolute error", "broad-region error", "missed-fault error")
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    for row, (key, label) in enumerate(zip(names, labels)):
        gap = val_maps[key] - train_maps[key]
        limit = max(float(train_maps[key].max()), float(val_maps[key].max()), 1e-6)
        gap_limit = max(float(np.abs(gap).max()), 1e-6)
        images = (
            axes[row, 0].imshow(train_maps[key], cmap="magma", vmin=0.0, vmax=limit),
            axes[row, 1].imshow(val_maps[key], cmap="magma", vmin=0.0, vmax=limit),
            axes[row, 2].imshow(gap, cmap="coolwarm", vmin=-gap_limit, vmax=gap_limit),
        )
        axes[row, 0].set_title(f"Train {label}")
        axes[row, 1].set_title(f"Validation {label}")
        axes[row, 2].set_title(f"Validation - train {label}")
        figure.colorbar(images[0], ax=axes[row, :2], shrink=0.75)
        figure.colorbar(images[2], ax=axes[row, 2], shrink=0.75)
        for axis in axes[row]:
            axis.set_xlabel("BEV column")
            axis.set_ylabel("BEV row")
    figure.suptitle("PFS-Radar spatial overfitting diagnosis")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    resize_hw = (
        int(saved_args.get("resize_height", 320)),
        int(saved_args.get("resize_width", 320)),
    )
    model = PFSRadarReliabilityModel(
        base_channels=int(saved_args.get("base_channels", 16)),
        dropout=float(saved_args.get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_paths = load_paths(
        args.train_root, args.radar_root, args.max_samples_per_split
    )
    val_paths = load_paths(args.val_root, args.radar_root, args.max_samples_per_split)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        RadarReliabilityDataset(train_paths, Path(args.radar_root), resize_hw),
        **loader_options,
    )
    val_loader = DataLoader(
        RadarReliabilityDataset(val_paths, Path(args.radar_root), resize_hw),
        **loader_options,
    )

    train_maps, train_groups = evaluate_split(
        model,
        train_loader,
        device,
        args.threshold,
        args.target_fault_threshold,
        "train diagnosis",
    )
    val_maps, val_groups = evaluate_split(
        model,
        val_loader,
        device,
        args.threshold,
        args.target_fault_threshold,
        "validation diagnosis",
    )

    faults = sorted(set(train_groups) | set(val_groups))
    fault_gaps = {}
    for fault in faults:
        if fault not in train_groups or fault not in val_groups:
            fault_gaps[fault] = {
                "available": False,
                "reason": "Fault is absent from one evaluated split.",
            }
            continue
        fault_gaps[fault] = {
            "available": True,
            **{
                key: val_groups[fault][key] - train_groups[fault][key]
                for key in ("mae", "brier_score", "iou", "precision", "recall")
            },
        }

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_root / "spatial_overfitting_maps.npz",
        **{f"train_{key}": value for key, value in train_maps.items()},
        **{f"val_{key}": value for key, value in val_maps.items()},
        **{
            f"gap_{key}": val_maps[key] - train_maps[key]
            for key in train_maps
        },
    )
    save_spatial_figure(
        train_maps, val_maps, output_root / "spatial_overfitting_diagnosis.png"
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "threshold": args.threshold,
        "train_samples": len(train_paths),
        "validation_samples": len(val_paths),
        "train_by_fault": train_groups,
        "validation_by_fault": val_groups,
        "validation_minus_train_by_fault": fault_gaps,
    }
    (output_root / "overfitting_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved diagnostics to {output_root}")


if __name__ == "__main__":
    main()
