from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.visualization_utils import (
    add_label_above,
    add_reliability_colorbar,
    blue_red_reliability,
    draw_cell_boundaries,
    localization_match_overlay,
    make_grid_like,
    save_image,
    side_by_side,
)
from Fault_Localization_Model.sample_utils import filter_paths_by_fault
from PFS.training_utils import resolve_device
from PFS_Radar.datasets import RadarEvaluationDataset
from PFS_Radar.pfs_radar_model import load_model_checkpoint
from PFS_Radar.radar_data import (
    filter_samples_with_radar_cache,
    radar_cache_requirements_from_checkpoint,
)


def list_npz(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.npz"))
    if paths:
        return paths

    if root.name in {"train", "val", "test"}:
        flat_paths = sorted(root.parent.glob("*.npz"))
        if flat_paths:
            train_end = int(len(flat_paths) * 0.70)
            val_end = train_end + int(len(flat_paths) * 0.15)
            if root.name == "train":
                return flat_paths[:train_end]
            if root.name == "val":
                return flat_paths[train_end:val_end]
            return flat_paths[val_end:]

    raise FileNotFoundError(f"No .npz files found in {root}")


def metadata_with_cell_sizes(metadata_json: str, shape: tuple[int, int]) -> dict:
    metadata = json.loads(metadata_json)
    rows, cols = shape
    x_range = metadata.get("x_range", [0.0, 64.0])
    y_range = metadata.get("y_range", [-32.0, 32.0])
    metadata["x_cell_size_m"] = (float(x_range[1]) - float(x_range[0])) / rows
    metadata["y_cell_size_m"] = (float(y_range[1]) - float(y_range[0])) / cols
    return metadata


def tensor_rgb(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().numpy()
    image = np.moveaxis(image[:3], 0, -1)
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ideal-vs-predicted PFS-Radar test visualizations."
    )
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--visual-grid-size", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--localization-tolerance-m", type=float, default=0.20)
    parser.add_argument("--target-fault-threshold", type=float, default=0.0)
    parser.add_argument(
        "--no-match-overlay",
        action="store_true",
        help="Skip the expensive one-to-one localization overlay and export only input/ideal/predicted panels.",
    )
    parser.add_argument("--include-faults", nargs="*", default=None)
    parser.add_argument("--exclude-faults", nargs="*", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.max_images < 1:
        parser.error("--max-images must be at least 1")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be strictly between 0 and 1")

    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(Path(args.checkpoint), device)
    cache_requirements = radar_cache_requirements_from_checkpoint(checkpoint)

    paths = list_npz(Path(args.test_root))
    if args.include_faults or args.exclude_faults:
        paths, _ = filter_paths_by_fault(
            paths,
            args.include_faults,
            args.exclude_faults,
            strict_fault_names=True,
        )
    paths, missing = filter_samples_with_radar_cache(
        paths,
        Path(args.radar_root),
        **cache_requirements,
    )
    if missing:
        print(f"Skipping {len(missing)} test samples without compatible radar cache.")
    if not paths:
        raise FileNotFoundError("No test samples remain after radar/fault filtering")

    rng = random.Random(args.seed)
    selected = rng.sample(paths, min(args.max_images, len(paths)))
    loader = DataLoader(
        RadarEvaluationDataset(
            selected,
            Path(args.radar_root),
            (args.resize_height, args.resize_width),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    output_root = Path(args.output_root)
    image_dir = output_root / "ideal_vs_predicted_100"
    rows = []
    model.eval()
    with torch.inference_mode():
        written = 0
        for lidar, radar, target, metadata_jsons in loader:
            logits = model(
                lidar.to(device, non_blocking=True),
                radar.to(device, non_blocking=True),
            )
            predictions = torch.sigmoid(logits).cpu().numpy()
            targets = target.cpu().numpy()

            for index, metadata_json in enumerate(metadata_jsons):
                metadata = metadata_with_cell_sizes(
                    metadata_json,
                    targets[index, 0].shape,
                )
                target_map = make_grid_like(
                    targets[index, 0],
                    grid_size=args.visual_grid_size,
                )
                prediction_map = make_grid_like(
                    predictions[index, 0],
                    grid_size=args.visual_grid_size,
                )
                target_rgb = draw_cell_boundaries(
                    blue_red_reliability(target_map),
                    grid_size=args.visual_grid_size,
                )
                prediction_rgb = draw_cell_boundaries(
                    blue_red_reliability(prediction_map),
                    grid_size=args.visual_grid_size,
                )
                match_rgb = None
                if not args.no_match_overlay:
                    match_rgb = draw_cell_boundaries(
                        localization_match_overlay(
                            target_map,
                            prediction_map,
                            metadata,
                            prediction_threshold=args.threshold,
                            target_fault_threshold=args.target_fault_threshold,
                            tolerance_m=args.localization_tolerance_m,
                        ),
                        grid_size=args.visual_grid_size,
                    )
                input_rgb = tensor_rgb(lidar[index])

                fault = metadata.get("fault", "unknown_fault")
                severity = metadata.get("severity", "unknown")
                timestamp = metadata.get("timestamp", "unknown_timestamp")
                stem = f"{written:04d}_{fault}_s{severity}_{timestamp}"
                label = f"{fault} S{severity}"

                panels = [
                        add_label_above(input_rgb, f"LiDAR input: {label}"),
                        add_reliability_colorbar(
                            add_label_above(target_rgb, "ideal / target")
                        ),
                        add_reliability_colorbar(
                            add_label_above(prediction_rgb, "predicted")
                        ),
                ]
                if match_rgb is not None:
                    panels.append(
                        add_label_above(
                            match_rgb,
                            "match: white=both cyan=pred green=target red=miss yellow=false",
                        )
                    )
                comparison = side_by_side(panels)
                save_image(image_dir / f"{stem}_ideal_vs_predicted.png", comparison)
                rows.append(
                    {
                        "file": f"{stem}_ideal_vs_predicted.png",
                        "source_sample": str(selected[written]),
                        "fault": fault,
                        "severity": severity,
                        "timestamp": timestamp,
                        "threshold": args.threshold,
                        "mae": float(np.mean(np.abs(prediction_map - target_map))),
                        "mean_prediction": float(np.mean(prediction_map)),
                        "mean_target": float(np.mean(target_map)),
                    }
                )
                written += 1
                if written >= args.max_images:
                    break
            if written >= args.max_images:
                break

    write_rows(output_root / "prediction_visualization_samples.csv", rows)
    print(f"Saved {len(rows)} visualizations to: {image_dir}")
    print(f"Saved sample CSV to: {output_root / 'prediction_visualization_samples.csv'}")


if __name__ == "__main__":
    main()
