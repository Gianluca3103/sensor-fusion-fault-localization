from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
FAULT_MODEL_DIR = REPO_ROOT / "Fault_Localization_Model"
for path in (REPO_ROOT, FAULT_MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel
from PFS_Radar.radar_data import filter_samples_with_radar_cache, radar_cache_path
from train_reliability_map import (
    add_label_above,
    add_reliability_colorbar,
    blue_red_reliability,
    draw_cell_boundaries,
    localization_match_overlay,
    save_image,
    side_by_side,
)


def radar_to_rgb(radar_bev):
    rgb = np.zeros((radar_bev.shape[1], radar_bev.shape[2], 3), dtype=np.float32)
    rgb[..., 0] = radar_bev[3]
    rgb[..., 1] = radar_bev[2]
    rgb[..., 2] = np.maximum(radar_bev[0], radar_bev[1])
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def resize_chw(array, size, mode):
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
    kwargs = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs).squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="Visualize radar-conditioned PFS fault predictions.")
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--resize-height", type=int, default=320)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--visual-grid-size", type=int, default=320)
    parser.add_argument("--prediction-threshold", type=float, default=0.045)
    parser.add_argument("--target-fault-threshold", type=float, default=0.0)
    parser.add_argument("--localization-tolerance-m", type=float, default=0.20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    # Checkpoints produced by the local trainer include configuration data in
    # addition to tensors, so PyTorch's weights-only loader is not sufficient.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = PFSRadarReliabilityModel(
        base_channels=int(saved_args.get("base_channels", 16)),
        dropout=float(saved_args.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths, missing_paths = filter_samples_with_radar_cache(
        sorted(Path(args.test_root).glob("*.npz")), Path(args.radar_root)
    )
    if missing_paths:
        print(f"Skipping {len(missing_paths)} samples without aligned radar cache")
    paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No .npz samples found in {args.test_root}")

    size = (args.resize_height, args.resize_width)
    with torch.no_grad():
        for index, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as data:
                faulty_rgb = data["faulty_rgb"].astype(np.uint8)
                target = data["fault_heatmap"].astype(np.float32)
                metadata = json.loads(str(data["metadata_json"]))
            radar_path = radar_cache_path(Path(args.radar_root), metadata)
            with np.load(radar_path, allow_pickle=False) as data:
                radar_bev = data["radar_bev"].astype(np.float32)

            lidar = torch.from_numpy(faulty_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
            lidar = resize_chw(lidar.numpy(), size, "bilinear").unsqueeze(0).to(device)
            radar = resize_chw(radar_bev, size, "bilinear").unsqueeze(0).to(device)
            prediction = torch.sigmoid(model(lidar, radar))[0, 0].cpu().numpy()
            target_tensor = resize_chw(target[None], size, "nearest")[0].numpy()

            faulty_panel = np.asarray(
                Image.fromarray(faulty_rgb).resize((args.resize_width, args.resize_height), Image.Resampling.BILINEAR)
            )
            radar_panel = radar_to_rgb(radar.cpu().numpy()[0])
            ideal_panel = draw_cell_boundaries(
                blue_red_reliability(target_tensor), args.visual_grid_size
            )
            prediction_panel = draw_cell_boundaries(
                blue_red_reliability(prediction), args.visual_grid_size
            )
            match_panel = draw_cell_boundaries(
                localization_match_overlay(
                    target_tensor,
                    prediction,
                    metadata,
                    prediction_threshold=args.prediction_threshold,
                    target_fault_threshold=args.target_fault_threshold,
                    tolerance_m=args.localization_tolerance_m,
                ),
                args.visual_grid_size,
            )
            fault = metadata.get("fault", "unknown")
            severity = metadata.get("severity", "?")
            tolerance_cm = int(round(args.localization_tolerance_m * 100))
            comparison = side_by_side(
                [
                    add_label_above(faulty_panel, f"faulty LiDAR: {fault} S{severity}"),
                    add_label_above(radar_panel, "clean radar condition"),
                    add_reliability_colorbar(add_label_above(ideal_panel, "ideal LiDAR reliability")),
                    add_reliability_colorbar(add_label_above(prediction_panel, "radar-conditioned prediction")),
                    add_label_above(match_panel, f"localization match: {tolerance_cm} cm"),
                ]
            )
            destination = output_root / f"{index:04d}_{path.stem}_comparison.png"
            save_image(destination, comparison)
            print(f"Saved {index + 1}/{len(paths)}: {destination.name}")
    print(f"Comparisons saved to {output_root}")


if __name__ == "__main__":
    main()
