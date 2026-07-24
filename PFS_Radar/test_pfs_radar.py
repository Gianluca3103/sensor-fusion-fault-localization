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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PFS_Radar.pfs_radar_model import load_model_checkpoint
from PFS_Radar.radar_data import (
    filter_samples_with_radar_cache,
    radar_cache_path,
    radar_cache_requirements_from_checkpoint,
)
from Fault_Localization_Model.sample_utils import (
    validate_heatmap_array,
    validate_radar_array,
    validate_rgb_array,
)
from Fault_Localization_Model.model_blocks import resize_reliability_map
from PFS.training_utils import resolve_device
from Fault_Localization_Model.visualization_utils import (
    add_label_above,
    add_reliability_colorbar,
    blue_red_reliability,
    draw_cell_boundaries,
    localization_match_overlay,
    save_image,
    side_by_side,
)


def radar_to_rgb(radar_bev):
    radar_bev = validate_radar_array(radar_bev, path="<in-memory radar BEV>")
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


def pool_map(array, grid_size):
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    return F.adaptive_avg_pool2d(tensor, (grid_size, grid_size))[0, 0].numpy()


def upscale_rgb(array, size):
    return np.asarray(
        Image.fromarray(array).resize(
            (size[1], size[0]),
            Image.Resampling.NEAREST,
        )
    )


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
    if args.max_images < 1:
        parser.error("--max-images must be at least 1")
    if (
        args.resize_height < 1
        or args.resize_width < 1
        or args.visual_grid_size < 1
    ):
        parser.error("Resize dimensions and --visual-grid-size must be positive")
    if args.visual_grid_size > min(args.resize_height, args.resize_width):
        parser.error(
            "--visual-grid-size cannot exceed the smaller resized input dimension"
        )
    if not 0.0 < args.prediction_threshold < 1.0:
        parser.error("--prediction-threshold must lie strictly between 0 and 1")
    if not 0.0 <= args.target_fault_threshold < 1.0:
        parser.error("--target-fault-threshold must lie in [0,1)")
    if args.localization_tolerance_m < 0.0:
        parser.error("--localization-tolerance-m must be non-negative")

    device = resolve_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    cache_requirements = radar_cache_requirements_from_checkpoint(checkpoint)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths, missing_paths = filter_samples_with_radar_cache(
        sorted(Path(args.test_root).glob("*.npz")),
        Path(args.radar_root),
        **cache_requirements,
    )
    if missing_paths:
        print(f"Skipping {len(missing_paths)} samples without aligned radar cache")
    paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"No .npz samples found in {args.test_root}")

    size = (args.resize_height, args.resize_width)
    with torch.inference_mode():
        for index, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as data:
                faulty_rgb = validate_rgb_array(
                    data["faulty_rgb"], name="faulty_rgb", path=path
                ).astype(np.uint8)
                target = validate_heatmap_array(
                    data["fault_heatmap"], path=path
                ).astype(np.float32)
                metadata = json.loads(str(data["metadata_json"]))
            radar_path = radar_cache_path(Path(args.radar_root), metadata)
            with np.load(radar_path, allow_pickle=False) as data:
                radar_bev = validate_radar_array(
                    data["radar_bev"], path=radar_path
                ).astype(np.float32)

            lidar_chw = faulty_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            lidar = resize_chw(lidar_chw, size, "bilinear").unsqueeze(0).to(device)
            radar = resize_chw(radar_bev, size, "bilinear").unsqueeze(0).to(device)
            prediction = torch.sigmoid(model(lidar, radar))[0, 0].cpu().numpy()
            target_tensor = resize_reliability_map(
                torch.from_numpy(target[None])[None],
                size,
            )[0, 0].numpy()
            target_grid = pool_map(target_tensor, args.visual_grid_size)
            prediction_grid = pool_map(prediction, args.visual_grid_size)

            faulty_panel = np.asarray(
                Image.fromarray(faulty_rgb).resize((args.resize_width, args.resize_height), Image.Resampling.BILINEAR)
            )
            radar_panel = radar_to_rgb(radar.cpu().numpy()[0])
            ideal_panel = draw_cell_boundaries(
                upscale_rgb(blue_red_reliability(target_grid), size),
                args.visual_grid_size,
            )
            prediction_panel = draw_cell_boundaries(
                upscale_rgb(blue_red_reliability(prediction_grid), size),
                args.visual_grid_size,
            )
            match_panel = draw_cell_boundaries(
                upscale_rgb(
                    localization_match_overlay(
                        target_grid,
                        prediction_grid,
                        metadata,
                        prediction_threshold=args.prediction_threshold,
                        target_fault_threshold=args.target_fault_threshold,
                        tolerance_m=args.localization_tolerance_m,
                    ),
                    size,
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
