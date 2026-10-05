"""Render joint relation/diffusion validation scans in a rotatable 3D viewer.

Clean LiDAR and labels are loaded only for comparison after radar/faulty-LiDAR
inference. This viewer never passes either one to the reconstruction models.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from models.two_stage_reconstruction_head.cross_modal_data import CrossModalVoDDataset
from models.two_stage_reconstruction_head.cross_modal_encoders import EncoderGrid
from models.two_stage_reconstruction_head.diffusion_process.joint_relation_diffusion import (
    RadarRelationDiffusion, RadarRelationEncoder, sample_joint_full_scan,
)
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.voxelization.inputs import read_sample_metadata
from scripts.train_radar_gated_ray_diffusion import _paths
from scripts.visualize_range_view_reconstruction import (
    _load_annotated_boxes, _radar_box_stats, _save_interactive_html, _save_ply,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--return-threshold", type=float, default=0.5)
    parser.add_argument("--max-plot-points", type=int, default=20000)
    parser.add_argument("--device", default="cpu",
                        help="Use cpu alongside training, or cuda for faster sampling")
    args = parser.parse_args()
    if (not args.sample_indices or min(args.sample_indices) < 0 or
            args.steps < 2 or not 0 <= args.return_threshold <= 1 or
            args.max_plot_points < 1):
        parser.error("Invalid sample, DDIM, return threshold, or plot setting")
    return args


def main() -> None:
    args = _arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("stage") not in (
            "joint_radar_relation_diffusion_v1",
            "joint_radar_relation_diffusion_v2"):
        raise ValueError("Expected a joint relation/diffusion checkpoint")
    geometry = RangeGeometry(**saved["geometry_parameters"])
    if asdict(geometry) != saved["geometry_parameters"]:
        raise ValueError("Checkpoint geometry could not be restored exactly")
    settings = saved["settings"]
    device = torch.device(args.device)
    relation = RadarRelationEncoder(
        geometry, EncoderGrid(), width=settings["width"],
        history_scans=20).to(device).eval()
    diffusion = RadarRelationDiffusion(
        geometry, relation_width=settings["width"],
        hidden=settings["hidden"],
        timesteps=settings["timesteps"],
        objective_version=("legacy_v1" if saved["stage"].endswith("_v1")
                           else "metric_depth_v2"),
        metric_depth_weight=settings.get("metric_depth_weight", 0.3)).to(device).eval()
    relation.load_state_dict(saved["relation"])
    diffusion.load_state_dict(saved["diffusion"])

    all_paths = _paths(args.samples_root, "val", None)
    if max(args.sample_indices) >= len(all_paths):
        raise IndexError(f"Validation index must be below {len(all_paths)}")
    paths = [all_paths[index] for index in args.sample_indices]
    dataset = CrossModalVoDDataset(
        paths, args.vod_root, radar_variant=saved["radar_variant"],
        include_clean=True,
        radar_height_filter=saved["radar_height_filter"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        item = dataset[index]
        observed = item["observed_lidar"][None].to(device)
        radar = item["radar"][None].to(device)
        observed_valid = torch.ones(observed.shape[:2], dtype=torch.bool,
                                    device=device)
        radar_valid = torch.ones(radar.shape[:2], dtype=torch.bool,
                                 device=device)
        generator = torch.Generator(device=device).manual_seed(
            42 + int(item["frame_id"]))
        with torch.inference_mode():
            cloud = sample_joint_full_scan(
                relation, diffusion, radar, radar_valid,
                observed, observed_valid,
                tile_rows=settings["tile_rows"],
                tile_cols=settings["tile_cols"],
                steps=args.steps,
                return_threshold=args.return_threshold,
                generator=generator)
        generated = cloud[len(item["observed_lidar"]):].cpu().numpy()
        faulty = item["observed_lidar"].numpy()
        clean = item["clean_lidar"].numpy()
        radar_points = item["radar"].numpy()
        faulty = faulty[faulty[:, 0] >= 0]
        clean = clean[clean[:, 0] >= 0]
        generated = generated[generated[:, 0] >= 0]
        radar_points = radar_points[radar_points[:, 0] >= 0]
        reconstructed = np.concatenate((faulty, generated), axis=0)

        frame = dataset.frames[(item["split"], item["frame_id"])]
        label = (frame.lidar_path.parent.parent / "label_2" /
                 f"{item['frame_id']}.txt")
        boxes = (_load_annotated_boxes({
            "dataset": "View-of-Delft",
            "source_relative_path": str(frame.lidar_path),
        }) if label.is_file() else [])
        radar_stats = _radar_box_stats(radar_points, boxes) if boxes else None
        metadata = read_sample_metadata(path)
        destination = args.output_root / path.stem
        destination.mkdir(parents=True, exist_ok=True)
        for name, points in (("faulty", faulty), ("clean", clean),
                             ("generated", generated),
                             ("reconstructed", reconstructed),
                             ("radar", radar_points[:, :3])):
            _save_ply(destination / f"{name}.ply", points)
        viewer = destination / "interactive.html"
        _save_interactive_html(
            viewer, faulty=faulty, clean=clean, original=faulty,
            generated=generated, sample_name=path.stem,
            fault=str(metadata.get("fault", "unknown")),
            epoch=int(saved["epoch"]), max_plot_points=args.max_plot_points,
            radar=radar_points, boxes=boxes, radar_stats=radar_stats)
        (destination / "metadata.json").write_text(json.dumps({
            "checkpoint": str(args.checkpoint.resolve()),
            "epoch": int(saved["epoch"]), "sample": str(path),
            "frame_id": item["frame_id"],
            "faulty_points": len(faulty), "clean_points": len(clean),
            "generated_points": len(generated),
            "reconstructed_points": len(reconstructed),
            "radar_points": len(radar_points),
            "radar_box_stats": radar_stats,
            "clean_lidar_used_at_inference": False,
        }, indent=2), encoding="utf-8")
        print(f"Saved {viewer} (+{len(generated)} generated points)", flush=True)


if __name__ == "__main__":
    main()
