"""Export joint radar-relation diffusion scans for matched SVEFusion validation.

This inference path opens no clean LiDAR and preserves all faulty LiDAR points.
The output is compatible with prepare_vod_official_faults.py's reconstructed
validation input. Use a fresh output root for each checkpoint or threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root
from models.two_stage_reconstruction_head.cross_modal_data import (
    CrossModalVoDDataset, collate_cross_modal,
)
from models.two_stage_reconstruction_head.cross_modal_encoders import EncoderGrid
from models.two_stage_reconstruction_head.diffusion_process.joint_relation_diffusion import (
    RadarRelationDiffusion, RadarRelationEncoder, sample_joint_full_scan,
)
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from scripts.export_vod_pvrcnn import detector_points
from scripts.train_radar_gated_ray_diffusion import _paths


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--return-threshold", type=float, default=0.5)
    parser.add_argument("--tile-rows", type=int)
    parser.add_argument("--tile-cols", type=int)
    parser.add_argument("--limit", type=int,
                        help="Smoke-test first N validation frames; not scoreable")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic sampling seed combined with frame ID")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.steps < 2 or not 0 <= args.return_threshold <= 1 or
            args.num_workers < 0 or
            (args.limit is not None and args.limit < 1)):
        parser.error("Invalid sampling settings")
    return args


def main() -> None:
    args = _arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("stage") != "joint_radar_relation_diffusion_v1":
        raise ValueError("Checkpoint is not the joint relation/diffusion model")
    geometry = RangeGeometry(**saved["geometry_parameters"])
    if asdict(geometry) != saved["geometry_parameters"]:
        raise ValueError("Checkpoint geometry could not be restored exactly")
    settings = saved["settings"]
    tile_rows = args.tile_rows or settings["tile_rows"]
    tile_cols = args.tile_cols or settings["tile_cols"]
    relation = RadarRelationEncoder(
        geometry, EncoderGrid(), width=settings["width"],
        history_scans=20).to(device).eval()
    diffusion = RadarRelationDiffusion(
        geometry, relation_width=settings["width"],
        hidden=settings["hidden"],
        timesteps=settings["timesteps"]).to(device).eval()
    relation.load_state_dict(saved["relation"])
    diffusion.load_state_dict(saved["diffusion"])
    public = resolve_vod_public_root(args.vod_root)
    paths = _paths(args.samples_root, "val", args.limit)
    dataset = CrossModalVoDDataset(
        paths, public, radar_variant=saved["radar_variant"],
        include_clean=False,
        radar_height_filter=saved["radar_height_filter"])
    if args.limit is None:
        official = (public / "lidar" / "ImageSets" / "val.txt").read_text(
            encoding="utf-8").split()
        actual = [key[1] for key in dataset.keys]
        if len(actual) != len(official) or set(actual) != set(official):
            raise ValueError("Export requires every official validation frame exactly once")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=collate_cross_modal)
    root = args.output_root / "lidar" / "reconstructed"
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "lidar", "condition": "reconstructed",
        "splits": {"train": 0, "val": len(dataset)},
        "forward_only": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": saved["epoch"],
        "model": "joint_radar_relation_diffusion_v1",
        "radar_variant": saved["radar_variant"],
        "radar_height_filter": saved["radar_height_filter"],
        "steps": args.steps,
        "return_threshold": args.return_threshold,
        "tile_rows": tile_rows, "tile_cols": tile_cols,
        "seed": args.seed,
        "support_policy": "measured_radar_and_missing_observed_lidar",
        "clean_lidar_used_at_inference": False,
        "empty_cloud_sentinel": [0.01, 0.0, -2.9, 0.0],
        "export_type": "sve_reconstructed_validation_only",
    }
    manifest_path = root / "export_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in manifest.items():
            if previous.get(key) != value:
                raise ValueError(f"Existing export disagrees on {key}; use a fresh output root")
    else:
        manifest_path.write_text(json.dumps({**manifest, "complete": False}, indent=2),
                                 encoding="utf-8")
    for batch in tqdm(loader, desc="joint reconstructed val", leave=True):
        frame_id = batch["frame_id"][0]
        destination = root / "training" / "velodyne" / f"{frame_id}.bin"
        if destination.exists():
            if destination.stat().st_size < 16 or destination.stat().st_size % 16:
                raise ValueError(f"Incomplete existing detector cloud: {destination}")
            continue
        radar = batch["radar"].to(device)
        radar_valid = batch["radar_valid"].to(device)
        observed = batch["observed_lidar"].to(device)
        observed_valid = batch["observed_lidar_valid"].to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + int(frame_id))
        with torch.inference_mode():
            cloud = sample_joint_full_scan(
                relation, diffusion, radar, radar_valid,
                observed, observed_valid,
                tile_rows=tile_rows, tile_cols=tile_cols,
                steps=args.steps,
                return_threshold=args.return_threshold,
                generator=generator)
        points = cloud.cpu().numpy().astype(np.float32)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".bin.tmp")
        detector_points(points, None, forward_only=True).astype("<f4").tofile(temporary)
        os.replace(temporary, destination)
    manifest_path.write_text(json.dumps({**manifest, "complete": True}, indent=2),
                             encoding="utf-8")
    print(f"Exported {len(dataset)} validation frames to {root}", flush=True)


if __name__ == "__main__":
    main()
