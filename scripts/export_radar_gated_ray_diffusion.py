"""Export validation scans from a trained radar-gated range-view checkpoint.

The output layout is accepted by SVEFusion prepare_vod_official_faults.py.
Clean LiDAR is never opened. Each output contains the exact faulty points plus
radar-supported additions from the calibrated diffusion model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root
from models.two_stage_reconstruction_head.cross_modal_data import CrossModalVoDDataset
from models.two_stage_reconstruction_head.cross_modal_encoders import EncoderGrid
from models.two_stage_reconstruction_head.diffusion_process.ray_view_diffusion import (
    RadarGatedRayDiffusion, sample_full_scan,
)
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.ray_depth_attention import RayDepthBlueprintModel
from models.two_stage_reconstruction_head.voxelization.inputs import read_sample_metadata


def _sample_index(root: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in sorted((root / "val").rglob("*.npz")):
        metadata = read_sample_metadata(path)
        frame_id = str(metadata.get("frame_id", "")).zfill(5)
        if metadata.get("split") != "val" or not metadata.get("range_view_full_scan"):
            raise ValueError(f"Expected a full-scan VoD validation artifact: {path}")
        if frame_id in selected:
            raise ValueError(f"Multiple fault samples for validation frame {frame_id}")
        selected[frame_id] = path
    return selected


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", help="Defaults to the checkpoint's training radar variant")
    parser.add_argument("--radar-height-filter", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Defaults to the checkpoint's observed-LiDAR height filter")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--return-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="Only for a partial smoke export")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.steps < 2 or not 0 <= args.return_threshold <= 1 or
            (args.limit is not None and args.limit < 1)):
        parser.error("Invalid sampling steps, return threshold, or limit")
    return args


def main() -> None:
    args = _arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_radar_variant = checkpoint.get("radar_variant", "radar_20frames_verified")
    if args.radar_variant is not None and args.radar_variant != checkpoint_radar_variant:
        raise ValueError("Export radar variant disagrees with the training checkpoint")
    radar_variant = checkpoint_radar_variant
    radar_height_filter = bool(checkpoint.get("radar_height_filter", False))
    if (args.radar_height_filter is not None and
            args.radar_height_filter != radar_height_filter):
        raise ValueError("Export radar height filter disagrees with the training checkpoint")
    parameters = checkpoint["geometry_parameters"]
    geometry = RangeGeometry(**parameters)
    settings = checkpoint["settings"]
    calibration = checkpoint.get("calibration")
    if calibration is None or "threshold" not in calibration:
        raise ValueError("Checkpoint lacks validation-calibrated reliability threshold")
    threshold = float(calibration["threshold"])
    blueprint = RayDepthBlueprintModel(
        geometry, EncoderGrid(), width=int(settings["width"]),
        history_scans=20,
    ).to(device).eval()
    diffusion = RadarGatedRayDiffusion(
        geometry, blueprint_width=int(settings["width"]),
        hidden=int(settings["hidden"]), timesteps=int(settings["timesteps"]),
        max_correction_m=float(settings["max_correction_m"]),
    ).to(device).eval()
    blueprint.load_state_dict(checkpoint["blueprint"])
    diffusion.load_state_dict(checkpoint["diffusion"])

    public = resolve_vod_public_root(args.vod_root)
    official = (public / "lidar" / "ImageSets" / "val.txt").read_text(
        encoding="utf-8").split()
    if len(set(official)) != len(official):
        raise ValueError("Official VoD validation IDs contain duplicates")
    selected = _sample_index(args.samples_root)
    if set(selected) != set(official):
        missing = sorted(set(official) - set(selected))
        extra = sorted(set(selected) - set(official))
        raise ValueError(f"Fault cache does not match official validation: "
                         f"missing={missing[:5]}, extra={extra[:5]}")
    ids = official[:args.limit] if args.limit else official
    paths = [selected[frame_id] for frame_id in ids]
    dataset = CrossModalVoDDataset(
        paths, public, radar_variant=radar_variant, include_clean=False,
        radar_height_filter=radar_height_filter)
    root = args.output_root / "lidar" / "reconstructed"
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "calibration": calibration,
        "steps": args.steps,
        "return_threshold": args.return_threshold,
        "radar_variant": radar_variant,
        "radar_height_filter": radar_height_filter,
        "frame_ids": ids,
        "forward_only": True,
    }
    provenance_file = root / "run_provenance.json"
    if provenance_file.exists():
        if json.loads(provenance_file.read_text(encoding="utf-8")) != provenance:
            raise ValueError(f"Existing export has different provenance: {root}")
    else:
        if list((root / "training" / "velodyne").glob("*.bin")):
            raise ValueError(f"Untracked detector clouds already exist: {root}")
        provenance_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    destination_root = root / "training" / "velodyne"
    destination_root.mkdir(parents=True, exist_ok=True)
    tiles_per_frame = (math.ceil(geometry.shape[0] / int(settings["tile_rows"])) *
                       math.ceil(geometry.shape[1] / int(settings["tile_cols"])))
    progress = tqdm(total=len(ids) * tiles_per_frame, desc="reconstruction",
                    unit="tile", disable=not sys.stderr.isatty())
    for index, frame_id in enumerate(ids, 1):
        destination = destination_root / f"{frame_id}.bin"
        if destination.exists():
            if destination.stat().st_size % 16:
                raise ValueError(f"Incomplete existing detector cloud: {destination}")
            progress.update(tiles_per_frame)
            continue
        item = dataset[index - 1]
        if item["frame_id"] != frame_id:
            raise ValueError("Dataset frame order differs from official validation")
        radar = item["radar"].unsqueeze(0).to(device)
        observed = item["observed_lidar"].unsqueeze(0).to(device)
        merged = sample_full_scan(
            blueprint, diffusion, radar,
            torch.ones(radar.shape[:2], dtype=torch.bool, device=device),
            observed, torch.ones(observed.shape[:2], dtype=torch.bool, device=device),
            tile_rows=int(settings["tile_rows"]),
            tile_cols=int(settings["tile_cols"]),
            steps=args.steps, reliability_threshold=threshold,
            return_threshold=args.return_threshold,
            on_tile=progress.update,
        )
        if threshold == 1.0:
            progress.update(tiles_per_frame)
        points = merged.detach().cpu().numpy().astype("<f4", copy=False)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError(f"Model produced invalid LiDAR points for {frame_id}")
        points = np.ascontiguousarray(points[points[:, 0] >= 0])
        if len(points) == 0:
            # The detector's KITTI loader requires a nonempty point cloud.
            points = np.asarray([[0.01, 0.0, -2.9, 0.0]], dtype="<f4")
        temporary = destination.with_suffix(".bin.tmp")
        points.tofile(temporary)
        os.replace(temporary, destination)
        if index % 25 == 0 or index == len(ids):
            print(f"Reconstructed validation LiDAR: {index}/{len(ids)}", flush=True)
    progress.close()
    manifest = {
        "mode": "lidar", "condition": "reconstructed",
        "splits": {"train": 0, "val": len(ids)},
        "forward_only": True, "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "export_type": "sve_reconstructed_validation_only",
        "reconstruction_model": "radar_gated_ray_diffusion",
        "radar_variant": radar_variant,
        "radar_height_filter": radar_height_filter,
        "reliability_threshold": threshold,
        "complete_official_validation": args.limit is None,
    }
    (root / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Export complete: {root}", flush=True)


if __name__ == "__main__":
    main()
