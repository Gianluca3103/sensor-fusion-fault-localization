"""Export additive Stage-II validation clouds for the frozen SVEFusion detector.

Inference loads aligned radar and cached faulty LiDAR; it never opens clean LiDAR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root
from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.data import fault_region_from_metadata
from models.radar_lidar_stage2_residual.model import ResidualRadarLidarStage2
from scripts.export_stage2_vod_detector import detector_cloud
from scripts.visualize_stage1_confidence_cloud import load_encoder
from scripts.visualize_stage2_reconstruction import load_faulty_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fault-samples-root", type=Path, required=True)
    parser.add_argument("--fault-pattern", default="*")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--limit", type=int, help="First N frames for smoke test only")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("model_type") != "stage2_residual_faulty_conditioned_v1" or saved.get("diffusion_enabled") is not False:
        raise ValueError("Expected a deterministic faulty-conditioned residual Stage-II checkpoint")
    values = dict(saved["config"])
    for key in ("channels", "query_radii_m", "expansion_zyx"):
        values[key] = tuple(values[key])
    config = ResidualStage2Config(**values)
    if args.fault_pattern != saved["fault_pattern"]:
        raise ValueError("Inference fault pattern differs from training")
    stage1_path = args.stage1_checkpoint or Path(saved["stage1_checkpoint"])
    stage1, stage1_saved = load_encoder(stage1_path, args.device)
    variant = saved["radar_variant"]
    trained_variant = stage1_saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != variant:
        raise ValueError("Stage-I and residual Stage-II radar variants differ")
    model = ResidualRadarLidarStage2(stage1.config.channels, config).to(args.device).eval()
    model.load_state_dict(saved["model"])
    public = resolve_vod_public_root(args.vod_root)
    official = (public / "lidar" / "ImageSets" / "val.txt").read_text(encoding="utf-8").split()
    if not official or len(official) != len(set(official)):
        raise ValueError("Official validation IDs are empty or duplicated")
    ids = official[:args.limit] if args.limit else official
    dataset = VoDStage1Dataset(public, "val", frame_ids=ids,
                               radar_variant=variant, include_clean=False)
    if {frame.frame_id for frame in dataset.frames} != set(ids) or len(dataset) != len(ids):
        raise ValueError("Radar dataset does not match official validation IDs")
    root = args.output_root / "lidar" / "reconstructed"
    destination_root = root / "training" / "velodyne"
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "export_manifest.json"
    provenance = {
        "mode": "lidar", "condition": "reconstructed", "splits": {"train": 0, "val": len(ids)},
        "forward_only": True, "export_type": "sve_reconstructed_validation_only",
        "reconstruction_model": "stage2_residual_faulty_conditioned_v1",
        "detector_input_mode": "merged", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint), "checkpoint_epoch": int(saved["epoch"]),
        "stage1_checkpoint": str(stage1_path.resolve()),
        "radar_variant_for_reconstruction": variant,
        "fault_samples_root": str(args.fault_samples_root.resolve()),
        "fault_pattern": args.fault_pattern, "generated_intensity": 0.0,
        "reconstruction_model_uses_faulty_lidar": True,
        "faulty_lidar_merged_after_inference": True,
        "detector_input_includes_faulty_lidar": True,
        "clean_lidar_used_at_inference": False, "diffusion_enabled": False,
        "measured_ray_suppression": True, "ray_tolerance_m": config.ray_tolerance_m,
        "complete_official_validation": args.limit is None, "frame_ids": ids,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in provenance.items():
            if previous.get(key) != value:
                raise ValueError(f"Existing export differs on {key}; use a new output root")
    else:
        if any(destination_root.glob("*.bin")):
            raise ValueError(f"Untracked detector clouds already exist: {destination_root}")
        manifest_path.write_text(json.dumps({**provenance, "complete": False}, indent=2), encoding="utf-8")
    for sample in tqdm(dataset, desc="Residual Stage II merged val", unit="frame"):
        frame_id = sample["frame_id"]
        destination = destination_root / f"{frame_id}.bin"
        if destination.exists():
            if destination.stat().st_size < 16 or destination.stat().st_size % 16:
                raise ValueError(f"Incomplete detector cloud: {destination}")
            continue
        faulty, _, meta = load_faulty_sample(args.fault_samples_root, "val", frame_id,
                                             args.fault_pattern)
        region = fault_region_from_metadata(meta, f"val/{frame_id}")
        batch = collate_stage1([sample])
        batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        evidence = stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)
        faulty_tensor = torch.from_numpy(faulty).to(args.device)[None]
        faulty_valid = torch.ones(faulty_tensor.shape[:2], device=args.device, dtype=torch.bool)
        output = model(evidence, stage1.config.grid, faulty_tensor, faulty_valid,
                       [region])
        generated = output.reconstructed_points_xyz.detach().cpu().numpy()
        points = detector_cloud(generated, faulty, mode="merged")
        temporary = destination.with_suffix(".bin.tmp")
        points.tofile(temporary)
        os.replace(temporary, destination)
    actual_files = {path.stem for path in destination_root.glob("*.bin")}
    if actual_files != set(ids):
        raise ValueError("Export does not contain exactly the requested validation IDs")
    manifest_path.write_text(json.dumps({**provenance, "complete": True}, indent=2), encoding="utf-8")
    print(f"Export complete: {len(ids)} frames, merged, {root}", flush=True)


if __name__ == "__main__":
    main()
