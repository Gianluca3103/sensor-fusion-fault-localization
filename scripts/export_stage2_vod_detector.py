"""Export Stage-II validation LiDAR for the frozen SVEFusion detector.

Stage I and Stage II see radar only. The optional faulty LiDAR merge occurs
after inference, and clean LiDAR is never opened by this export path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root
from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage2.config import Stage2Config
from models.radar_lidar_stage2.reconstruction_model import RadarLidarStage2
from scripts.export_vod_pvrcnn import detector_points
from scripts.visualize_stage1_confidence_cloud import load_encoder
from scripts.visualize_stage2_reconstruction import load_faulty_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detector_cloud(generated_xyz: np.ndarray, faulty_xyzi: np.ndarray | None, *,
                   mode: str) -> np.ndarray:
    """Create four-column detector input without consulting clean LiDAR.

    Stage II predicts XYZ only; generated reflectivity is explicitly zero.
    The merged mode retains the measured faulty XYZI rows unchanged.
    """
    xyz = np.asarray(generated_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Stage II must produce finite XYZ rows")
    generated = np.concatenate((xyz, np.zeros((len(xyz), 1), np.float32)), axis=1)
    if mode == "merged":
        if faulty_xyzi is None:
            raise ValueError("Merged mode requires matched faulty LiDAR")
        faulty = np.asarray(faulty_xyzi, dtype=np.float32)
        if faulty.ndim != 2 or faulty.shape[1] != 4 or not np.isfinite(faulty).all():
            raise ValueError("Faulty LiDAR must contain finite XYZI rows")
        generated = np.concatenate((faulty, generated), axis=0)
    elif mode != "generated-only":
        raise ValueError(f"Unknown export mode: {mode}")
    return detector_points(generated, None, forward_only=True).astype("<f4", copy=False)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("merged", "generated-only"), default="merged")
    parser.add_argument("--fault-samples-root", type=Path,
                        help="Required for merged mode; contains val/*.npz")
    parser.add_argument("--fault-pattern", default="*",
                        help="Select one fault variant per validation frame")
    parser.add_argument("--stage1-checkpoint", type=Path,
                        help="Override the path saved in the Stage-II checkpoint")
    parser.add_argument("--limit", type=int,
                        help="Export first N frames for a smoke test; not scoreable")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.mode == "merged" and args.fault_samples_root is None:
        parser.error("--fault-samples-root is required in merged mode")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_values = dict(saved["config"])
    for key in ("channels", "query_radii_m", "expansion_zyx"):
        cfg_values[key] = tuple(cfg_values[key])
    config = Stage2Config(**cfg_values)
    if saved.get("diffusion_enabled") is not False:
        raise ValueError("This exporter expects deterministic Stage II with diffusion disabled")
    stage1_path = args.stage1_checkpoint or Path(saved["stage1_checkpoint"])
    stage1, stage1_saved = load_encoder(stage1_path, args.device)
    variant = saved["radar_variant"]
    trained_variant = stage1_saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != variant:
        raise ValueError("Stage-I and Stage-II radar variants differ")
    model = RadarLidarStage2(stage1.config.channels, config).to(args.device).eval()
    model.load_state_dict(saved["model"])

    public = resolve_vod_public_root(args.vod_root)
    official = (public / "lidar" / "ImageSets" / "val.txt").read_text(
        encoding="utf-8").split()
    if not official or len(set(official)) != len(official):
        raise ValueError("Official validation IDs are missing or duplicated")
    ids = official[:args.limit] if args.limit else official
    dataset = VoDStage1Dataset(public, "val", frame_ids=ids,
                               radar_variant=variant, include_clean=False)
    actual = [frame.frame_id for frame in dataset.frames]
    if len(actual) != len(ids) or set(actual) != set(ids):
        raise ValueError("Radar dataset does not match requested official validation IDs")

    root = args.output_root / "lidar" / "reconstructed"
    destination_root = root / "training" / "velodyne"
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "export_manifest.json"
    provenance = {
        "mode": "lidar", "condition": "reconstructed",
        "splits": {"train": 0, "val": len(ids)}, "forward_only": True,
        "export_type": "sve_reconstructed_validation_only",
        "reconstruction_model": "stage2_radar_only_sparse_unet",
        "detector_input_mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch": int(saved["epoch"]),
        "stage1_checkpoint": str(stage1_path.resolve()),
        "radar_variant_for_reconstruction": variant,
        "fault_samples_root": (str(args.fault_samples_root.resolve())
                               if args.fault_samples_root else None),
        "fault_pattern": args.fault_pattern if args.mode == "merged" else None,
        "generated_intensity": 0.0,
        "reconstruction_model_uses_faulty_lidar": False,
        "faulty_lidar_merged_after_inference": args.mode == "merged",
        "detector_input_includes_faulty_lidar": args.mode == "merged",
        "clean_lidar_used_at_inference": False,
        "diffusion_enabled": False,
        "complete_official_validation": args.limit is None,
        "frame_ids": ids,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in provenance.items():
            if previous.get(key) != value:
                raise ValueError(f"Existing detector export differs on {key}; use a new output root")
    else:
        if any(destination_root.glob("*.bin")):
            raise ValueError(f"Untracked detector clouds already exist: {destination_root}")
        manifest_path.write_text(json.dumps({**provenance, "complete": False}, indent=2),
                                 encoding="utf-8")

    for sample in tqdm(dataset, desc=f"Stage II {args.mode} val", unit="frame"):
        frame_id = sample["frame_id"]
        destination = destination_root / f"{frame_id}.bin"
        if destination.exists():
            if destination.stat().st_size < 16 or destination.stat().st_size % 16:
                raise ValueError(f"Incomplete existing detector cloud: {destination}")
            continue
        batch = collate_stage1([sample])
        batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        evidence = stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)
        output = model(evidence, stage1.config.grid)
        generated = output.reconstructed_points_xyz.detach().cpu().numpy()
        faulty = None
        if args.mode == "merged":
            faulty, _, _ = load_faulty_sample(args.fault_samples_root, "val",
                                              frame_id, args.fault_pattern)
        points = detector_cloud(generated, faulty, mode=args.mode)
        temporary = destination.with_suffix(".bin.tmp")
        points.tofile(temporary)
        os.replace(temporary, destination)

    actual_files = {path.stem for path in destination_root.glob("*.bin")}
    if actual_files != set(ids):
        raise ValueError("Detector export does not contain exactly the requested frame IDs")
    manifest_path.write_text(json.dumps({**provenance, "complete": True}, indent=2),
                             encoding="utf-8")
    print(f"Export complete: {len(ids)} frames, {args.mode}, {root}", flush=True)


if __name__ == "__main__":
    main()
