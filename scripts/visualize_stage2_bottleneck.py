"""Export aligned Stage-I support, Stage-II additions, completed scan and errors."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage2_bottleneck.config import BottleneckDiffusionConfig
from models.radar_lidar_stage2_bottleneck.model import BottleneckSparseDiffusion
from models.radar_lidar_stage2_bottleneck.train import MODEL_TYPE, load_frozen_stage1
from models.radar_lidar_stage2_diffusion.cache import file_sha256
from models.radar_lidar_stage2_residual.data import fault_region_from_metadata
from scripts.visualize_stage1_confidence_cloud import save_viewer, write_ply
from scripts.visualize_stage2_reconstruction import load_faulty_sample


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--fault-samples-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--sampling-method", choices=("ddpm", "ddim"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("model_type") != MODEL_TYPE:
        parser.error("Expected a bottleneck sparse diffusion v2 checkpoint")
    config = BottleneckDiffusionConfig.from_dict(saved["config"])
    stage1_path = args.stage1_checkpoint or Path(saved["stage1_checkpoint"])
    if file_sha256(stage1_path) != saved["stage1_sha256"]:
        parser.error("Stage-I checkpoint content differs from Stage-II training")
    if args.fault_samples_root.resolve() != Path(saved["fault_samples_root"]).resolve():
        parser.error("Fault sample root differs from Stage-II training")
    stage1, _ = load_frozen_stage1(stage1_path, args.device, saved["radar_variant"])
    model = BottleneckSparseDiffusion(stage1.config.conditioning_channels, config).to(args.device).eval()
    model.load_state_dict(saved["model"])
    dataset = VoDStage1Dataset(args.vod_root, args.split,
                               frame_ids=args.frame_id, radar_variant=saved["radar_variant"],
                               include_clean=True)
    found = {frame.frame_id for frame in dataset.frames}
    if set(args.frame_id) != found:
        parser.error(f"Missing frame IDs: {sorted(set(args.frame_id) - found)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    grid = stage1.config.grid
    limits = (grid.minimum_xyz[0], grid.maximum_xyz[0],
              grid.minimum_xyz[1], grid.maximum_xyz[1],
              grid.minimum_xyz[2], grid.maximum_xyz[2])
    for sample in dataset:
        frame_id = sample["frame_id"]
        faulty_xyzi, fault_path, fault_metadata = load_faulty_sample(
            args.fault_samples_root, args.split, frame_id, saved["fault_pattern"])
        region = fault_region_from_metadata(fault_metadata, fault_path)
        batch = collate_stage1([sample])
        radar = batch["radar"].to(args.device)
        radar_valid = batch["radar_valid"].to(args.device)
        evidence = stage1(radar, radar_valid, defer_diagnostics=True)
        faulty = torch.from_numpy(faulty_xyzi)[None].to(args.device)
        valid = torch.ones(faulty.shape[:2], dtype=torch.bool, device=args.device)
        context = model.prepare(evidence, grid, faulty, valid, [region])
        precision = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                     if saved.get("amp") == "bf16" and args.device.startswith("cuda")
                     else nullcontext())
        with precision:
            output = model.sample(context, frame_seeds=[config.sample_seed + int(frame_id)],
                                  method=args.sampling_method, sampling_steps=args.sampling_steps)
        selected = ((output.addition_probability > config.residual.occupancy_threshold)
                    & output.coverage.may_add)
        added = output.reconstructed_points_xyz.detach().cpu().numpy()
        scores = output.addition_probability[selected].detach().cpu().numpy()
        if len(added) != len(scores):
            raise AssertionError("Exported addition scores and points differ")
        radar_xyz = sample["radar"][:, :3].numpy()
        clean_xyz = sample["clean_lidar"][:, :3].numpy()
        faulty_xyz = faulty_xyzi[:, :3]
        completed = np.concatenate((faulty_xyz, added), axis=0)
        support_xyz = context.domain.centers_xyz.detach().cpu().numpy()
        support_score = context.domain.confidence.detach().cpu().numpy()
        raw_xyz = evidence.confidence_region.centers_xyz(grid).detach().cpu().numpy()
        raw_score = evidence.confidence_region.features[:, 0].detach().cpu().numpy()
        error_m = (cKDTree(clean_xyz).query(added, workers=1)[0].astype(np.float32)
                   if len(clean_xyz) and len(added) else np.full(len(added), np.inf, np.float32))
        error_color = np.minimum(error_m, 1.).astype(np.float32)
        missing_distance = (cKDTree(completed).query(clean_xyz, workers=1)[0].astype(np.float32)
                            if len(completed) and len(clean_xyz) else
                            np.full(len(clean_xyz), np.inf, np.float32))
        missing_clean = clean_xyz[missing_distance > .2]
        prefix = args.output_root / f"{frame_id}_stage2_bottleneck"
        for suffix, xyz, value in (
            ("radar", radar_xyz, None), ("clean_lidar", clean_xyz, None),
            ("faulty_lidar", faulty_xyz, None),
            ("stage1_raw_support", raw_xyz, raw_score),
            ("stage1_interpolated_support", support_xyz, support_score),
            ("additions", added, scores), ("completed", completed, None),
            ("addition_error_clipped_1m", added, error_color),
            ("missing_clean_0p2m", missing_clean, None),
        ):
            write_ply(prefix.with_name(prefix.name + f"_{suffix}.ply"), xyz, value)
        save_viewer(prefix.with_suffix(".html"), frame_id=frame_id, epoch=saved["epoch"],
                    radar=radar_xyz, lidar=clean_xyz, sites=added, confidence=scores,
                    limits=limits, max_points=None, trained=True, calibrated=False,
                    third_name="Stage-II additions", stage_name="Stage-II bottleneck diffusion",
                    description="Shared metric camera. Toggle faulty LiDAR over additions to inspect the completed scan.",
                    notice="Clean LiDAR is displayed only; inference uses radar and faulty LiDAR.",
                    score_name="Addition probability", overlay=faulty_xyz,
                    overlay_name="faulty LiDAR", overlay_color="#bd79e3",
                    overlay_checked=False)
        save_viewer(prefix.with_name(prefix.name + "_stage1.html"),
                    frame_id=frame_id, epoch=saved["epoch"], radar=radar_xyz,
                    lidar=clean_xyz, sites=support_xyz, confidence=support_score,
                    limits=limits, max_points=None, trained=True, calibrated=False,
                    third_name="Edge-aware Stage-I support", stage_name="Stage-I repair support",
                    description="The region is eligible for reconstruction; it is not measured occupancy.",
                    notice="Raw Stage-I seed voxels are exported as a separate PLY.",
                    score_name="Support score")
        save_viewer(prefix.with_name(prefix.name + "_error.html"),
                    frame_id=frame_id, epoch=saved["epoch"], radar=radar_xyz,
                    lidar=clean_xyz, sites=added, confidence=error_color,
                    limits=limits, max_points=None, trained=True, calibrated=False,
                    third_name="Addition error", stage_name="Stage-II geometric error",
                    description="Each addition is colored by distance to the nearest clean LiDAR point, clipped at 1 m.",
                    notice="Clean LiDAR is used for evaluation only.",
                    score_name="Nearest clean distance / 1 m")
        save_viewer(prefix.with_name(prefix.name + "_completed_overlay.html"),
                    frame_id=frame_id, epoch=saved["epoch"], radar=radar_xyz,
                    lidar=clean_xyz, sites=completed,
                    confidence=np.concatenate((np.zeros(len(faulty_xyz), np.float32),
                                               scores)).astype(np.float32),
                    limits=limits, max_points=None, trained=True, calibrated=False,
                    third_name="Completed LiDAR", stage_name="Stage-II completed scan",
                    description="Toggle clean LiDAR over the completed scan. The camera and physical coordinates are shared.",
                    notice="Clean LiDAR is evaluation-only; no point limit is applied.",
                    score_name="Addition score (faulty points shown at 0)",
                    overlay=clean_xyz, overlay_name="clean LiDAR",
                    overlay_color="#45c97a", overlay_checked=False)
        prefix.with_suffix(".json").write_text(json.dumps({
            "frame_id": frame_id, "fault_kind": fault_metadata.get("fault"),
            "stage1_checkpoint": str(stage1_path.resolve()),
            "stage2_checkpoint": str(args.checkpoint.resolve()),
            "stage2_epoch": saved["epoch"], "sampling": output.metadata,
            "radar_points": len(radar_xyz), "clean_points": len(clean_xyz),
            "faulty_points": len(faulty_xyz), "raw_support_sites": len(raw_xyz),
            "interpolated_support_sites": len(support_xyz),
            "additions": len(added), "completed_points": len(completed),
            "clean_points_farther_than_0p2m_from_completed": len(missing_clean),
            "addition_error_mean_m": (float(np.mean(error_m[np.isfinite(error_m)]))
                                      if np.isfinite(error_m).any() else None),
            "clean_lidar_used_for_inference": False,
        }, indent=2), encoding="utf-8")
        print(prefix.with_suffix(".html"), flush=True)


if __name__ == "__main__":
    main()
