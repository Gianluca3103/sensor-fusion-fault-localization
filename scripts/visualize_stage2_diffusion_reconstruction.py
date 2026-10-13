"""Inspect sparse diffusion additions beside VoD radar and clean LiDAR.

Clean LiDAR is loaded for display only. Inference receives radar and the saved
faulty LiDAR sample, exactly as in validation.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from models.radar_lidar_stage2_diffusion.model import ResidualSparseDiffusion
from models.radar_lidar_stage2_residual.config import ResidualStage2Config
from models.radar_lidar_stage2_residual.data import fault_region_from_metadata
from scripts.visualize_stage1_confidence_cloud import load_encoder, save_viewer, write_ply
from scripts.visualize_stage2_reconstruction import load_faulty_sample


def checkpoint_config(saved: dict) -> DiffusionConfig:
    if (saved.get("model_type") != "stage2_residual_sparse_diffusion_v1"
            or saved.get("diffusion_enabled") is not True):
        raise ValueError("Expected a sparse diffusion Stage-II checkpoint")
    residual = dict(saved["config"]["residual"])
    for key in ("channels", "query_radii_m", "expansion_zyx"):
        residual[key] = tuple(residual[key])
    diffusion = {key: value for key, value in saved["config"].items()
                 if key != "residual"}
    return DiffusionConfig(residual=ResidualStage2Config(**residual), **diffusion)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fault-samples-root", type=Path, required=True)
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--fault-pattern")
    parser.add_argument("--sampling-steps", type=int,
                        help="Defaults to all timesteps saved in the checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=("off", "bf16"),
                        help="Defaults to the precision saved in the checkpoint")
    parser.add_argument("--max-plot-points", type=int, default=75000,
                        help="Browser display limit for additions and overlays; PLY remains complete")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--z-min", type=float, default=-5.0)
    parser.add_argument("--z-max", type=float, default=7.0)
    args = parser.parse_args()

    if args.max_plot_points < 1 or len(set(args.frame_id)) != len(args.frame_id):
        parser.error("Display limit must be positive and frame IDs must be unique")
    limits = (args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)
    if any(lo >= hi for lo, hi in zip(limits[::2], limits[1::2])):
        parser.error("Each display minimum must be below its maximum")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_config(saved)
    if args.sampling_steps is not None and not 1 <= args.sampling_steps <= config.timesteps:
        parser.error(f"--sampling-steps must be between 1 and {config.timesteps}")
    amp = args.amp or saved.get("amp", "off")
    if amp == "bf16" and torch.device(args.device).type != "cuda":
        parser.error("bf16 autocast requires CUDA")
    fault_pattern = args.fault_pattern or saved["fault_pattern"]
    if fault_pattern != saved["fault_pattern"]:
        raise ValueError("Fault pattern differs from the Stage-II training run")
    if args.fault_samples_root.resolve() != Path(saved["fault_samples_root"]).resolve():
        raise ValueError("Fault sample root differs from the Stage-II training run")

    stage1_path = args.stage1_checkpoint or Path(saved["stage1_checkpoint"])
    stage1, stage1_saved = load_encoder(stage1_path, args.device)
    variant = saved["radar_variant"]
    trained_variant = stage1_saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != variant:
        raise ValueError("Stage-I and Stage-II radar variants differ")
    model = ResidualSparseDiffusion(stage1.config.channels, config).to(args.device).eval()
    model.load_state_dict(saved["model"])
    dataset = VoDStage1Dataset(args.vod_root, args.split, frame_ids=args.frame_id,
                               radar_variant=variant, include_clean=True)
    found = {frame.frame_id for frame in dataset.frames}
    missing = set(args.frame_id) - found
    if missing:
        raise ValueError(f"Frames absent from {args.split}: {sorted(missing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    for sample in dataset:
        frame_id = sample["frame_id"]
        faulty_xyzi, fault_path, fault_metadata = load_faulty_sample(
            args.fault_samples_root, args.split, frame_id, fault_pattern)
        fault_region = fault_region_from_metadata(fault_metadata, fault_path)
        batch = collate_stage1([sample])
        radar = batch["radar"].to(args.device)
        radar_valid = batch["radar_valid"].to(args.device)
        evidence = stage1(radar, radar_valid, defer_diagnostics=True)
        faulty = torch.from_numpy(faulty_xyzi)[None].to(args.device)
        faulty_valid = torch.ones(faulty.shape[:2], dtype=torch.bool, device=args.device)
        context = model.prepare(evidence, stage1.config.grid, faulty, faulty_valid,
                                [fault_region])
        precision = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                     if amp == "bf16" else nullcontext())
        with precision:
            output = model.sample(context,
                                  frame_seeds=[config.sample_seed + int(frame_id)],
                                  sampling_steps=args.sampling_steps)

        selected = ((output.addition_probability > config.residual.occupancy_threshold)
                    & output.coverage.may_add)
        added = output.reconstructed_points_xyz.detach().cpu().numpy()
        scores = output.addition_probability[selected].detach().cpu().numpy()
        if len(added) != len(scores):
            raise AssertionError("Diffusion point count differs from selected scores")
        radar_xyz = sample["radar"][:, :3].numpy()
        clean_xyz = sample["clean_lidar"][:, :3].numpy()
        faulty_xyz = faulty_xyzi[:, :3]
        merged = np.concatenate((faulty_xyz, added), axis=0)
        prefix = args.output_root / f"{frame_id}_stage2_diffusion"
        write_ply(prefix.with_name(prefix.name + "_radar.ply"), radar_xyz)
        write_ply(prefix.with_name(prefix.name + "_clean_lidar.ply"), clean_xyz)
        write_ply(prefix.with_name(prefix.name + "_faulty_lidar.ply"), faulty_xyz)
        write_ply(prefix.with_name(prefix.name + "_additions.ply"), added, scores)
        write_ply(prefix.with_name(prefix.name + "_merged.ply"), merged)
        counts = save_viewer(
            prefix.with_suffix(".html"), frame_id=frame_id, epoch=saved.get("epoch"),
            radar=radar_xyz, lidar=clean_xyz, sites=added, confidence=scores,
            limits=limits, max_points=args.max_plot_points, trained=True,
            calibrated=False, third_name="Diffusion additions",
            stage_name="Stage-II sparse diffusion",
            description="Drag to rotate · wheel or pinch to zoom · Shift-drag to pan. "
                        "The three panels share one metric camera. Toggle faulty LiDAR "
                        "over the additions to inspect the completed scan.",
            notice="Radar and faulty LiDAR are inference inputs. Clean LiDAR is shown "
                   "for comparison only. Colors are model addition scores, not "
                   "calibrated probabilities of correct geometry.",
            score_name="Addition score", overlay=faulty_xyz,
            overlay_name="faulty LiDAR", overlay_color="#bd79e3",
            overlay_checked=False,
        )
        prefix.with_suffix(".json").write_text(json.dumps({
            "frame_id": frame_id, "split": args.split,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": saved.get("epoch"),
            "stage1_checkpoint": str(stage1_path.resolve()),
            "radar_variant": variant, "fault_sample": str(fault_path),
            "fault": fault_metadata.get("fault"),
            "severity": fault_metadata.get("severity"),
            "diffusion_timesteps": config.timesteps,
            "sampling_steps": args.sampling_steps or config.timesteps,
            "candidate_sites": len(context.domain.coordinates),
            "predicted_additions": len(added),
            "faulty_points": len(faulty_xyz), "clean_points": len(clean_xyz),
            "radar_points": len(radar_xyz),
            "display_limits_xyz_m": limits, "display_counts": counts,
            "clean_lidar_used_at_inference": False,
        }, indent=2), encoding="utf-8")
        print(f"{frame_id}: {len(added):,} additions, {len(faulty_xyz):,} faulty points | "
              f"{prefix.with_suffix('.html')}", flush=True)


if __name__ == "__main__":
    main()
