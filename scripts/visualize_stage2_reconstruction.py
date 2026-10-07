"""Save interactive radar/clean/reconstructed VoD comparisons from Stage II."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage2.config import Stage2Config
from models.radar_lidar_stage2.reconstruction_model import RadarLidarStage2
from scripts.visualize_stage1_confidence_cloud import load_encoder, save_viewer, write_ply


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path,
                        help="Override the Stage-I checkpoint path recorded in Stage II")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-plot-points", type=int, default=30000)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--z-min", type=float, default=-5.0)
    parser.add_argument("--z-max", type=float, default=7.0)
    args = parser.parse_args()
    if args.max_plot_points < 1:
        parser.error("--max-plot-points must be positive")
    limits = (args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)
    if any(low >= high for low, high in zip(limits[::2], limits[1::2])):
        parser.error("Each display minimum must be below its maximum")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_data = dict(saved["config"])
    for key in ("channels", "query_radii_m", "expansion_zyx"):
        config_data[key] = tuple(config_data[key])
    config = Stage2Config(**config_data)
    stage1_path = args.stage1_checkpoint or Path(saved["stage1_checkpoint"])
    stage1, stage1_saved = load_encoder(stage1_path, args.device)
    variant = saved["radar_variant"]
    stage1_variant = stage1_saved.get("data", {}).get("radar_variant")
    if stage1_variant and stage1_variant != variant:
        raise ValueError("Stage-I and Stage-II radar variants differ")
    model = RadarLidarStage2(stage1.config.channels, config).to(args.device).eval()
    model.load_state_dict(saved["model"])
    dataset = VoDStage1Dataset(args.vod_root, args.split, frame_ids=args.frame_id,
                               radar_variant=variant)
    found = {frame.frame_id for frame in dataset.frames}
    missing = set(args.frame_id) - found
    if missing:
        raise ValueError(f"Frame IDs absent from {args.split}: {sorted(missing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    for sample in dataset:
        batch = collate_stage1([sample])
        batch = {name: value.to(args.device) if isinstance(value, torch.Tensor) else value
                 for name, value in batch.items()}
        evidence = stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)
        output = model(evidence, stage1.config.grid)
        selected = output.occupancy_probability > config.occupancy_threshold
        occupancy_score = output.occupancy_probability[selected].detach().cpu().numpy()
        radar = sample["radar"][:, :3].numpy()
        clean = sample["clean_lidar"][:, :3].numpy()
        reconstructed = output.reconstructed_points_xyz.detach().cpu().numpy()
        prefix = args.output_root / f"{sample['frame_id']}_stage2"
        write_ply(prefix.with_name(prefix.name + "_radar.ply"), radar)
        write_ply(prefix.with_name(prefix.name + "_clean_lidar.ply"), clean)
        write_ply(prefix.with_name(prefix.name + "_reconstructed.ply"), reconstructed,
                  occupancy_score)
        counts = save_viewer(
            prefix.with_suffix(".html"), frame_id=sample["frame_id"], epoch=saved.get("epoch"),
            radar=radar, lidar=clean, sites=reconstructed, confidence=occupancy_score,
            limits=limits, max_points=args.max_plot_points, trained=True, calibrated=False,
            third_name="Reconstructed LiDAR", stage_name="Stage-II deterministic reconstruction",
            description="Drag to rotate · wheel or pinch to zoom · Shift-drag to pan. "
                        "All three panels share one camera. Radar can be overlaid on reconstruction.",
            notice="Predicted points come from radar only. Clean LiDAR is shown for comparison; "
                   "diffusion is disabled. Occupancy scores are not calibrated as probabilities "
                   "of a correct surface.",
            score_name="Occupancy score")
        metadata = {"frame_id": sample["frame_id"], "split": args.split,
                    "stage2_checkpoint": str(args.checkpoint.resolve()),
                    "stage1_checkpoint": str(stage1_path.resolve()),
                    "stage2_epoch": saved.get("epoch"), "diffusion_enabled": False,
                    "radar_variant": variant, "candidate_sites": len(output.candidate_coordinates),
                    "reconstructed_points": len(reconstructed), "clean_points": len(clean),
                    "radar_points": len(radar), "occupancy_threshold": config.occupancy_threshold,
                    "display_limits_xyz_m": limits, "display_counts": counts,
                    "note": "Display clouds are capped; PLY files contain all points. "
                            "Reconstruction excludes any surviving faulty LiDAR."}
        prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"{sample['frame_id']}: {len(reconstructed):,} reconstructed points | "
              f"{prefix.with_suffix('.html')}", flush=True)


if __name__ == "__main__":
    main()
