"""Held-out Stage-II metrics, evidence ablations, and XYZ inspection clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from scripts.visualize_stage1_confidence_cloud import load_encoder, write_ply
from .config import Stage2Config
from .metrics import Stage2MetricAccumulator, clean_points_in_domain
from .reconstruction_model import RadarLidarStage2
from .voxel_target import decode_centroids, make_targets


def _clouds(root: Path, frame_id: str, batch: dict, output, target, threshold: float,
            ablation: str) -> None:
    folder = root / f"{frame_id}_{ablation}_tau{threshold:.2f}"
    folder.mkdir(parents=True, exist_ok=True)
    radar = batch["radar"][0, batch["radar_valid"][0], :3].detach().cpu().numpy()
    clean = batch["clean_lidar"][0, batch["clean_lidar_valid"][0], :3].detach().cpu().numpy()
    candidate = output.domain.centers_xyz.detach().cpu().numpy()
    confidence = output.confidence.detach().cpu().numpy()
    oracle = decode_centroids(output.domain, target.offsets_normalized)[target.occupied].detach().cpu().numpy()
    predicted = output.reconstructed_points_xyz.detach().cpu().numpy()
    predicted_confidence = output.reconstructed_confidence.detach().cpu().numpy()
    comparison = clean_points_in_domain(output, batch["clean_lidar"], batch["clean_lidar_valid"])
    dp = cKDTree(comparison).query(predicted)[0] if len(predicted) and len(comparison) else np.full(len(predicted), np.inf)
    dc = cKDTree(predicted).query(comparison)[0] if len(comparison) and len(predicted) else np.full(len(comparison), np.inf)
    for name, xyz, scalar in (("radar",radar,None),("clean",clean,None),
                              ("candidates",candidate,confidence),("oracle_occupied",oracle,None),
                              ("predicted",predicted,predicted_confidence),
                              ("false_predicted_0.2m",predicted[dp>.2],predicted_confidence[dp>.2]),
                              ("missed_clean_0.2m",comparison[dc>.2],None)):
        write_ply(folder / f"{name}.ply", xyz, scalar)
    precision = float(np.mean(dp <= .2)) if len(dp) else 0.0
    recall = float(np.mean(dc <= .2)) if len(dc) else 0.0
    metadata = {"frame_id": frame_id, "ablation": ablation, "candidate_threshold": threshold,
                "occupancy_threshold": output.metadata.get("occupancy_threshold"),
                "candidate_sites": len(candidate), "predicted_points": len(predicted),
                "precision_0.2m": precision, "recall_0.2m": recall,
                "f1_0.2m": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                "candidate_point_coverage": target.clean_point_coverage}
    (folder / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="Stage-II best_geom.ckpt or last.ckpt")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage1-checkpoint", help="Override Stage-I path stored in Stage-II checkpoint")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidate-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--ablations", nargs="+", default=("real",),
                        choices=("real","zero","shuffle","wrong_sample","no_confidence","s1_only","s4_only"))
    parser.add_argument("--visualize-frames", nargs="*", default=())
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_values = dict(saved["config"])
    for key in ("channels", "query_radii_m", "expansion_zyx"):
        cfg_values[key] = tuple(cfg_values[key])
    config = Stage2Config(**cfg_values)
    stage1_path = args.stage1_checkpoint or saved["stage1_checkpoint"]
    stage1, saved_stage1 = load_encoder(Path(stage1_path), args.device)
    trained_variant = saved_stage1.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != saved["radar_variant"]:
        raise ValueError("Stage-II and Stage-I radar variants differ")
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)
    model = RadarLidarStage2(stage1.config.channels, config).to(args.device).eval()
    model.load_state_dict(saved["model"])
    dataset = VoDStage1Dataset(args.vod_root, args.split, radar_variant=saved["radar_variant"])
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    if "wrong_sample" in args.ablations and len(dataset) < 2:
        parser.error("wrong_sample ablation requires at least two validation frames")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_stage1)
    thresholds = args.candidate_thresholds or [config.confidence_threshold]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for threshold in thresholds:
        for ablation in args.ablations:
            metrics = Stage2MetricAccumulator()
            with torch.no_grad():
                for i, raw in enumerate(tqdm(loader, desc=f"tau={threshold:.2f} {ablation}", dynamic_ncols=True, leave=False)):
                    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k,v in raw.items()}
                    evidence = stage1(batch["radar"], batch["radar_valid"], defer_diagnostics=True)
                    replacement = None
                    if ablation == "wrong_sample":
                        other = collate_stage1([dataset[(i+1) % len(dataset)]])
                        other = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k,v in other.items()}
                        replacement = stage1(other["radar"], other["radar_valid"], defer_diagnostics=True)
                    output = model(evidence, stage1.config.grid, ablation=ablation,
                                   replacement_stage1=replacement,
                                   confidence_threshold=threshold)
                    target = make_targets(output.domain, batch["clean_lidar"], batch["clean_lidar_valid"],
                                          free_ray_tolerance_m=config.free_ray_tolerance_m)
                    metrics.add(output, target, batch["clean_lidar"], batch["clean_lidar_valid"],
                                occupancy_threshold=config.occupancy_threshold)
                    if batch["frame_id"][0] in args.visualize_frames:
                        _clouds(root / "clouds", batch["frame_id"][0], batch, output, target,
                                threshold, ablation)
            summary = metrics.summary()
            summary.update({"candidate_threshold": threshold, "ablation": ablation,
                            "occupancy_threshold": config.occupancy_threshold,
                            "stage1_checkpoint": str(stage1_path), "stage2_checkpoint": str(args.checkpoint),
                            "diffusion_enabled": False})
            results.append(summary)
            print(f"tau {threshold:.2f} {ablation}: candidate={summary['candidate_sites']} "
                  f"coverage={summary['candidate_point_coverage']:.3f} predicted={summary['predicted_sites']} "
                  f"P/R/F1@.2m={summary['geometry']['0.2m']['precision']:.3f}/"
                  f"{summary['geometry']['0.2m']['recall']:.3f}/"
                  f"{summary['geometry']['0.2m']['f1']:.3f}", flush=True)
    (root / "stage2_evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
