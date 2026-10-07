"""Audit Stage-II candidate coverage and voxel-centroid representation ceiling.

No reconstruction network, faulty LiDAR, diffusion, or semantic labels are used.
The clean scan influences targets and diagnostics only, never candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset
from models.radar_lidar_stage2 import decode_centroids, make_candidates, make_targets
from scripts.visualize_stage1_confidence_cloud import load_encoder, write_ply


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--radar-variant")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--expand-z", type=int, default=1)
    parser.add_argument("--expand-y", type=int, default=1)
    parser.add_argument("--expand-x", type=int, default=1)
    parser.add_argument("--max-candidate-sites", type=int,
                        help="Optional legacy point-proposal limit; learned regions are always uncapped")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    encoder, saved = load_encoder(args.checkpoint, args.device)
    variant = args.radar_variant or saved.get("data", {}).get(
        "radar_variant", "radar_20frames_verified_doppler_radial")
    dataset = VoDStage1Dataset(args.vod_root, args.split, frame_ids=args.frame_id,
                               radar_variant=variant)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sample in dataset:
        radar = sample["radar"].unsqueeze(0).to(args.device)
        lidar = sample["clean_lidar"].unsqueeze(0).to(args.device)
        radar_valid = torch.ones(radar.shape[:2], dtype=torch.bool, device=args.device)
        lidar_valid = torch.ones(lidar.shape[:2], dtype=torch.bool, device=args.device)
        output = encoder(radar, radar_valid)
        domain = make_candidates(output, encoder.config.grid,
                                 confidence_threshold=args.confidence_threshold,
                                 expansion_zyx=(args.expand_z, args.expand_y, args.expand_x),
                                 max_sites=args.max_candidate_sites)
        targets = make_targets(domain, lidar, lidar_valid)
        decoded = decode_centroids(domain, targets.offsets_normalized)
        if bool(targets.occupied.any()):
            error = torch.linalg.vector_norm(
                decoded[targets.occupied] - targets.clean_centroid_xyz[targets.occupied], dim=-1)
            if not bool((error < 1e-4).all()):
                raise AssertionError("Centroid encode/decode round trip failed")
        distances = targets.point_centroid_errors_m.cpu().numpy()
        prefix = args.output_root / sample["frame_id"]
        write_ply(prefix.with_name(prefix.name + "_radar.ply"), radar[0, :, :3].cpu().numpy())
        write_ply(prefix.with_name(prefix.name + "_clean_lidar.ply"), lidar[0, :, :3].cpu().numpy())
        write_ply(prefix.with_name(prefix.name + "_candidate.ply"),
                  domain.centers_xyz.cpu().numpy(), domain.confidence.cpu().numpy())
        write_ply(prefix.with_name(prefix.name + "_oracle_centroid.ply"),
                  decoded[targets.occupied].cpu().numpy())
        summary = {"frame_id": sample["frame_id"], "split": args.split,
                   "checkpoint": str(args.checkpoint.resolve()), "epoch": saved.get("epoch"),
                   "diffusion_enabled": False, "radar_variant": variant,
                   "confidence_threshold": args.confidence_threshold,
                   "expansion_zyx": [args.expand_z, args.expand_y, args.expand_x],
                   "counts": domain.counts,
                   "occupied_candidate_voxels": int(targets.occupied.sum()),
                   "candidate_occupancy_ratio": targets.candidate_occupancy_ratio,
                   "clean_points_in_grid": targets.clean_points_in_grid,
                   "clean_points_in_candidates": targets.clean_points_in_candidates,
                   "candidate_clean_point_coverage": targets.clean_point_coverage,
                   "same_voxel_centroid_error_mean_m": float(distances.mean()) if len(distances) else None,
                   "same_voxel_centroid_error_p95_m": float(np.quantile(distances, .95)) if len(distances) else None,
                   "represented_clean_point_within_m": {
                       str(t): float(np.mean(distances <= t)) if len(distances) else None
                       for t in (.1, .2, .5)},
                   "note": "These are representation ceilings, not learned reconstruction scores. "
                           "Unoccupied candidate voxels may be unobserved rather than free."}
        prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output_root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
