"""Time the real Stage-II candidate and neighbor construction on VoD frames.

Run after pausing training; this executes the frozen Stage-I encoder on the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

from models.radar_lidar_stage2.candidate_domain import make_candidates
from models.radar_lidar_stage2.stage1_conditioning import Stage1Conditioning
from models.radar_lidar_stage2_residual.coverage import faulty_coverage
from models.radar_lidar_stage2_residual.data import PairedFaultDataset, collate_paired
from models.radar_lidar_stage2_residual.targets import make_residual_targets
from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from scripts.visualize_stage1_confidence_cloud import load_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--fault-samples-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--config", default="configs/radar_lidar_stage2_diffusion.json")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--fault-pattern", default="*")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--start-index", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kd-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, args.kd_workers) < 1 or args.start_index < 0:
        parser.error("Batch size, repeats, workers and start index must be valid")
    os.environ["STAGE2_KDTREE_WORKERS"] = str(args.kd_workers)
    config = DiffusionConfig.from_json(args.config)
    data = PairedFaultDataset(args.vod_root, args.fault_samples_root, args.split,
                              radar_variant=args.radar_variant, fault_pattern=args.fault_pattern)
    if args.start_index + args.batch_size > len(data):
        parser.error("Requested batch extends beyond the dataset")
    ticks: dict[str, list[float]] = {}

    def sync() -> None:
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize()

    def timed(name, operation):
        sync()
        start = time.perf_counter()
        output = operation()
        sync()
        ticks.setdefault(name, []).append(time.perf_counter() - start)
        return output

    raw = timed("load_and_collate", lambda: collate_paired(
        [data[i] for i in range(args.start_index, args.start_index + args.batch_size)]))
    batch = timed("transfer_to_device", lambda: {
        key: value.to(args.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()})
    stage1, saved = load_encoder(Path(args.stage1_checkpoint), args.device)
    stage1.eval()
    if not (stage1.config.surface_region_enabled or stage1.config.bottleneck_support_only):
        raise ValueError("Expected the learned-region Stage-I checkpoint")
    trained_variant = saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != args.radar_variant:
        raise ValueError("Radar variant differs from the Stage-I checkpoint")
    conditioner = Stage1Conditioning(stage1.config.conditioning_channels, config.residual).to(args.device).eval()
    records = []
    with torch.no_grad():
        for _ in range(args.repeats):
            evidence = timed("stage1_forward", lambda: stage1(
                batch["radar"], batch["radar_valid"], defer_diagnostics=True))
            domain = timed("region_rasterization", lambda: make_candidates(
                evidence, stage1.config.grid,
                confidence_threshold=config.residual.confidence_threshold,
                expansion_zyx=config.residual.expansion_zyx,
                max_sites=config.residual.max_candidate_sites))
            coverage = timed("faulty_coverage", lambda: faulty_coverage(
                domain, batch["faulty_lidar"], batch["faulty_lidar_valid"],
                batch["fault_region"], ray_tolerance_m=config.residual.ray_tolerance_m))
            timed("stage1_neighbors_and_projection", lambda: conditioner(evidence, domain))
            target = timed("clean_target_and_free_ray_search", lambda: make_residual_targets(
                domain, batch["clean_lidar"], batch["clean_lidar_valid"], coverage,
                free_ray_tolerance_m=config.residual.free_ray_tolerance_m))
            sites = len(domain.coordinates)
            records.append({"candidate_sites": sites,
                            "accepted_regions": domain.counts.get("accepted_region_proposals",
                                                                  domain.counts.get("selected_support_sites", 0)),
                            "unknown_fraction": float((~target.supervised).float().mean()) if sites else 0.,
                            "may_add_fraction": float(coverage.may_add.float().mean()) if sites else 0.})
    print(json.dumps({"frames": batch["frame_id"], "device": args.device,
                      "kd_workers": args.kd_workers, "repeats": args.repeats,
                      "seconds_median": {key: statistics.median(values) for key, values in ticks.items()},
                      "seconds_each": ticks, "iterations": records}, indent=2))


if __name__ == "__main__":
    main()
