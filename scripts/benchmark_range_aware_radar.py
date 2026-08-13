"""Benchmark the learned range-aware Radar aggregation in isolation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.reconstruction_head.coarse_reconstruction.range_aware_radar import (
    RangeAwareRadarAggregation,
    RangeAwareRadarConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--occupancy-fraction", type=float, default=0.10)
    parser.add_argument("--active-fraction", type=float, default=0.50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--spatial-chunk-size", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.height < 1 or args.width < 1:
        raise ValueError("batch size and spatial dimensions must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations positive")
    for name, value in (
        ("occupancy_fraction", args.occupancy_fraction),
        ("active_fraction", args.active_fraction),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")

    device = torch.device(args.device)
    config = RangeAwareRadarConfig(
        enabled=True,
        spatial_chunk_size=args.spatial_chunk_size,
    )
    module = RangeAwareRadarAggregation(4, config).to(device).eval()
    radar = torch.rand(
        args.batch_size, 4, args.height, args.width, device=device
    )
    radar[:, 0] = (
        torch.rand_like(radar[:, 0]) < args.occupancy_fraction
    ).to(radar.dtype)
    active = (
        torch.rand(
            args.batch_size, 1, args.height, args.width, device=device
        )
        < args.active_fraction
    ).to(radar.dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            module(radar, active)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            output, debug = module(radar, active)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    report = {
        "device": str(device),
        "input_shape": list(radar.shape),
        "output_shape": list(output.shape),
        "candidate_window": [
            2
            * math.ceil(
                config.max_radius_m
                * args.height
                / (config.x_max_m - config.x_min_m)
            )
            + 1,
            2
            * math.ceil(
                config.max_radius_m
                * args.width
                / (config.y_max_m - config.y_min_m)
            )
            + 1,
        ],
        "iterations": args.iterations,
        "mean_forward_ms": 1000.0 * elapsed / args.iterations,
        "throughput_samples_per_second": (
            args.iterations * args.batch_size / elapsed
        ),
        "mean_valid_neighbors": float(
            debug["valid_neighbor_count"].float().mean().cpu()
        ),
    }
    if device.type == "cuda":
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
