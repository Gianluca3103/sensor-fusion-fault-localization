"""Benchmark sparse global Radar PillarAttention without the HRNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.reconstruction_head import (
    RadarPillarAttention,
    RadarPillarAttentionConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pillars-per-sample", type=int, default=12000)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.pillars_per_sample < 1:
        raise ValueError("batch size and pillar count must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations positive")
    device = torch.device(args.device)
    config = RadarPillarAttentionConfig(
        enabled=True,
        attention_dim=args.attention_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_blocks=args.num_blocks,
    )
    module = RadarPillarAttention(args.feature_dim, config).to(device).eval()
    total_pillars = args.batch_size * args.pillars_per_sample
    features = torch.randn(total_pillars, args.feature_dim, device=device)
    batches = torch.repeat_interleave(
        torch.arange(args.batch_size, device=device),
        args.pillars_per_sample,
    )
    local_indices = torch.arange(
        args.pillars_per_sample, device=device
    ).repeat(args.batch_size)
    coordinates = torch.stack(
        (batches, local_indices // 320, local_indices % 320), dim=1
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            module(features, coordinates, args.batch_size)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.iterations):
            output, debug = module(features, coordinates, args.batch_size)
        _synchronize(device)
        elapsed = time.perf_counter() - start

    report = {
        "device": str(device),
        "radar_points": "not measured by this module-only benchmark",
        "batch_size": args.batch_size,
        "occupied_pillars_per_sample": args.pillars_per_sample,
        "attention_token_shape_per_sample": [
            1,
            args.pillars_per_sample,
            args.attention_dim,
        ],
        "attention_score_shape_per_head": [
            args.pillars_per_sample,
            args.pillars_per_sample,
        ],
        "radar_feature_dimension": args.feature_dim,
        "occupied_bev_percentage": (
            100.0 * args.pillars_per_sample / (320 * 320)
        ),
        "attention_pairs_per_sample": args.pillars_per_sample**2,
        "output_shape": list(output.shape),
        "mean_forward_ms": 1000.0 * elapsed / args.iterations,
        "token_counts": debug["token_counts"].cpu().tolist(),
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
