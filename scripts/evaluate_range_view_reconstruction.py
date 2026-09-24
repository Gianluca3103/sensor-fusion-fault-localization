"""Evaluate generated XYZ scans for append-only and conservative-delete ablations."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from models.two_stage_reconstruction_head.range_view.evaluation import evaluate_range_model
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.range_view.merge import MergeConfig
from models.two_stage_reconstruction_head.range_view.model import RangeModelConfig, RangeViewReconstructor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fault-map-root", type=Path)
    parser.add_argument("--disable-radar", action="store_true")
    parser.add_argument("--disable-fault-map", action="store_true")
    parser.add_argument("--add-threshold", type=float, default=0.5)
    parser.add_argument("--delete-thresholds", type=float, nargs="+", default=(0.95, 0.99, 0.995, 0.999))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("representation") != "range_view":
        raise ValueError("checkpoint is not a range-view reconstruction model")
    geometry = RangeGeometry(**checkpoint["geometry"])
    forward_only = bool(checkpoint["merge_config"].get("forward_only", False))
    config = RangeModelConfig(**checkpoint["model_config"])
    if args.disable_radar or args.disable_fault_map:
        config = replace(config, use_radar=config.use_radar and not args.disable_radar,
                         use_fault_map_conditioning=config.use_fault_map_conditioning and not args.disable_fault_map)
    if config.use_fault_map_conditioning and args.fault_map_root is None:
        parser.error("checkpoint uses predicted fault-map conditioning; supply --fault-map-root or --disable-fault-map")
    device = torch.device(args.device)
    model = RangeViewReconstructor(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    paths = sorted((args.data_root / args.split).glob("*.npz"))
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(f"No artifacts in {args.data_root / args.split}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for threshold in args.delete_thresholds:
        for allow_delete in ((False, True) if threshold == args.delete_thresholds[0] else (True,)):
            name = f"{'conservative' if allow_delete else 'append_only'}_delete_{threshold:g}"
            summary = evaluate_range_model(
                model, paths, args.radar_root, geometry, device=device,
                merge_config=MergeConfig(allow_original_deletion=allow_delete,
                                         delete_threshold=threshold,
                                         add_threshold=args.add_threshold,
                                         forward_only=forward_only),
                fault_map_root=(args.fault_map_root if config.use_fault_map_conditioning else None),
                output_path=args.output_root / f"{name}.json",
                visualization_root=args.output_root / name / "visualizations",
                visualization_limit=3 if allow_delete else 0,
            )
            results[name] = {"overall": summary["overall_macro"], "by_fault": summary["by_fault_macro"]}
            print(json.dumps({"ablation": name, **results[name]}), flush=True)
    (args.output_root / "ablation_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
