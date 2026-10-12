"""Cache frozen Stage-I geometry, faulty coverage, neighbors, and labels.

Run once for a fixed radar stack, fault cache, Stage-I checkpoint, and Stage-II
geometry configuration. The cache is safe to resume after interruption.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from tqdm import tqdm

from models.radar_lidar_stage2_diffusion.cache import (
    prepare_frame, provenance, save_frame,
)
from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from models.radar_lidar_stage2_residual.data import PairedFaultDataset, collate_paired
from scripts.visualize_stage1_confidence_cloud import load_encoder


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def _cached(path: Path, frame_id: str) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        if str(data["frame_id"].item()) != frame_id or "neighbor_index_3" not in data:
            raise ValueError(f"Incomplete or mismatched cache entry: {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True)
    parser.add_argument("--fault-samples-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", default="configs/radar_lidar_stage2_diffusion.json")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--fault-pattern", default="*")
    parser.add_argument("--compression", choices=("fast", "none"), default="fast")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = DiffusionConfig.from_json(args.config)
    stage1, saved = load_encoder(Path(args.stage1_checkpoint), args.device)
    stage1.eval()
    if not stage1.config.surface_region_enabled:
        raise ValueError("The preprocessing cache requires learned Stage-I regions")
    trained_variant = saved.get("data", {}).get("radar_variant")
    if trained_variant and trained_variant != args.radar_variant:
        raise ValueError("Radar variant differs from the Stage-I checkpoint")

    datasets = {
        split: PairedFaultDataset(args.vod_root, args.fault_samples_root, split,
                                  radar_variant=args.radar_variant,
                                  fault_pattern=args.fault_pattern)
        for split in ("train", "val")
    }
    split_ids = {split: [frame.frame_id for frame in dataset.base.frames]
                 for split, dataset in datasets.items()}
    identity = provenance(stage1_checkpoint=args.stage1_checkpoint,
                          vod_root=args.vod_root,
                          fault_samples_root=args.fault_samples_root,
                          radar_variant=args.radar_variant,
                          fault_pattern=args.fault_pattern,
                          config=config, compression=args.compression)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = {**identity, "grid": json.loads(json.dumps(asdict(stage1.config.grid))),
                "splits": split_ids, "complete": False}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in manifest.items():
            if key != "complete" and previous.get(key) != value:
                raise ValueError(f"Existing preprocessing cache differs on {key}; use a new root")
    _write_manifest(manifest_path, manifest)

    total = sum(len(dataset) for dataset in datasets.values())
    written_bytes, built = 0, 0
    progress = tqdm(total=total, desc="Stage-II preprocessing", unit="frame", dynamic_ncols=True)
    with torch.no_grad():
        for split, dataset in datasets.items():
            for index, frame_id in enumerate(split_ids[split]):
                path = root / split / f"{frame_id}.npz"
                if _cached(path, frame_id):
                    progress.update(1)
                    continue
                raw = collate_paired([dataset[index]])
                batch = {key: value.to(args.device) if isinstance(value, torch.Tensor)
                         else value for key, value in raw.items()}
                evidence = stage1(batch["radar"], batch["radar_valid"],
                                  defer_diagnostics=True)
                arrays = prepare_frame(evidence, stage1.config.grid, batch, config)
                save_frame(path, arrays, args.compression)
                written_bytes += path.stat().st_size
                built += 1
                progress.update(1)
                progress.set_postfix(sites=len(arrays["coordinates"]),
                                     mib=round(path.stat().st_size / 2**20, 1))
                if built == 16:
                    remaining = total - progress.n
                    estimated_remaining = written_bytes / built * remaining * 1.2
                    available = shutil.disk_usage(root).free
                    print(f"Cache pilot: {written_bytes / built / 2**20:.1f} MiB/frame; "
                          f"estimated remaining {estimated_remaining / 2**30:.1f} GiB; "
                          f"free {available / 2**30:.1f} GiB", flush=True)
                    if estimated_remaining > available:
                        raise OSError("Insufficient free space for the estimated full cache. "
                                      "Partial files remain reusable; choose a larger output volume.")
    progress.close()
    manifest["complete"] = True
    _write_manifest(manifest_path, manifest)
    print(f"Cache complete: train={len(split_ids['train'])} val={len(split_ids['val'])} "
          f"new_frames={built} root={root}", flush=True)


if __name__ == "__main__":
    main()
