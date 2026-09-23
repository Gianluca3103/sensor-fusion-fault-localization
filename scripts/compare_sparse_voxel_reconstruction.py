"""Compare faulty, generated, and clean 3D voxels for validation components.

This uses the checkpoint's DDIM sampler.  It does not feed the clean target to
the model; the target is loaded only for scoring and the comparison figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.two_stage_reconstruction_head.diffusion_process import (
    SparseVoxelDiffusionBaseline,
    SparseVoxelDiffusionConfig,
    SparseVoxelExample,
    collate_sparse_voxel_examples,
    voxel_set_metrics_at_distance,
)
from scripts.cache_sparse_voxel_supervision import CACHE_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-voxels", type=int, default=25_000)
    parser.add_argument("--min-missing", type=int, default=10)
    parser.add_argument("--sampling-steps", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.count < 1 or args.max_voxels < 1 or args.min_missing < 1:
        parser.error("count, max-voxels and min-missing must be positive")
    if args.sampling_steps < 1 or not 0 < args.threshold < 1:
        parser.error("sampling-steps must be positive and threshold must be in (0, 1)")
    return args


def _select_components(
    cache_root: Path, split: str, count: int, max_voxels: int, min_missing: int
) -> list[tuple[Path, int, int, int]]:
    candidates = []
    for path in sorted((cache_root / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            version = int(archive["cache_version"].item()) if "cache_version" in archive else None
            if version != CACHE_VERSION:
                raise ValueError(f"{path} is cache version {version}; rebuild version {CACHE_VERSION}")
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
            target = np.asarray(archive["target_occupancy"][:, 0]) > 0.5
            faulty = np.asarray(archive["faulty_occupancy"][:, 0]) > 0.5
            editable = np.asarray(archive["editable_mask"][:, 0]) > 0.5
            for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
                size = int(stop - start)
                if size > max_voxels:
                    continue
                missing = int((target[start:stop] & ~faulty[start:stop] & editable[start:stop]).sum())
                if missing >= min_missing:
                    candidates.append((path, index, size, missing))
    # Pick visible repair cases from different frames.  This is an inspection
    # set, not a representative sample or an aggregate validation estimate.
    selected = []
    used_files = set()
    for item in sorted(candidates, key=lambda item: (-item[3], item[2], item[0].name)):
        if item[0] in used_files:
            continue
        selected.append(item)
        used_files.add(item[0])
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("No cache components meet the requested size and missing-voxel limits")
    return selected


def _load_example(path: Path, index: int) -> SparseVoxelExample:
    with np.load(path, allow_pickle=False) as archive:
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return SparseVoxelExample(
            coords_zyx=torch.from_numpy(np.asarray(archive["coords_zyx"][start:stop], dtype=np.int64)),
            coords_xyz_m=torch.from_numpy(np.asarray(archive["coords_xyz_m"][start:stop], dtype=np.float32)),
            condition_features=torch.from_numpy(np.asarray(archive["condition_features"][start:stop], dtype=np.float32)),
            target_occupancy=torch.from_numpy(np.asarray(archive["target_occupancy"][start:stop], dtype=np.float32)),
            faulty_occupancy=torch.from_numpy(np.asarray(archive["faulty_occupancy"][start:stop], dtype=np.float32)),
            editable_mask=torch.from_numpy(np.asarray(archive["editable_mask"][start:stop], dtype=np.float32)),
        )


def _metric_dict(probability: torch.Tensor, batch) -> dict[str, float]:
    return {
        name: float(value.detach().cpu())
        for name, value in voxel_set_metrics_at_distance(probability, batch).items()
    }


def _draw_comparison(path: Path, xyz: np.ndarray, editable: np.ndarray, states: list[tuple[str, np.ndarray, str]], title: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, (name, occupied, color) in enumerate(states):
        context = np.flatnonzero(occupied & ~editable)
        if len(context) > 5_000:
            context = context[np.linspace(0, len(context) - 1, 5_000, dtype=np.int64)]
        changed = np.flatnonzero(occupied & editable)
        for row, (horizontal, vertical, xlabel, ylabel) in enumerate(
            ((0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)"))
        ):
            axis = axes[row, column]
            axis.scatter(xyz[context, horizontal], xyz[context, vertical], s=1, c="0.7", alpha=0.35, rasterized=True)
            axis.scatter(xyz[changed, horizontal], xyz[changed, vertical], s=9, c=color, alpha=0.85, rasterized=True)
            axis.set_xlim(float(xyz[:, horizontal].min()) - 0.5, float(xyz[:, horizontal].max()) + 0.5)
            axis.set_ylim(float(xyz[:, vertical].min()) - 0.5, float(xyz[:, vertical].max()) + 0.5)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.15)
            if row == 0:
                axis.set_title(f"{name}\n{len(changed)} occupied edit voxels")
    figure.suptitle(title + "  |  grey = unchanged context; colour = editable region")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = SparseVoxelDiffusionConfig(**checkpoint["model_config"])
    if config.condition_feature_dim != SparseVoxelDiffusionConfig().condition_feature_dim:
        raise ValueError("Old target-leaking checkpoint; rebuild cache version 2 and retrain")
    model = SparseVoxelDiffusionBaseline(config).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    selected = _select_components(
        args.cache_root, args.split, args.count, args.max_voxels, args.min_missing
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, (cache_path, component_index, size, missing) in enumerate(selected, 1):
        batch = collate_sparse_voxel_examples([_load_example(cache_path, component_index)]).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        with torch.inference_mode():
            sampled = model.sample(
                batch, sampling_steps=args.sampling_steps,
                occupancy_threshold=args.threshold, generator=generator,
            )
            faulty_metrics = _metric_dict(batch.faulty_occupancy, batch)
            generated_metrics = _metric_dict(sampled["occupancy_probability"], batch)
        xyz = batch.coords_xyz_m[0].cpu().numpy()
        faulty = (batch.faulty_occupancy[0, :, 0] > 0.5).cpu().numpy()
        generated = sampled["occupied_mask"][0, :, 0].cpu().numpy().astype(bool)
        target = (batch.target_occupancy[0, :, 0] > 0.5).cpu().numpy()
        editable = (batch.editable_mask[0, :, 0] > 0.5).cpu().numpy()
        trivial_fill = faulty | editable
        added = generated & ~faulty & editable
        restored = added & target
        filename = f"{rank:02d}_{cache_path.stem}_component_{component_index:03d}"
        figure_path = args.output_root / f"{filename}.png"
        _draw_comparison(
            figure_path, xyz, editable,
            [("Faulty LiDAR", faulty, "#c83f3f"),
             ("Generated", generated, "#1986c9"),
             ("Clean target", target, "#329b5b")],
            f"{cache_path.stem} / component {component_index} / DDIM {args.sampling_steps} steps",
        )
        np.savez_compressed(
            args.output_root / f"{filename}.npz",
            coords_xyz_m=xyz, faulty_occupied=faulty, generated_occupied=generated,
            clean_target_occupied=target, editable_mask=editable,
        )
        record = {
            "cache_file": str(cache_path), "component_index": component_index,
            "candidate_voxels": size, "missing_clean_voxels": missing,
            "faulty_edit_occupied": int((faulty & editable).sum()),
            "generated_edit_occupied": int((generated & editable).sum()),
            "clean_edit_occupied": int((target & editable).sum()),
            "editable_empty_voxels": int((editable & ~target).sum()),
            "trivial_selector_fill_equals_target": bool(np.array_equal(trivial_fill, target)),
            "generated_equals_trivial_selector_fill": bool(np.array_equal(generated, trivial_fill)),
            "new_generated_voxels": int(added.sum()),
            "correctly_restored_voxels": int(restored.sum()),
            "incorrect_new_voxels": int((added & ~target).sum()),
            "faulty_metrics": faulty_metrics,
            "generated_metrics": generated_metrics,
            "figure": str(figure_path),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del batch, sampled
    (args.output_root / "summary.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"],
                    "sampling_steps": args.sampling_steps, "examples": records}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
