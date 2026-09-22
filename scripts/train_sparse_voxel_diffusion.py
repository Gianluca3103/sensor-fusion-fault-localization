"""Train the selector-local sparse 3D diffusion baseline on reconstruction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Iterator

import numpy as np
import torch

from models.two_stage_reconstruction_head.diffusion_process import (
    SparseVoxelDiffusionBaseline,
    SparseVoxelDiffusionConfig,
    SparseVoxelExample,
    build_sparse_voxel_example,
    collate_sparse_voxel_examples,
)
from voxelization.inputs import LIDAR_FIELDS, load_clean_lidar_from_metadata
from voxelization import (
    HardVoxelizer,
    OracleFaultSelector3DConfig,
    build_voxel_fault_targets,
    load_voxelization_config,
    select_oracle_fault_regions_3d,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument(
        "--vod-public-root",
        type=Path,
        help=(
            "Deprecated compatibility option. Clean LiDAR is resolved from "
            "each artifact's metadata, so this is not used."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--supervision-cache-root", type=Path)
    parser.add_argument(
        "--voxel-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "voxelization_3d.json",
    )
    parser.add_argument("--train-fraction", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--lambda-diffusion", type=float, default=1.0)
    parser.add_argument("--lambda-bce", type=float, default=1.0)
    parser.add_argument("--lambda-chamfer", type=float, default=0.5)
    parser.add_argument("--chamfer-chunk-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 < args.train_fraction <= 1 or not 0 < args.val_fraction <= 1:
        parser.error("dataset fractions must be in (0, 1]")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or any(
        weight < 0 for weight in (args.lambda_diffusion, args.lambda_bce, args.lambda_chamfer)
    ):
        parser.error("optimization parameters must be non-negative and learning rate positive")
    if args.chamfer_chunk_size < 1 or args.log_every < 1:
        parser.error("chamfer-chunk-size and log-every must be positive")
    return args


def _subset(paths: list[Path], fraction: float, seed: int) -> list[Path]:
    amount = max(1, round(len(paths) * fraction))
    generator = random.Random(seed)
    return sorted(generator.sample(paths, amount))


def _radar_names(width: int) -> tuple[str, ...]:
    if width < 3:
        raise ValueError("radar points require at least XYZ columns")
    return ("x", "y", "z") + tuple(f"radar_feature_{index}" for index in range(width - 3))


def _load_examples(
    sample_path: Path,
    *,
    radar_root: Path,
    lidar_voxelizer: HardVoxelizer,
    radar_voxelizer: HardVoxelizer,
    selector_config: OracleFaultSelector3DConfig,
    supervision_cache_root: Path | None = None,
):
    if supervision_cache_root is not None:
        cache_path = supervision_cache_root / sample_path.parent.name / f"{sample_path.stem}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Sparse supervision cache is missing: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as archive:
            offsets = np.asarray(archive["offsets"], dtype=np.int64)
            coords = np.asarray(archive["coords_zyx"], dtype=np.int64)
            xyz = np.asarray(archive["coords_xyz_m"], dtype=np.float32)
            condition = np.asarray(archive["condition_features"], dtype=np.float32)
            target = np.asarray(archive["target_occupancy"], dtype=np.float32)
            faulty = np.asarray(archive["faulty_occupancy"], dtype=np.float32)
            editable = np.asarray(archive["editable_mask"], dtype=np.float32)
        return tuple(
            SparseVoxelExample(
                coords_zyx=torch.from_numpy(coords[offsets[index]:offsets[index + 1]]),
                coords_xyz_m=torch.from_numpy(xyz[offsets[index]:offsets[index + 1]]),
                condition_features=torch.from_numpy(condition[offsets[index]:offsets[index + 1]]),
                target_occupancy=torch.from_numpy(target[offsets[index]:offsets[index + 1]]),
                faulty_occupancy=torch.from_numpy(faulty[offsets[index]:offsets[index + 1]]),
                editable_mask=torch.from_numpy(editable[offsets[index]:offsets[index + 1]]),
            )
            for index in range(len(offsets) - 1)
        )
    with np.load(sample_path, allow_pickle=False) as archive:
        faulty = np.asarray(archive["faulty_lidar_points"], dtype=np.float32)
        source_ids = np.asarray(archive["faulty_source_ids"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    frame_id = str(metadata["frame_id"])
    radar_path = radar_root / f"{int(frame_id):05d}.npz"
    if not radar_path.is_file():
        raise FileNotFoundError(f"Missing aligned radar input for {sample_path.name}: {radar_path}")
    clean = load_clean_lidar_from_metadata(metadata)
    with np.load(radar_path, allow_pickle=False) as archive:
        radar = np.asarray(archive["radar_points"], dtype=np.float32)
    targets = build_voxel_fault_targets(clean, faulty, source_ids, lidar_voxelizer.grid)
    selection = select_oracle_fault_regions_3d(
        targets.repair_mask, targets.remove_mask, lidar_voxelizer.grid, selector_config
    )
    if not selection.components:
        return ()
    faulty_voxels = lidar_voxelizer.voxelize(faulty, LIDAR_FIELDS)
    radar_voxels = radar_voxelizer.voxelize(radar, _radar_names(radar.shape[1]))
    return tuple(
        build_sparse_voxel_example(
            faulty_lidar=faulty_voxels,
            radar=radar_voxels,
            targets=targets,
            selection=selection,
            component=component,
            grid=lidar_voxelizer.grid,
        )
        for component in selection.components
    )


def _batches(
    paths: list[Path],
    *,
    batch_size: int,
    **loader_kwargs,
) -> Iterator:
    pending = []
    for sample_path in paths:
        pending.extend(_load_examples(sample_path, **loader_kwargs))
        while len(pending) >= batch_size:
            yield collate_sparse_voxel_examples(pending[:batch_size])
            del pending[:batch_size]
    if pending:
        yield collate_sparse_voxel_examples(pending)


def _run_epoch(model, optimizer, paths, *, device, batch_size, loader_kwargs, log_every, label):
    train = optimizer is not None
    model.train(train)
    aggregates = {"loss": 0.0, "diffusion_loss": 0.0, "bce_loss": 0.0, "chamfer_loss": 0.0}
    batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for cpu_batch in _batches(paths, batch_size=batch_size, **loader_kwargs):
            batch = cpu_batch.to(device)
            output = model(batch, compute_metrics=not train)
            if train:
                optimizer.zero_grad(set_to_none=True)
                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for key in aggregates:
                aggregates[key] += float(output[key].detach().cpu())
            if not train:
                for key in ("iou_at_0_2m", "f1_at_0_2m"):
                    aggregates.setdefault(key, 0.0)
                    aggregates[key] += float(output[key].detach().cpu())
            batches += 1
            if batches % log_every == 0:
                print(json.dumps({
                    "stage": label,
                    "batches": batches,
                    "mean_loss": aggregates["loss"] / batches,
                    "mean_chamfer_loss": aggregates["chamfer_loss"] / batches,
                }), flush=True)
    if not batches:
        raise RuntimeError("No selector components were available for this split")
    return {key: value / batches for key, value in aggregates.items()} | {"batches": batches}


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    voxelization = load_voxelization_config(args.voxel_config)
    lidar_voxelizer = HardVoxelizer(voxelization.grid, max_points_per_voxel=voxelization.lidar.max_points_per_voxel)
    radar_voxelizer = HardVoxelizer(voxelization.grid, max_points_per_voxel=voxelization.radar.max_points_per_voxel)
    selector_config = OracleFaultSelector3DConfig()
    train_paths = _subset(sorted((args.samples_root / "train").glob("*.npz")), args.train_fraction, args.seed)
    val_paths = _subset(sorted((args.samples_root / "val").glob("*.npz")), args.val_fraction, args.seed + 1)
    if not train_paths or not val_paths:
        raise FileNotFoundError("Both train and val reconstruction splits must contain samples")
    config = SparseVoxelDiffusionConfig(
        grid_dimensions_zyx=voxelization.grid.dimensions_zyx,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        lambda_chamfer=args.lambda_chamfer,
        lambda_diffusion=args.lambda_diffusion,
        lambda_bce=args.lambda_bce,
        chamfer_chunk_size=args.chamfer_chunk_size,
    )
    model = SparseVoxelDiffusionBaseline(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.json").write_text(json.dumps({
        "arguments": vars(args), "model_config": config.__dict__,
        "train_samples": len(train_paths), "val_samples": len(val_paths),
    }, indent=2, default=str), encoding="utf-8")
    loader_kwargs = {
        "radar_root": args.radar_root / "train",
        "lidar_voxelizer": lidar_voxelizer,
        "radar_voxelizer": radar_voxelizer,
        "selector_config": selector_config,
        "supervision_cache_root": args.supervision_cache_root,
    }
    # Radar caches are split-specific; use the same loader contract for each.
    history_path = output_root / "history.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            train_metrics = _run_epoch(
                model, optimizer, train_paths, device=device,
                batch_size=args.batch_size, loader_kwargs=loader_kwargs,
                log_every=args.log_every, label=f"epoch_{epoch}_train",
            )
            val_kwargs = dict(loader_kwargs)
            val_kwargs["radar_root"] = args.radar_root / "val"
            val_metrics = _run_epoch(
                model, None, val_paths, device=device,
                batch_size=args.batch_size, loader_kwargs=val_kwargs,
                log_every=args.log_every, label=f"epoch_{epoch}_val",
            )
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.write(json.dumps(record) + "\n")
            history.flush()
            torch.save({"epoch": epoch, "model_config": config.__dict__, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "metrics": record}, output_root / "last_checkpoint.pt")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
