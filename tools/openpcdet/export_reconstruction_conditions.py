"""Export clean/faulty/coarse/fine VoD point clouds for one frozen detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from Fault_Localization_Model.sample_utils import load_sample_metadata
from Fault_Localization_Model.vod_dataset import load_vod_lidar, resolve_vod_public_root
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head import CoarseReconstructionDataset, coarse_reconstruction_collate
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _load_selector_config,
    _move_batch,
)
from pcdet_integration.reconstructed_points import repair_point_cloud
from pcdet_integration.stage_inference import load_frozen_reconstruction_pipeline
from pcdet_integration.vod_dataset import export_vod_custom_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--coarse-checkpoint", required=True, type=Path)
    parser.add_argument("--fine-checkpoint", required=True, type=Path)
    parser.add_argument("--selector-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _condition_layout(root: Path, condition: str) -> Path:
    destination = root / condition
    (destination / "points").mkdir(parents=True, exist_ok=True)
    return destination


def _copy_metadata(clean_root: Path, condition_root: Path) -> None:
    for name in ("ImageSets", "labels"):
        source = clean_root / name
        target = condition_root / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    public_root = resolve_vod_public_root(args.vod_root)

    clean_root = _condition_layout(args.output_root, "clean")
    export_vod_custom_dataset(
        public_root,
        clean_root,
        splits=(args.split,),
        overwrite=args.overwrite,
        label_root=args.label_root,
    )
    condition_roots = {
        name: _condition_layout(args.output_root, name)
        for name in ("clean", "faulty", "coarse", "fine")
    }
    for condition in ("faulty", "coarse", "fine"):
        _copy_metadata(clean_root, condition_roots[condition])

    pipeline, coarse_checkpoint, fine_checkpoint = load_frozen_reconstruction_pipeline(
        args.coarse_checkpoint, args.fine_checkpoint, device
    )
    selector = _load_selector_config(args.selector_config)
    paths = _split_paths(args.data_root, args.split, None, args.seed)
    dataset = CoarseReconstructionDataset(
        paths,
        args.radar_root,
        data_root=args.data_root,
        selector_config=selector,
        use_pointpillars=pipeline.coarse_model.config.pointpillars_enabled,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=coarse_reconstruction_collate,
    )
    metadata = {str(path): load_sample_metadata(path) for path in paths}
    exported = 0
    with torch.inference_mode():
        for batch in loader:
            raw_faulty = [points.numpy().copy() for points in batch["faulty_lidar_points"]]
            inputs = _move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                coarse_bev, _ = pipeline.coarse_forward(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                )
                sampled = pipeline.sample(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    coarse_lidar_bev=coarse_bev,
                    sampling_steps=args.sampling_steps,
                )
            fine_bev = sampled["final_lidar_bev"]
            masks = inputs["reconstruction_mask"].detach().float().cpu().numpy()
            coarse_numpy = coarse_bev.detach().float().cpu().numpy()
            fine_numpy = fine_bev.detach().float().cpu().numpy()
            for index, sample_path in enumerate(batch["sample_path"]):
                frame_id = str(metadata[str(sample_path)]["frame_id"])
                frame_id = f"{int(frame_id):05d}"
                lidar_source = (
                    public_root / "lidar" / "training" / "velodyne" / f"{frame_id}.bin"
                )
                clean = load_vod_lidar(lidar_source).astype(np.float32, copy=False)
                faulty = raw_faulty[index]
                coarse_points = repair_point_cloud(
                    faulty, coarse_numpy[index], masks[index], dataset.grid_geometry
                )
                fine_points = repair_point_cloud(
                    faulty, fine_numpy[index], masks[index], dataset.grid_geometry
                )
                for condition, points in (
                    ("clean", clean),
                    ("faulty", faulty),
                    ("coarse", coarse_points),
                    ("fine", fine_points),
                ):
                    destination = condition_roots[condition] / "points" / f"{frame_id}.npy"
                    if args.overwrite or not destination.is_file():
                        np.save(destination, points.astype(np.float32), allow_pickle=False)
                exported += 1
                if exported % 100 == 0:
                    print(f"Exported {exported}/{len(dataset)} frames", flush=True)

    manifest = {
        "split": args.split,
        "frames": exported,
        "coarse_checkpoint": str(args.coarse_checkpoint.resolve()),
        "coarse_epoch": int(coarse_checkpoint.get("epoch", -1)),
        "fine_checkpoint": str(args.fine_checkpoint.resolve()),
        "fine_epoch": int(fine_checkpoint.get("epoch", -1)),
        "point_adapter": "pcdet_integration.reconstructed_points.repair_point_cloud",
        "conditions": {name: str(path.resolve()) for name, path in condition_roots.items()},
    }
    (args.output_root / "condition_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
