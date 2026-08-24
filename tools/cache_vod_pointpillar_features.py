"""Cache clean/faulty/radar post-PillarScatter features for VoD Stage-I."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from Fault_Localization_Model.sample_utils import load_sample_metadata
from Fault_Localization_Model.io_utils import atomic_savez_compressed
from Fault_Localization_Model.vod_dataset import load_vod_lidar, resolve_vod_public_root
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head import (
    CoarseReconstructionDataset,
    coarse_reconstruction_collate,
    load_frozen_coarse_model,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _load_selector_config,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.pointpillar_feature_reconstruction import (
    project_mask_between_bev_grids,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument("--selector-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize-masks", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    public_root = resolve_vod_public_root(args.vod_root)
    model, checkpoint = load_frozen_coarse_model(
        args.encoder_checkpoint, device, allow_pointpillars=True
    )
    if model.lidar_pillar_encoder is None or model.radar_pillar_encoder is None:
        raise ValueError("encoder-checkpoint must contain LiDAR and radar PointPillars encoders")
    lidar_encoder = model.lidar_pillar_encoder.eval()
    radar_encoder = model.radar_pillar_encoder.eval()
    feature_geometry = lidar_encoder.scatter.geometry
    if radar_encoder.scatter.geometry != feature_geometry:
        raise ValueError("LiDAR and radar post-scatter geometries differ")
    selector = _load_selector_config(args.selector_config)
    counts = {}
    visualized = 0

    for split in dict.fromkeys(args.splits):
        paths = _split_paths(args.data_root, split, None, args.seed)
        dataset = CoarseReconstructionDataset(
            paths,
            args.radar_root,
            data_root=args.data_root,
            selector_config=selector,
            use_pointpillars=True,
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
        written = 0
        with torch.inference_mode():
            for batch in loader:
                clean_points = []
                for sample_path in batch["sample_path"]:
                    metadata = load_sample_metadata(sample_path)
                    frame_id = f"{int(metadata['frame_id']):05d}"
                    source = public_root / "lidar" / "training" / "velodyne" / f"{frame_id}.bin"
                    clean_points.append(
                        torch.from_numpy(load_vod_lidar(source).astype(np.float32)).to(device)
                    )
                faulty_points = model._select_lidar_fields(
                    tuple(points.to(device) for points in batch["faulty_lidar_points"])
                )
                clean_points = model._select_lidar_fields(tuple(clean_points))
                radar_points = model._select_radar_fields(
                    tuple(points.to(device) for points in batch["radar_points"])
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    clean_features, _ = lidar_encoder(clean_points)
                    faulty_features, _ = lidar_encoder(faulty_points)
                    radar_features, _ = radar_encoder(radar_points)
                for index, sample_path_value in enumerate(batch["sample_path"]):
                    sample_path = Path(sample_path_value)
                    destination = args.output_root / sample_path.relative_to(args.data_root)
                    if destination.is_file() and not args.overwrite:
                        written += 1
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source_geometry = dataset.grid_geometry
                    masks = {}
                    for source_name, target_name in (
                        ("reconstruction_mask", "feature_repair_mask"),
                        ("halo_mask", "feature_halo_mask"),
                        ("healthy_context_mask", "feature_healthy_context_mask"),
                    ):
                        masks[target_name] = project_mask_between_bev_grids(
                            batch[source_name][index].numpy(),
                            source_geometry,
                            feature_geometry,
                        )[None]
                    if visualized < args.visualize_masks:
                        import matplotlib

                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        source_mask = batch["reconstruction_mask"][index, 0].numpy()
                        projected_mask = masks["feature_repair_mask"][0]
                        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(source_mask, origin="upper")
                        axes[0].set_title("320x320 reconstruction mask")
                        axes[1].imshow(projected_mask, origin="upper")
                        axes[1].set_title("post-scatter feature mask")
                        if projected_mask.shape == source_mask.shape:
                            axes[2].imshow(
                                projected_mask - source_mask,
                                origin="upper",
                                vmin=-1,
                                vmax=1,
                                cmap="coolwarm",
                            )
                            axes[2].set_title("projected - source")
                        else:
                            axes[2].text(
                                0.5,
                                0.5,
                                "Different raster shapes\ncompare metric extents",
                                ha="center",
                                va="center",
                            )
                            axes[2].set_title("metric projection")
                        for axis in axes:
                            axis.set_axis_off()
                        mask_root = args.output_root / "mask_alignment"
                        mask_root.mkdir(parents=True, exist_ok=True)
                        figure.savefig(
                            mask_root / f"{split}_{visualized:03d}.png",
                            dpi=140,
                            bbox_inches="tight",
                        )
                        plt.close(figure)
                        visualized += 1
                    atomic_savez_compressed(
                        destination,
                        clean_features=clean_features[index].float().cpu().numpy().astype(np.float16),
                        faulty_features=faulty_features[index].float().cpu().numpy().astype(np.float16),
                        radar_features=radar_features[index].float().cpu().numpy().astype(np.float16),
                        **masks,
                    )
                    written += 1
                    if written % 100 == 0:
                        print(f"{split}: cached {written}/{len(dataset)}", flush=True)
        counts[split] = written

    manifest = {
        "format_version": 1,
        "tensor_name": "dense_features",
        "detector_interface_alias": "post_pillar_scatter",
        "shape": [
            int(lidar_encoder.scatter.channels),
            feature_geometry.height,
            feature_geometry.width,
        ],
        "geometry": feature_geometry.to_dict(),
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
        "encoder_epoch": int(checkpoint.get("epoch", -1)),
        "encoder_frozen": True,
        "same_lidar_encoder_for_clean_and_faulty": True,
        "splits": counts,
        "feature_storage_dtype": "float16",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
