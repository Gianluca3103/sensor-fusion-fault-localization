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
from torch.utils.data import DataLoader, Dataset

from Fault_Localization_Model.sample_utils import load_sample_metadata
from Fault_Localization_Model.io_utils import atomic_savez_compressed
from Fault_Localization_Model.vod_dataset import load_vod_lidar, resolve_vod_public_root
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head import (
    coarse_reconstruction_collate,
    load_bev_triplet,
    load_frozen_coarse_model,
)


class PointInputDataset(Dataset):
    """Load aligned faulty LiDAR and radar points without fault selection."""

    def __init__(self, paths, radar_root: Path):
        self.paths = tuple(Path(path) for path in paths)
        self.radar_root = radar_root

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        return load_bev_triplet(
            self.paths[index],
            self.radar_root,
            include_pointpillars_inputs=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--encoder-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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
    counts = {}

    for split in dict.fromkeys(args.splits):
        paths = _split_paths(args.data_root, split, None, args.seed)
        dataset = PointInputDataset(paths, args.radar_root)
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
                    atomic_savez_compressed(
                        destination,
                        clean_features=clean_features[index].float().cpu().numpy().astype(np.float16),
                        faulty_features=faulty_features[index].float().cpu().numpy().astype(np.float16),
                        radar_features=radar_features[index].float().cpu().numpy().astype(np.float16),
                    )
                    written += 1
                    if written % 100 == 0:
                        print(f"{split}: cached {written}/{len(dataset)}", flush=True)
        counts[split] = written

    manifest = {
        "format_version": 2,
        "tensor_name": "dense_features",
        "feature_interface_alias": "post_pillar_scatter",
        "lidar_shape": [
            int(lidar_encoder.scatter.channels),
            feature_geometry.height,
            feature_geometry.width,
        ],
        "radar_shape": [
            int(radar_encoder.scatter.channels),
            feature_geometry.height,
            feature_geometry.width,
        ],
        "geometry": feature_geometry.to_dict(),
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
        "encoder_epoch": int(checkpoint.get("epoch", -1)),
        "encoder_frozen": True,
        "same_lidar_encoder_for_clean_and_faulty": True,
        "fault_selector_used": False,
        "reconstruction_masks_used": False,
        "clean_features_are_targets_only": True,
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
