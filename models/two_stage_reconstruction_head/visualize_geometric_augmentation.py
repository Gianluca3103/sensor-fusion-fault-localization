"""Save forced geometric-augmentation sanity checks for one training sample."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.Fault_Localization.training_utils import _split_paths
from models.two_stage_reconstruction_head import (
    CoarseReconstructionDataset,
    GeometricTransform,
    ReconstructionGeometricAugmentation,
    build_augmentation_config,
    build_configs,
    load_config,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def _lidar_rgb(tensor):
    return tensor.float().clamp(0, 1).permute(1, 2, 0).numpy()


def _radar_rgb(tensor):
    radar = tensor.float().clamp(0, 1).numpy()
    return np.stack(
        (radar[2], radar[3], np.maximum(radar[0], radar[1])), axis=-1
    ).clip(0, 1)


def _overlay(item):
    lidar = item["faulty_bev"][0].numpy() >= 0.5
    radar = item["radar_bev"][0].numpy() >= 0.5
    return np.stack((radar, lidar, radar & lidar), axis=-1).astype(np.float32)


def _save_case(destination, original, augmented, title):
    figure, axes = plt.subplots(2, 5, figsize=(22, 9), facecolor="black")
    for row, (item, prefix) in enumerate(
        ((original, "Original"), (augmented, "Augmented"))
    ):
        panels = (
            (_lidar_rgb(item["clean_bev"]), f"{prefix} clean LiDAR", None),
            (_lidar_rgb(item["faulty_bev"]), f"{prefix} faulty LiDAR", None),
            (_radar_rgb(item["radar_bev"]), f"{prefix} radar", None),
            (
                item["reconstruction_mask"][0].numpy(),
                f"{prefix} reconstruction mask",
                "gray",
            ),
            (_overlay(item), f"{prefix} radar/LiDAR overlay", None),
        )
        for axis, (image, panel_title, cmap) in zip(axes[row], panels):
            axis.imshow(image, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
            axis.set_title(panel_title, color="white")
            axis.axis("off")
    figure.suptitle(title, color="white", fontsize=16)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=160,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def main():
    args = _parse_args()
    payload = load_config(args.config)
    model_config, _loss_config, selector_config = build_configs(payload)
    augmentation_config = build_augmentation_config(payload)
    paths = _split_paths(args.data_root, "train", None, 0)
    if not 0 <= args.sample_index < len(paths):
        raise IndexError(
            f"sample-index must be in [0,{len(paths) - 1}], got {args.sample_index}"
        )
    dataset = CoarseReconstructionDataset(
        [paths[args.sample_index]],
        args.radar_root,
        data_root=args.data_root,
        selector_config=selector_config,
        use_pointpillars=model_config.pointpillars.enabled,
    )
    original = dataset[0]
    augment = ReconstructionGeometricAugmentation(
        augmentation_config, dataset.grid_geometry
    )
    cases = {
        "01_flip": GeometricTransform(flip_y=True),
        "02_translation": GeometricTransform(
            translation_x_m=0.5, translation_y_m=-0.5
        ),
        "03_yaw": GeometricTransform(yaw_radians=math.radians(5.0)),
        "04_scale": GeometricTransform(scale=1.05),
        "05_combined": GeometricTransform(
            flip_y=True,
            scale=1.05,
            yaw_radians=math.radians(5.0),
            translation_x_m=0.5,
            translation_y_m=-0.5,
        ),
    }
    for name, transform in cases.items():
        augmented = augment.apply(original, transform=transform)
        destination = args.output_root / f"{name}.png"
        _save_case(destination, original, augmented, str(transform.to_dict()))
        print(destination)


if __name__ == "__main__":
    main()
