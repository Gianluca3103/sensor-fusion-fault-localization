"""Stage-I online augmentation keeps aligned sensor evidence and labels paired."""

import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.data import VoDStage1Dataset
from models.radar_lidar_stage1.online_augmentation import (
    GeometricAugmentationConfig, GeometricTransform, augment_pair, frame_generator,
    sample_transform,
)


def test_shared_transform_preserves_alignment_and_measured_attributes():
    grid = VoxelGrid((-4, -4, -2), (4, 4, 2), (0.5, 0.5, 0.5))
    radar = torch.tensor([[1., 2., 0.2, 9., -3., 0.5, 2.]])
    lidar = torch.tensor([[1., 2., 0.2, 0.8]])
    transform = GeometricTransform(True, 1.02, math.radians(4), 0.1, -0.2)
    radar_out, lidar_out = augment_pair(radar, lidar, transform, grid)
    torch.testing.assert_close(radar_out[:, :3], lidar_out[:, :3])
    torch.testing.assert_close(radar_out[:, 3:], radar[:, 3:])
    torch.testing.assert_close(lidar_out[:, 3:], lidar[:, 3:])
    torch.testing.assert_close(radar[:, :3], torch.tensor([[1., 2., 0.2]]))


def test_transform_drops_points_leaving_voxel_grid():
    grid = VoxelGrid((-2, -2, -2), (2, 2, 2), (0.5, 0.5, 0.5))
    radar = torch.tensor([[1.9, 0., 0., 1., 0., 0., 0.]])
    lidar = torch.tensor([[1.9, 0., 0., 1.]])
    r, l = augment_pair(radar, lidar, GeometricTransform(translation_x_m=0.2), grid)
    assert r.shape == (0, 7) and l.shape == (0, 4)


def test_per_frame_epoch_seed_is_deterministic_and_changes_each_epoch():
    config = GeometricAugmentationConfig(enabled=True)
    first = sample_transform(config, generator=frame_generator(9, 1, "train", "00001"))
    repeated = sample_transform(config, generator=frame_generator(9, 1, "train", "00001"))
    next_epoch = sample_transform(config, generator=frame_generator(9, 2, "train", "00001"))
    assert first == repeated
    assert first != next_epoch


def test_dataset_uses_same_online_transform_and_validation_is_unmodified():
    frame = SimpleNamespace(frame_id="00001", split="train", radar_path=Path("radar"),
                            lidar_path=Path("lidar"), lidar_calibration_path=Path("lidar_cal"),
                            radar_calibration_path=Path("radar_cal"))
    radar = np.array([[1., 2., 0., 5., -1., 0., 3.]], dtype=np.float32)
    lidar = np.array([[1., 2., 0., 0.7]], dtype=np.float32)
    with mock.patch("models.radar_lidar_stage1.data.discover_vod_frames", return_value=[frame]), \
         mock.patch("models.radar_lidar_stage1.data.load_vod_radar", return_value=radar), \
         mock.patch("models.radar_lidar_stage1.data.align_radar_to_lidar", side_effect=lambda value, _: value), \
         mock.patch("models.radar_lidar_stage1.data.load_vod_radar_to_lidar", return_value=None), \
         mock.patch("models.radar_lidar_stage1.data.load_vod_lidar", return_value=lidar):
        dataset = VoDStage1Dataset("unused", "train",
                                   augmentation_config=GeometricAugmentationConfig(enabled=True),
                                   augmentation_seed=10)
        dataset.set_epoch(1)
        first = dataset[0]
        dataset.set_epoch(2)
        second = dataset[0]
        dataset.set_epoch(1)
        repeated = dataset[0]
        torch.testing.assert_close(first["radar"][:, :3], first["clean_lidar"][:, :3])
        torch.testing.assert_close(first["radar"], repeated["radar"])
        assert first["augmentation_transform"] != second["augmentation_transform"]
        torch.testing.assert_close(first["radar"][:, 3:], torch.tensor([[5., -1., 0., 3.]]))
        with pytest.raises(ValueError, match="paired training"):
            VoDStage1Dataset("unused", "val",
                             augmentation_config=GeometricAugmentationConfig(enabled=True))


def test_shared_configuration_is_exactly_the_existing_bev_configuration():
    from models.two_stage_reconstruction_head.geometric_augmentation import (
        GeometricAugmentationConfig as ExistingConfig,
        ReconstructionGeometricAugmentation,
    )
    from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry
    assert ExistingConfig is GeometricAugmentationConfig
    config = ExistingConfig(enabled=True)
    bev = ReconstructionGeometricAugmentation(config, BEVGridGeometry(
        x_min=-4, x_max=4, y_min=-4, y_max=4, height=16, width=16))
    first = bev.sample_transform(generator=torch.Generator().manual_seed(7))
    second = sample_transform(config, generator=torch.Generator().manual_seed(7))
    assert first == second
