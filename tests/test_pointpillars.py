import unittest

import numpy as np
import torch

from Fault_Localization_Model.bev_utils import metric_to_grid
from models.two_stage_reconstruction_head import (
    BEVGridGeometry,
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    HRNetConfig,
    Pillarizer,
    PointPillarsConfig,
    PointPillarsEncoder,
    build_configs,
)


def _geometry() -> BEVGridGeometry:
    return BEVGridGeometry(0.0, 64.0, -32.0, 32.0, 320, 320)


class PointPillarsTests(unittest.TestCase):
    def test_grid_indices_match_existing_bev_orientation(self):
        geometry = _geometry()
        xyz = np.asarray(
            [
                [0.01, -31.99, 0.0],
                [10.11, 3.25, 1.0],
                [63.99, 31.99, -1.0],
            ],
            dtype=np.float32,
        )
        _, expected_rows, expected_cols, valid, _, _ = metric_to_grid(
            xyz,
            x_range=(geometry.x_min, geometry.x_max),
            y_range=(geometry.y_min, geometry.y_max),
            resolution=geometry.pillar_size_x,
        )
        pillarizer = Pillarizer(
            geometry,
            raw_channels=3,
            max_points_per_pillar=100,
            max_pillars=12000,
        )
        rows, cols, actual_valid = pillarizer.grid_indices(torch.from_numpy(xyz))
        self.assertTrue(np.array_equal(actual_valid.numpy(), valid))
        self.assertTrue(np.array_equal(rows[actual_valid].numpy(), expected_rows))
        self.assertTrue(np.array_equal(cols[actual_valid].numpy(), expected_cols))

    def test_encoder_returns_dense_320_grid_and_bounded_pillars(self):
        encoder = PointPillarsEncoder(
            _geometry(),
            raw_channels=4,
            output_channels=64,
            max_points_per_pillar=2,
            max_pillars=12000,
        ).eval()
        points = torch.tensor(
            [
                [1.01, 0.01, 0.0, 10.0],
                [1.02, 0.02, 0.1, 11.0],
                [1.03, 0.03, 0.2, 12.0],
                [5.0, 5.0, 1.0, 20.0],
            ]
        )
        with torch.no_grad():
            pseudo_bev, statistics = encoder((points,))
        self.assertEqual(tuple(pseudo_bev.shape), (1, 64, 320, 320))
        self.assertEqual(int(statistics["nonempty_pillars"][0]), 2)
        self.assertEqual(int(statistics["retained_points"][0]), 3)
        self.assertEqual(int(statistics["maximum_points_per_pillar"][0]), 3)
        self.assertGreater(float(statistics["empty_pillar_fraction"][0]), 0.99)

    def test_batched_encoder_matches_per_sample_reference(self):
        encoder = PointPillarsEncoder(
            _geometry(),
            raw_channels=4,
            output_channels=8,
            max_points_per_pillar=2,
            max_pillars=12000,
        ).eval()
        point_clouds = (
            torch.tensor(
                [
                    [1.01, 0.01, 0.0, 10.0],
                    [1.02, 0.02, 0.1, 11.0],
                    [1.03, 0.03, 0.2, 12.0],
                    [5.0, 5.0, 1.0, 20.0],
                ]
            ),
            torch.tensor(
                [
                    [2.01, -1.01, 0.0, 13.0],
                    [2.02, -1.02, 0.1, 14.0],
                    [8.0, 4.0, 1.0, 21.0],
                ]
            ),
        )
        with torch.no_grad():
            actual, actual_statistics = encoder(point_clouds)
            pillarized = tuple(
                encoder.pillarizer(points) for points in point_clouds
            )
            pillar_counts = [len(item.pillar_rows) for item in pillarized]
            features = torch.cat([item.features for item in pillarized])
            offsets = []
            offset = 0
            for item, count in zip(pillarized, pillar_counts):
                offsets.append(item.point_to_pillar + offset)
                offset += count
            encoded = encoder.feature_net(
                features,
                torch.cat(offsets),
                sum(pillar_counts),
            )
            expected = torch.stack(
                [
                    encoder.scatter(features, item.pillar_rows, item.pillar_cols)
                    for features, item in zip(
                        encoded.split(pillar_counts), pillarized
                    )
                ]
            )
        self.assertTrue(torch.equal(actual, expected))
        for name in actual_statistics:
            expected_statistic = torch.stack(
                [item.statistics[name] for item in pillarized]
            )
            self.assertTrue(
                torch.allclose(
                    actual_statistics[name],
                    expected_statistic,
                ),
                name,
            )

    def test_enabled_config_changes_inputs_but_not_three_channel_target(self):
        model_config, _, _ = build_configs(
            {
                "pointpillars": {"enabled": True},
                "coarse_reconstruction": {},
            }
        )
        self.assertEqual(model_config.lidar_channels, 64)
        self.assertEqual(model_config.radar_channels, 64)
        self.assertEqual(model_config.target_lidar_channels, 3)
        self.assertEqual(model_config.local_input_channels, 130)

    def test_model_masks_features_and_both_pillar_encoders_receive_gradients(self):
        pointpillars = PointPillarsConfig(
            enabled=True,
            output_channels=64,
            max_points_per_pillar=100,
            max_pillars=12000,
        )
        config = CoarseReconstructionConfig(
            lidar_channels=64,
            radar_channels=64,
            target_lidar_channels=3,
            pointpillars=pointpillars,
            hrnet=HRNetConfig(
                base_channels=2,
                blocks_per_stage=1,
                residual_blocks_per_branch=1,
            ),
        )
        model = CoarseReconstructionModel(config, grid_geometry=_geometry()).train()
        faulty_bev = torch.rand(1, 3, 320, 320)
        radar_bev = torch.zeros(1, 4, 320, 320)
        reconstruction = torch.zeros(1, 1, 320, 320)
        reconstruction[:, :, 260:300, 120:180] = 1.0
        healthy = 1.0 - reconstruction
        halo = healthy.clone()
        lidar_points = (
            torch.tensor(
                [
                    [5.0, 0.0, 0.0, 20.0],
                    [6.0, 1.0, 0.5, 30.0],
                    [20.0, -5.0, 1.0, 40.0],
                    [30.0, 8.0, -0.5, 50.0],
                ]
            ),
        )
        radar_points = (
            torch.tensor(
                [
                    [5.0, 0.0, 0.0, 0.2, -0.3],
                    [6.0, 1.0, 0.5, 0.4, 0.2],
                    [20.0, -5.0, 1.0, 0.1, 0.5],
                    [30.0, 8.0, -0.5, 0.6, -0.1],
                ]
            ),
        )
        outputs = model(
            faulty_bev,
            radar_bev,
            reconstruction,
            healthy,
            halo,
            faulty_lidar_points=lidar_points,
            radar_points=radar_points,
        )
        self.assertEqual(tuple(outputs["lidar_pillar_bev"].shape), (1, 64, 320, 320))
        self.assertEqual(tuple(outputs["radar_pillar_bev"].shape), (1, 64, 320, 320))
        self.assertEqual(tuple(outputs["reconstruction_mask"].shape), (1, 1, 320, 320))
        erased_values = outputs["erased_lidar_features"] * reconstruction
        self.assertEqual(float(erased_values.detach().abs().max()), 0.0)
        outside = 1.0 - reconstruction
        self.assertTrue(
            torch.equal(
                outputs["coarse_lidar_bev"] * outside,
                faulty_bev * outside,
            )
        )

        outputs["coarse_lidar_bev"].sum().backward()
        lidar_gradient = model.lidar_pillar_encoder.feature_net.linear.weight.grad
        radar_gradient = model.radar_pillar_encoder.feature_net.linear.weight.grad
        self.assertIsNotNone(lidar_gradient)
        self.assertIsNotNone(radar_gradient)
        self.assertGreater(float(lidar_gradient.abs().sum()), 0.0)
        self.assertGreater(float(radar_gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
