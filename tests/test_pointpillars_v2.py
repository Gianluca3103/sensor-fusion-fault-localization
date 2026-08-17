import unittest

import torch

from models.two_stage_reconstruction_head import (
    BEVGridGeometry,
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    HRNetConfig,
    NeighborAwarePillarEnhancer,
    PointPillarsEncoderV2,
    PointPillarsOutput,
    PointPillarsV2Config,
    build_configs,
)


def _geometry() -> BEVGridGeometry:
    return BEVGridGeometry(0.0, 64.0, -32.0, 32.0, 320, 320)


def _encoder(*, channels: int = 64) -> PointPillarsEncoderV2:
    return PointPillarsEncoderV2(
        _geometry(),
        raw_channels=4,
        output_channels=channels,
        max_points_per_pillar=100,
        max_pillars=None,
        neighbor_radius_m=0.4,
        neighbor_max_neighbors=16,
        neighbor_initial_scale=0.1,
    ).eval()


class NeighborAwarePillarEnhancerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.enhancer = NeighborAwarePillarEnhancer(
            _geometry(),
            4,
            radius_m=0.4,
            max_neighbors=16,
            initial_scale=0.1,
        ).eval()

    def test_no_neighbor_is_exact_passthrough(self):
        features = torch.randn(2, 4)
        batches = torch.zeros(2, dtype=torch.long)
        rows = torch.tensor([20, 40])
        cols = torch.tensor([20, 40])

        enhanced, statistics = self.enhancer(
            features,
            batches,
            rows,
            cols,
            batch_size=1,
            return_statistics=True,
        )

        self.assertTrue(torch.equal(enhanced, features))
        self.assertEqual(int(statistics["pillars_with_no_neighbors"][0]), 2)
        self.assertEqual(float(statistics["neighborless_pillar_fraction"][0]), 1.0)

    def test_neighboring_pillars_interact(self):
        batches = torch.zeros(2, dtype=torch.long)
        rows = torch.tensor([20, 21])
        cols = torch.tensor([20, 20])
        first = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        second = first.clone()
        second[1] = torch.tensor([3.0, -2.0, 1.0, 4.0])

        first_output = self.enhancer(first, batches, rows, cols)
        second_output = self.enhancer(second, batches, rows, cols)

        self.assertFalse(torch.allclose(first_output[0], second_output[0]))

    def test_distant_pillars_do_not_interact(self):
        batches = torch.zeros(2, dtype=torch.long)
        rows = torch.tensor([20, 30])
        cols = torch.tensor([20, 20])
        first = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        second = first.clone()
        second[1] = 100.0

        first_output = self.enhancer(first, batches, rows, cols)
        second_output = self.enhancer(second, batches, rows, cols)

        self.assertTrue(torch.equal(first_output[0], second_output[0]))

    def test_identical_coordinates_in_different_batches_are_isolated(self):
        batches = torch.tensor([0, 1])
        rows = torch.tensor([20, 20])
        cols = torch.tensor([20, 20])
        first = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        second = first.clone()
        second[1] = 100.0

        first_output = self.enhancer(first, batches, rows, cols)
        second_output = self.enhancer(second, batches, rows, cols)

        self.assertTrue(torch.equal(first_output[0], second_output[0]))

    def test_metric_radius_handles_non_square_cells(self):
        geometry = BEVGridGeometry(0.0, 2.0, 0.0, 4.0, 10, 10)
        enhancer = NeighborAwarePillarEnhancer(
            geometry,
            4,
            radius_m=0.3,
            max_neighbors=8,
        ).eval()
        features = torch.randn(3, 4)
        batches = torch.zeros(3, dtype=torch.long)
        rows = torch.tensor([5, 6, 5])
        cols = torch.tensor([5, 5, 6])

        _, statistics = enhancer(
            features,
            batches,
            rows,
            cols,
            batch_size=1,
            return_statistics=True,
        )

        # One row is 0.2 m and one column is 0.4 m for this geometry.
        self.assertEqual(int(statistics["maximum_neighbors_per_pillar"][0]), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_and_backward(self):
        enhancer = NeighborAwarePillarEnhancer(
            _geometry(),
            4,
            radius_m=0.4,
            max_neighbors=16,
        ).cuda()
        features = torch.randn(2, 4, device="cuda", requires_grad=True)
        batches = torch.zeros(2, dtype=torch.long, device="cuda")
        rows = torch.tensor([20, 21], device="cuda")
        cols = torch.tensor([20, 20], device="cuda")

        enhanced = enhancer(features, batches, rows, cols)
        enhanced.sum().backward()

        self.assertEqual(tuple(enhanced.shape), (2, 4))
        self.assertIsNotNone(features.grad)


class PointPillarsEncoderV2Tests(unittest.TestCase):
    def test_output_shape_for_two_samples(self):
        encoder = _encoder()
        point_clouds = (
            torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
            torch.tensor([[2.0, 1.0, 0.5, 2.0]]),
        )
        with torch.no_grad():
            dense, statistics = encoder(point_clouds)

        self.assertEqual(tuple(dense.shape), (2, 64, 320, 320))
        self.assertEqual(tuple(statistics["average_neighbors_per_pillar"].shape), (2,))

    def test_empty_point_cloud_batch_returns_valid_zero_output(self):
        encoder = _encoder(channels=8)
        empty = torch.empty(0, 4)
        with torch.no_grad():
            dense, statistics = encoder((empty, empty))

        self.assertEqual(tuple(dense.shape), (2, 8, 320, 320))
        self.assertEqual(int(torch.count_nonzero(dense)), 0)
        self.assertEqual(int(statistics["nonempty_pillars"].sum()), 0)
        self.assertEqual(
            int(statistics["pillars_with_no_neighbors"].sum()),
            0,
        )

    def test_dense_and_sparse_interfaces_use_enhanced_features(self):
        encoder = _encoder(channels=8)
        points = torch.tensor(
            [
                [1.01, 0.01, 0.0, 1.0],
                [1.21, 0.01, 0.0, 2.0],
            ]
        )
        with torch.no_grad():
            dense, statistics = encoder((points,), return_sparse=False)
            sparse = encoder((points,), return_sparse=True)

        self.assertIsInstance(sparse, PointPillarsOutput)
        self.assertTrue(torch.equal(dense, sparse.dense_features))
        self.assertEqual(tuple(sparse.sparse_features.shape), (2, 8))
        self.assertEqual(tuple(sparse.sparse_coordinates.shape), (2, 3))
        for index, (_, row, col) in enumerate(sparse.sparse_coordinates):
            self.assertTrue(
                torch.equal(
                    sparse.dense_features[0, :, row, col],
                    sparse.sparse_features[index],
                )
            )
        self.assertEqual(
            int(statistics["maximum_neighbors_per_pillar"][0]),
            1,
        )

    def test_configuration_selects_v2_without_enabling_baseline(self):
        model, _loss, _selector = build_configs(
            {
                "pointpillars": {"enabled": False},
                "pointpillars_v2": {
                    "enabled": True,
                    "max_pillars": None,
                },
                "coarse_reconstruction": {},
            }
        )

        self.assertTrue(model.pointpillars_v2.enabled)
        self.assertFalse(model.pointpillars.enabled)
        self.assertTrue(model.pointpillars_enabled)
        self.assertEqual(model.lidar_channels, 64)
        self.assertEqual(model.radar_channels, 64)

    def test_hrnet_integration_uses_v2_for_both_sensors(self):
        v2 = PointPillarsV2Config(
            enabled=True,
            output_channels=8,
            max_pillars=None,
        )
        config = CoarseReconstructionConfig(
            lidar_channels=8,
            radar_channels=8,
            pointpillars_v2=v2,
            hrnet=HRNetConfig(
                base_channels=2,
                blocks_per_stage=1,
                residual_blocks_per_branch=1,
            ),
        )
        restored = CoarseReconstructionConfig.from_dict(config.to_dict())
        model = CoarseReconstructionModel(
            restored,
            grid_geometry=_geometry(),
        ).eval()
        faulty = torch.zeros(1, 3, 320, 320)
        radar = torch.zeros(1, 4, 320, 320)
        repair = torch.zeros(1, 1, 320, 320)
        repair[:, :, 280:300, 150:170] = 1.0
        context = 1.0 - repair
        lidar_points = (
            torch.tensor(
                [[4.01, 0.01, 0.0, 1.0], [4.21, 0.01, 0.1, 2.0]]
            ),
        )
        radar_points = (
            torch.tensor(
                [
                    [4.01, 0.01, 0.0, 1.0, 0.2],
                    [4.21, 0.01, 0.1, 2.0, 0.3],
                ]
            ),
        )

        with torch.no_grad():
            outputs = model(
                faulty,
                radar,
                repair,
                context,
                context,
                faulty_lidar_points=lidar_points,
                radar_points=radar_points,
            )

        self.assertIsInstance(model.lidar_pillar_encoder, PointPillarsEncoderV2)
        self.assertIsInstance(model.radar_pillar_encoder, PointPillarsEncoderV2)
        self.assertEqual(tuple(outputs["coarse_lidar_bev"].shape), (1, 3, 320, 320))
        self.assertEqual(
            int(
                outputs["lidar_pillar_statistics"][
                    "maximum_neighbors_per_pillar"
                ][0]
            ),
            1,
        )

    def test_configuration_rejects_two_enabled_encoders(self):
        with self.assertRaisesRegex(ValueError, "Enable only one"):
            build_configs(
                {
                    "pointpillars": {"enabled": True},
                    "pointpillars_v2": {"enabled": True},
                    "coarse_reconstruction": {},
                }
            )

    def test_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, "neighbor_radius_m"):
            PointPillarsV2Config(neighbor_radius_m=0.0).validate()
        with self.assertRaisesRegex(ValueError, "neighbor_max_neighbors"):
            PointPillarsV2Config(neighbor_max_neighbors=0).validate()
        with self.assertRaisesRegex(ValueError, "neighbor_initial_scale"):
            PointPillarsV2Config(neighbor_initial_scale=-0.1).validate()


if __name__ == "__main__":
    unittest.main()
