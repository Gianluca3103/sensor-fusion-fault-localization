import unittest

import torch

from models.two_stage_reconstruction_head import (
    BEVGridGeometry,
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    PillarFeatureNet,
    PillarFeatureNetV3,
    PointPillarsEncoderV3,
    PointPillarsOutput,
    PointPillarsV3Config,
    build_configs,
)


def _geometry() -> BEVGridGeometry:
    return BEVGridGeometry(-32.0, 32.0, -32.0, 32.0, 320, 320)


def _encoder(*, channels: int = 64, **overrides) -> PointPillarsEncoderV3:
    settings = {
        "use_mean_pool": True,
        "use_point_residual": True,
        "point_residual_hidden_channels": 64,
        "initial_residual_scale": 0.1,
    }
    settings.update(overrides)
    return PointPillarsEncoderV3(
        _geometry(),
        raw_channels=4,
        output_channels=channels,
        max_points_per_pillar=100,
        max_pillars=None,
        **settings,
    )


class PillarFeatureNetV3Tests(unittest.TestCase):
    def test_single_point_pooling_and_attention(self):
        network = PillarFeatureNetV3(4, 8).eval()
        features = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output, debug = network(
            features,
            torch.tensor([0]),
            1,
            return_diagnostics=True,
        )
        encoded = network._encode_points(features)
        self.assertTrue(torch.allclose(debug["max_pooled"], encoded))
        self.assertTrue(torch.allclose(debug["mean_pooled"], encoded))
        self.assertTrue(
            torch.allclose(debug["attention_weights"], torch.ones(1, 1))
        )
        self.assertEqual(output.shape, (1, 8))

    def test_mean_pool_matches_manual_mean(self):
        network = PillarFeatureNetV3(3, 2)
        encoded = torch.tensor(
            [[1.0, 2.0], [3.0, 6.0], [10.0, 20.0]]
        )
        membership = torch.tensor([0, 0, 1])
        means, counts = network._mean_pool(encoded, membership, 2)
        expected = torch.tensor([[2.0, 4.0], [10.0, 20.0]])
        self.assertTrue(torch.equal(means, expected))
        self.assertTrue(torch.equal(counts, torch.tensor([2, 1])))

    def test_attention_normalizes_independently_per_pillar(self):
        network = PillarFeatureNetV3(3, 4)
        scores = torch.tensor([[1.0], [3.0], [-2.0], [0.5], [4.0]])
        membership = torch.tensor([0, 0, 1, 1, 1])
        weights = network._segment_softmax(scores, membership, 2)
        sums = torch.zeros(2, 1).index_add_(0, membership, weights)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))
        self.assertTrue(torch.isfinite(weights).all())

    def test_attention_supports_half_precision_scores(self):
        network = PillarFeatureNetV3(3, 4)
        scores = torch.tensor(
            [[1.0], [3.0], [-2.0], [0.5], [4.0]], dtype=torch.float16
        )
        membership = torch.tensor([0, 0, 1, 1, 1])
        weights = network._segment_softmax(scores, membership, 2)
        sums = torch.zeros(2, 1, dtype=weights.dtype).index_add_(
            0, membership, weights
        )
        self.assertEqual(weights.dtype, torch.float16)
        self.assertTrue(torch.allclose(sums.float(), torch.ones_like(sums).float()))
        self.assertTrue(torch.isfinite(weights).all())

    def test_pillars_are_isolated(self):
        torch.manual_seed(4)
        network = PillarFeatureNetV3(3, 8).eval()
        membership = torch.tensor([0, 0, 1, 1])
        features = torch.randn(4, 3)
        first = network(features, membership, 2)
        changed = features.clone()
        changed[2:] += 100.0
        second = network(changed, membership, 2)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertFalse(torch.equal(first[1], second[1]))

    def test_point_residual_is_sensitive_to_individual_points(self):
        torch.manual_seed(9)
        network = PillarFeatureNetV3(3, 8).eval()
        membership = torch.zeros(3, dtype=torch.long)
        features = torch.randn(3, 3)
        first = network(features, membership, 1)
        changed = features.clone()
        changed[1] += 5.0
        second = network(changed, membership, 1)
        self.assertFalse(torch.allclose(first, second))

    def test_backward_reaches_every_v3_branch(self):
        torch.manual_seed(12)
        network = PillarFeatureNetV3(5, 8)
        features = torch.randn(7, 5, requires_grad=True)
        membership = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        output = network(features, membership, 3)
        output.square().sum().backward()
        parameters = (
            network.linear.weight,
            network.point_residual_mlp[0].weight,
            network.point_score.weight,
            network.fusion[0].weight,
            network.residual_scale,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_baseline_compatible_mode_matches_baseline_feature_net(self):
        torch.manual_seed(15)
        baseline = PillarFeatureNet(5, 8).eval()
        v3 = PillarFeatureNetV3(
            5,
            8,
            use_mean_pool=False,
            use_point_residual=False,
        ).eval()
        v3.linear.load_state_dict(baseline.linear.state_dict())
        v3.normalization.load_state_dict(baseline.normalization.state_dict())
        features = torch.randn(6, 5)
        membership = torch.tensor([0, 0, 1, 1, 1, 2])
        self.assertTrue(
            torch.equal(
                baseline(features, membership, 3),
                v3(features, membership, 3),
            )
        )

    def test_mean_only_and_point_residual_only_ablations(self):
        features = torch.randn(6, 5)
        membership = torch.tensor([0, 0, 1, 1, 1, 2])
        for mean_enabled, residual_enabled in ((True, False), (False, True)):
            network = PillarFeatureNetV3(
                5,
                8,
                use_mean_pool=mean_enabled,
                use_point_residual=residual_enabled,
            )
            output = network(features, membership, 3)
            self.assertEqual(output.shape, (3, 8))
            self.assertTrue(torch.isfinite(output).all())

    def test_empty_feature_net_is_finite(self):
        network = PillarFeatureNetV3(5, 8)
        output = network(
            torch.empty(0, 5), torch.empty(0, dtype=torch.long), 0
        )
        self.assertEqual(output.shape, (0, 8))
        self.assertTrue(torch.isfinite(output).all())


class PointPillarsEncoderV3Tests(unittest.TestCase):
    def test_dense_shape_statistics_and_sparse_interface(self):
        encoder = _encoder()
        point_clouds = (
            torch.tensor(
                [[1.0, 1.0, 0.2, 0.5], [1.05, 1.02, 0.3, 0.7]]
            ),
            torch.tensor([[2.0, -1.0, 0.1, 0.2]]),
        )
        dense, statistics = encoder(point_clouds)
        sparse = encoder(point_clouds, return_sparse=True)
        self.assertEqual(dense.shape, (2, 64, 320, 320))
        self.assertIsInstance(sparse, PointPillarsOutput)
        self.assertEqual(sparse.sparse_features.shape[1], 64)
        self.assertEqual(sparse.sparse_coordinates.shape[1], 3)
        self.assertTrue(torch.isfinite(dense).all())
        for key in (
            "average_attention_entropy",
            "maximum_attention_weight",
            "average_max_mean_feature_difference",
            "average_points_receiving_attention",
            "residual_scale",
        ):
            self.assertEqual(statistics[key].shape, (2,))
            self.assertTrue(torch.isfinite(statistics[key]).all())

    def test_empty_samples_return_zero_dense_output(self):
        encoder = _encoder(channels=8)
        dense, statistics = encoder(
            (torch.empty(0, 4), torch.empty(0, 4))
        )
        self.assertEqual(dense.shape, (2, 8, 320, 320))
        self.assertEqual(torch.count_nonzero(dense).item(), 0)
        self.assertEqual(statistics["nonempty_pillars"].tolist(), [0, 0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_forward_backward(self):
        encoder = _encoder(channels=8).cuda()
        points = torch.tensor(
            [[1.0, 1.0, 0.2, 0.5], [1.05, 1.02, 0.3, 0.7]],
            device="cuda",
        )
        dense, _ = encoder((points,))
        dense.square().sum().backward()
        self.assertTrue(torch.isfinite(dense).all())
        self.assertIsNotNone(encoder.feature_net.point_score.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_amp_forward_backward_with_statistics(self):
        encoder = _encoder(channels=8).cuda()
        points = torch.tensor(
            [[1.0, 1.0, 0.2, 0.5], [1.05, 1.02, 0.3, 0.7]],
            device="cuda",
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            dense, statistics = encoder((points,))
            loss = dense.float().square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(dense).all())
        for value in statistics.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertIsNotNone(encoder.feature_net.point_score.weight.grad)


class PointPillarsV3IntegrationTests(unittest.TestCase):
    def test_config_selects_v3_and_rejects_multiple_encoders(self):
        model, loss, _ = build_configs(
            {"pointpillars_v3": {"enabled": True, "max_pillars": None}}
        )
        self.assertTrue(model.pointpillars_v3.enabled)
        self.assertFalse(model.pointpillars_v2.enabled)
        self.assertEqual(model.lidar_channels, 64)
        self.assertEqual(loss.occupancy.type, "existing")
        with self.assertRaisesRegex(ValueError, "Enable only one"):
            build_configs(
                {
                    "pointpillars_v2": {"enabled": True},
                    "pointpillars_v3": {"enabled": True},
                }
            )

    def test_hrnet_uses_v3_for_both_sensors(self):
        v3 = PointPillarsV3Config(enabled=True, output_channels=8)
        config = CoarseReconstructionConfig(
            lidar_channels=8,
            radar_channels=8,
            pointpillars_v3=v3,
        )
        model = CoarseReconstructionModel(config, grid_geometry=_geometry())
        self.assertIsInstance(model.lidar_pillar_encoder, PointPillarsEncoderV3)
        self.assertIsInstance(model.radar_pillar_encoder, PointPillarsEncoderV3)
        restored = CoarseReconstructionConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)


if __name__ == "__main__":
    unittest.main()
