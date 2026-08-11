import unittest

import torch

from models.reconstruction_head import (
    BEVGridGeometry,
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    PointPillarsConfig,
    PointPillarsEncoder,
    PointPillarsOutput,
    SSTBackbone,
    SSTConfig,
    SparseRegionalAttention,
    SparseToDenseScatter,
    regional_group_indices,
)


def _geometry():
    return BEVGridGeometry(0.0, 64.0, -32.0, 32.0, 320, 320)


class SSTReconstructionTests(unittest.TestCase):
    def test_pointpillars_optionally_preserves_sparse_features_and_coordinates(self):
        encoder = PointPillarsEncoder(
            _geometry(),
            raw_channels=4,
            output_channels=8,
            max_points_per_pillar=10,
            max_pillars=100,
        ).eval()
        points = (
            torch.tensor(
                [
                    [1.01, 0.01, 0.0, 0.1],
                    [1.02, 0.02, 0.1, 0.2],
                    [5.00, 2.00, 0.5, 0.3],
                ]
            ),
        )
        with torch.no_grad():
            output = encoder(points, return_sparse=True)
        self.assertIsInstance(output, PointPillarsOutput)
        self.assertEqual(tuple(output.dense_features.shape), (1, 8, 320, 320))
        self.assertEqual(tuple(output.sparse_features.shape), (2, 8))
        self.assertEqual(tuple(output.sparse_coordinates.shape), (2, 3))
        for feature, coordinate in zip(
            output.sparse_features, output.sparse_coordinates
        ):
            batch, row, col = coordinate
            self.assertTrue(
                torch.equal(
                    feature,
                    output.dense_features[batch, :, row, col],
                )
            )

    def test_normal_and_shifted_region_membership(self):
        coordinates = torch.tensor([[0, 11, 3], [0, 12, 3]])
        normal = regional_group_indices(coordinates, 12, shift_size_cells=0)
        shifted = regional_group_indices(coordinates, 12, shift_size_cells=6)
        self.assertFalse(torch.equal(normal[0], normal[1]))
        self.assertTrue(torch.equal(shifted[0], shifted[1]))
        self.assertTrue(torch.equal(coordinates, torch.tensor([[0, 11, 3], [0, 12, 3]])))

    def test_attention_does_not_cross_region_boundaries(self):
        torch.manual_seed(3)
        config = SSTConfig(
            token_dim=8,
            num_blocks=1,
            num_heads=2,
            mlp_hidden_dim=16,
            region_size_cells=4,
            shift_size_cells=2,
        )
        attention = SparseRegionalAttention(
            config, shift_size_cells=0
        ).eval()
        coordinates = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 8, 8], [0, 8, 9]]
        )
        features = torch.randn(4, 8)
        changed = features.clone()
        changed[0] += 100.0
        with torch.no_grad():
            original, _ = attention(features, coordinates, (12, 12))
            modified, _ = attention(changed, coordinates, (12, 12))
        self.assertTrue(torch.equal(original[2:], modified[2:]))
        self.assertFalse(torch.equal(original[:2], modified[:2]))

    def test_sparse_attention_supports_mixed_precision_packing(self):
        config = SSTConfig(
            token_dim=8,
            num_blocks=1,
            num_heads=2,
            mlp_hidden_dim=16,
            region_size_cells=4,
            shift_size_cells=2,
        )
        attention = SparseRegionalAttention(
            config, shift_size_cells=0
        ).eval()
        coordinates = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 5, 5]]
        )
        features = torch.randn(3, 8)
        with torch.no_grad(), torch.autocast(
            device_type="cpu", dtype=torch.bfloat16
        ):
            output, counts = attention(features, coordinates, (8, 8))
        self.assertEqual(tuple(output.shape), (3, 8))
        self.assertEqual(int(counts.sum()), 3)
        self.assertTrue(torch.isfinite(output).all())

    def test_backbone_is_single_stride_and_scatter_is_high_resolution(self):
        config = SSTConfig(
            token_dim=8,
            num_blocks=3,
            num_heads=2,
            mlp_hidden_dim=16,
            region_size_cells=12,
            shift_size_cells=6,
        )
        backbone = SSTBackbone(config).eval()
        coordinates = torch.tensor(
            [[0, 0, 0], [0, 11, 11], [0, 12, 12], [1, 319, 319]]
        )
        features = torch.randn(4, 8)
        with torch.no_grad():
            final, debug = backbone(features, coordinates, (320, 320))
            dense = SparseToDenseScatter(8)(
                final, coordinates, 2, (320, 320)
            )
        self.assertTrue(
            torch.equal(debug["coordinates_before"], coordinates)
        )
        self.assertTrue(torch.equal(debug["coordinates_after"], coordinates))
        self.assertEqual(tuple(dense.shape), (2, 8, 320, 320))

    def test_full_sst_model_has_no_lidar_leakage_gradients_and_outside_invariant(self):
        pointpillars = PointPillarsConfig(
            enabled=True,
            output_channels=8,
            max_points_per_pillar=10,
            max_pillars=100,
        )
        sst = SSTConfig(
            token_dim=8,
            num_blocks=2,
            num_heads=2,
            mlp_hidden_dim=16,
            region_size_cells=12,
            shift_size_cells=6,
        )
        config = CoarseReconstructionConfig(
            backbone="sst",
            lidar_channels=8,
            radar_channels=8,
            pointpillars=pointpillars,
            sst=sst,
        )
        model = CoarseReconstructionModel(
            config, grid_geometry=_geometry()
        ).train()
        faulty = torch.rand(1, 3, 320, 320)
        radar_bev = torch.zeros(1, 4, 320, 320)
        reconstruction = torch.zeros(1, 1, 320, 320)
        reconstruction[:, :, 293:297, 159:163] = 1.0
        healthy = torch.zeros_like(reconstruction)
        healthy[:, :, 289:301, 155:167] = 1.0 - reconstruction[:, :, 289:301, 155:167]
        halo = healthy.clone()
        lidar_points = (
            torch.tensor(
                [
                    [5.0, 0.0, 0.0, 0.1],
                    [5.1, 0.1, 0.1, 0.2],
                    [5.6, 0.0, 0.2, 0.3],
                    [5.7, 0.1, 0.3, 0.4],
                ]
            ),
        )
        radar_points = (
            torch.tensor(
                [
                    [5.0, 0.0, 0.0, 0.4, -0.2],
                    [5.1, 0.1, 0.1, 0.3, 0.1],
                    [5.6, 0.0, 0.2, 0.2, 0.3],
                    [5.7, 0.1, 0.3, 0.1, -0.1],
                ]
            ),
        )
        outputs = model(
            faulty,
            radar_bev,
            reconstruction,
            healthy,
            halo,
            faulty_lidar_points=lidar_points,
            radar_points=radar_points,
        )
        self.assertEqual(tuple(outputs["coarse_lidar_bev"].shape), (1, 3, 320, 320))
        self.assertTrue(
            torch.equal(
                outputs["sst_coordinates_before"],
                outputs["sst_coordinates_after"],
            )
        )
        trusted = outputs["trusted_lidar_coordinates"]
        trusted_mask_values = reconstruction[
            trusted[:, 0], 0, trusted[:, 1], trusted[:, 2]
        ]
        self.assertEqual(float(trusted_mask_values.max()), 0.0)
        outside = 1.0 - reconstruction
        self.assertTrue(
            torch.equal(outputs["coarse_lidar_bev"] * outside, faulty * outside)
        )

        outputs["coarse_lidar_bev"].sum().backward()
        modules = [
            model.lidar_pillar_encoder.feature_net,
            model.radar_pillar_encoder.feature_net,
            model.sst_token_builder.projection,
            *model.sst_backbone.blocks,
            model.sst_reconstruction_head,
        ]
        for module in modules:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
