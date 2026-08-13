import unittest

import torch

from models.reconstruction_head import (
    BEVGridGeometry,
    CoarseReconstructionModel,
    RadarPillarAttention,
    RadarPillarAttentionConfig,
    build_configs,
)


def _attention_config(**overrides):
    values = {
        "enabled": True,
        "attention_dim": 8,
        "num_heads": 2,
        "ff_dim": 16,
        "num_blocks": 1,
        "dropout": 0.0,
    }
    values.update(overrides)
    return RadarPillarAttentionConfig(**values)


class RadarPillarAttentionTests(unittest.TestCase):
    def test_configuration_selects_only_pointpillars_hrnet(self):
        model_config, _loss, _selector = build_configs(
            {
                "model": {"backbone": "hrnet"},
                "hrnet": {
                    "base_channels": 2,
                    "blocks_per_stage": 1,
                    "residual_blocks_per_branch": 1,
                },
                "pointpillars": {
                    "enabled": True,
                    "output_channels": 8,
                    "max_points_per_pillar": 8,
                    "max_pillars": 32,
                },
                "radar_pillar_attention": _attention_config().to_dict(),
                "coarse_reconstruction": {},
            }
        )
        self.assertTrue(model_config.radar_pillar_attention.enabled)
        self.assertEqual(model_config.local_input_channels, 18)

    def test_configuration_rejects_combined_radar_experiments(self):
        with self.assertRaisesRegex(ValueError, "requires PointPillars"):
            build_configs(
                {
                    "model": {"backbone": "hrnet"},
                    "radar_pillar_attention": _attention_config().to_dict(),
                    "coarse_reconstruction": {},
                }
            )
        with self.assertRaisesRegex(ValueError, "range-aware Radar"):
            build_configs(
                {
                    "model": {"backbone": "hrnet"},
                    "pointpillars": {
                        "enabled": True,
                        "output_channels": 8,
                    },
                    "radar_pillar_attention": _attention_config().to_dict(),
                    "range_aware_radar": {
                        "enabled": True,
                        "output_channels": 8,
                    },
                    "coarse_reconstruction": {},
                }
            )

    def test_only_sparse_tokens_are_attended_and_shapes_are_preserved(self):
        module = RadarPillarAttention(6, _attention_config()).eval()
        features = torch.randn(7, 6)
        coordinates = torch.tensor(
            [
                [0, 2, 3],
                [0, 4, 5],
                [0, 6, 7],
                [1, 1, 2],
                [1, 3, 4],
                [1, 5, 6],
                [1, 7, 8],
            ],
            dtype=torch.long,
        )
        original_coordinates = coordinates.clone()
        with torch.no_grad():
            output, debug = module(
                features,
                coordinates,
                2,
                return_attention_weights=True,
            )
        self.assertEqual(output.shape, features.shape)
        self.assertTrue(torch.equal(coordinates, original_coordinates))
        self.assertEqual(debug["token_counts"].tolist(), [3, 4])
        weights = debug["attention_weights"]
        self.assertEqual(weights[0][0].shape, (2, 3, 3))
        self.assertEqual(weights[1][0].shape, (2, 4, 4))
        self.assertLess(output.shape[0], 320 * 320)

    def test_batch_samples_do_not_attend_to_each_other(self):
        torch.manual_seed(5)
        module = RadarPillarAttention(6, _attention_config()).eval()
        first = torch.randn(3, 6)
        second = torch.randn(4, 6)
        together = torch.cat((first, second), dim=0)
        coordinates = torch.tensor(
            [[0, 1, 1], [0, 2, 2], [0, 3, 3]]
            + [[1, 4, 4], [1, 5, 5], [1, 6, 6], [1, 7, 7]],
            dtype=torch.long,
        )
        with torch.no_grad():
            batched, _ = module(together, coordinates, 2)
            first_only, _ = module(first, coordinates[:3].clone(), 1)
            second_coordinates = coordinates[3:].clone()
            second_coordinates[:, 0] = 0
            second_only, _ = module(second, second_coordinates, 1)
        self.assertTrue(torch.allclose(batched[:3], first_only, atol=1e-6))
        self.assertTrue(torch.allclose(batched[3:], second_only, atol=1e-6))

    def test_gradients_reach_all_attention_components(self):
        module = RadarPillarAttention(6, _attention_config()).train()
        features = torch.randn(5, 6, requires_grad=True)
        coordinates = torch.tensor(
            [[0, index, index] for index in range(5)], dtype=torch.long
        )
        output, _ = module(features, coordinates, 1)
        output.square().mean().backward()
        parameters = (
            module.input_projection.weight,
            module.blocks[0].self_attention.in_proj_weight,
            module.blocks[0].self_attention.out_proj.weight,
            module.blocks[0].feed_forward[0].weight,
            module.output_projection.weight,
        )
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_mixed_precision_attention_keeps_sparse_shape(self):
        module = RadarPillarAttention(6, _attention_config()).train()
        features = torch.randn(5, 6, requires_grad=True)
        coordinates = torch.tensor(
            [[0, index, index] for index in range(5)], dtype=torch.long
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, _ = module(features, coordinates, 1)
            loss = output.square().mean()
        loss.backward()
        self.assertEqual(output.shape, features.shape)
        self.assertIsNotNone(features.grad)

    def test_scatter_active_mask_and_final_output_invariants(self):
        pointpillars = {
            "enabled": True,
            "output_channels": 8,
            "max_points_per_pillar": 8,
            "max_pillars": 32,
        }
        model_config, _loss, _selector = build_configs(
            {
                "model": {"backbone": "hrnet"},
                "hrnet": {
                    "base_channels": 2,
                    "num_stages": 2,
                    "blocks_per_stage": 1,
                    "residual_blocks_per_branch": 1,
                },
                "pointpillars": pointpillars,
                "radar_pillar_attention": _attention_config().to_dict(),
                "coarse_reconstruction": {},
            }
        )
        geometry = BEVGridGeometry(0, 64, -32, 32)
        model = CoarseReconstructionModel(
            model_config, grid_geometry=geometry
        ).train()
        faulty = torch.rand(1, 3, 320, 320)
        radar_bev = torch.zeros(1, 4, 320, 320)
        repair = torch.zeros(1, 1, 320, 320)
        repair[:, :, 250:270, 150:170] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 245:275, 145:175] = 1
        halo *= 1 - repair
        lidar_points = (
            torch.tensor([[10.0, 0.0, 0.0, 0.5], [20.0, 1.0, 1.0, 0.7]]),
        )
        radar_points = (
            torch.tensor(
                [
                    [10.0, 0.0, 0.0, 0.8, 0.1],
                    [20.0, 1.0, 1.0, 0.7, -0.2],
                    [30.0, -2.0, 0.5, 0.9, 0.3],
                ]
            ),
        )
        outputs = model(
            faulty,
            radar_bev,
            repair,
            halo,
            halo,
            faulty_lidar_points=lidar_points,
            radar_points=radar_points,
        )
        radar_pillars = outputs["radar_pillar_bev"]
        occupied = radar_pillars.abs().sum(dim=1) > 0
        self.assertEqual(int(occupied.sum()), 3)
        self.assertEqual(
            int(outputs["radar_pillar_attention_debug"]["token_counts"][0]),
            3,
        )
        outside_active = 1 - outputs["active_mask"]
        self.assertEqual(
            float(
                (outputs["local_radar_active"] * outside_active)
                .detach()
                .abs()
                .max()
            ),
            0.0,
        )
        outside_repair = 1 - repair
        self.assertTrue(
            torch.equal(
                outputs["coarse_lidar_bev"] * outside_repair,
                faulty * outside_repair,
            )
        )
        outputs["replacement_raw"].square().mean().backward()
        self.assertIsNotNone(
            model.radar_pillar_encoder.feature_net.linear.weight.grad
        )
        self.assertIsNotNone(
            model.radar_pillar_attention.blocks[
                0
            ].self_attention.in_proj_weight.grad
        )


if __name__ == "__main__":
    unittest.main()
