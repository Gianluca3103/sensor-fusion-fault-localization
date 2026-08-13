import unittest

import torch

from models.reconstruction_head import (
    CoarseReconstructionModel,
    RangeAwareRadarAggregation,
    RangeAwareRadarConfig,
    build_configs,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "output_channels": 16,
        "hidden_channels": 8,
        "min_radius_m": 0.3,
        "max_radius_m": 1.0,
        "range_min_m": 1.0,
        "range_max_m": 5.0,
        "x_min_m": 0.0,
        "x_max_m": 8.0,
        "y_min_m": -4.0,
        "y_max_m": 4.0,
        "spatial_chunk_size": 400,
    }
    values.update(overrides)
    return RangeAwareRadarConfig(**values)


class RangeAwareRadarTests(unittest.TestCase):
    def test_config_changes_only_prefusion_radar_channel_count(self):
        model_config, _loss, _selector = build_configs(
            {
                "model": {"backbone": "hrnet"},
                "hrnet": {
                    "base_channels": 2,
                    "blocks_per_stage": 1,
                    "residual_blocks_per_branch": 1,
                },
                "range_aware_radar": _config().to_dict(),
                "pointpillars": {"enabled": False},
                "coarse_reconstruction": {},
            }
        )
        self.assertEqual(model_config.radar_channels, 4)
        self.assertEqual(model_config.local_input_channels, 21)

    def test_output_shape_and_resolution_are_preserved(self):
        module = RangeAwareRadarAggregation(
            4,
            _config(
                x_max_m=64.0,
                y_min_m=-32.0,
                y_max_m=32.0,
                range_min_m=10.0,
                range_max_m=60.0,
                spatial_chunk_size=1024,
            ),
        ).eval()
        radar = torch.rand(1, 4, 320, 320)
        radar[:, 0] = (radar[:, 0] > 0.8).to(radar.dtype)
        active = torch.ones(1, 1, 320, 320)
        with torch.no_grad():
            output, debug = module(radar, active)
        self.assertEqual(output.shape, (1, 16, 320, 320))
        self.assertEqual(debug["valid_neighbor_count"].shape, (1, 1, 320, 320))

    def test_radius_is_monotonic_and_reaches_configured_limits(self):
        module = RangeAwareRadarAggregation(4, _config())
        radius = module.radius_map(
            40, 40, device=torch.device("cpu"), dtype=torch.float32
        )[0, 0]
        _, _, ranges, _, _ = module._geometry(
            40, 40, torch.device("cpu"), torch.float32
        )
        order = torch.argsort(ranges.flatten())
        sorted_radius = radius.flatten()[order]
        self.assertTrue(torch.all(sorted_radius[1:] >= sorted_radius[:-1]))
        self.assertAlmostEqual(float(radius.min()), 0.3, places=5)
        self.assertAlmostEqual(float(radius.max()), 1.0, places=5)

    def test_adaptive_mask_attention_and_neighbor_counts(self):
        module = RangeAwareRadarAggregation(4, _config()).eval()
        radar = torch.ones(1, 4, 40, 40)
        active = torch.ones(1, 1, 40, 40)
        with torch.no_grad():
            _, debug = module(radar, active, return_attention=True)
        weights = debug["attention_weights"]
        counts = debug["valid_neighbor_count"]
        radius = debug["radius_m"][0, 0]

        near_row, near_col = 35, 20
        far_row, far_col = 14, 20
        self.assertLess(
            int(counts[0, 0, near_row, near_col]),
            int(counts[0, 0, far_row, far_col]),
        )

        offsets = torch.arange(-5, 6, dtype=torch.float32) * 0.2
        dx, dy = torch.meshgrid(offsets, offsets, indexing="ij")
        distances = torch.sqrt(dx.square() + dy.square())
        for row, col in ((near_row, near_col), (far_row, far_col)):
            location_weights = weights[0, row, col]
            outside = distances > radius[row, col] + 1.0e-6
            self.assertEqual(float(location_weights[outside].abs().max()), 0.0)
            self.assertAlmostEqual(float(location_weights.sum()), 1.0, places=5)

    def test_empty_neighborhood_returns_zero_and_active_mask_does_not_leak(self):
        module = RangeAwareRadarAggregation(4, _config()).eval()
        radar = torch.zeros(1, 4, 40, 40)
        active = torch.zeros(1, 1, 40, 40)
        active[:, :, 15:25, 15:25] = 1
        with torch.no_grad():
            empty, debug = module(radar, active, return_attention=True)
        self.assertEqual(float(empty.abs().max()), 0.0)
        self.assertEqual(float(debug["attention_weights"].abs().max()), 0.0)

        radar.fill_(1.0)
        with torch.no_grad():
            output, _ = module(radar, active)
        self.assertEqual(float((output * (1 - active)).abs().max()), 0.0)

    def test_gradients_reach_neighbor_weight_and_center_paths(self):
        module = RangeAwareRadarAggregation(4, _config()).train()
        radar = torch.rand(1, 4, 20, 20, requires_grad=True)
        radar.data[:, 0] = 1.0
        active = torch.ones(1, 1, 20, 20)
        output, _ = module(radar, active)
        output.square().mean().backward()
        for component in (
            module.center_encoder,
            module.neighbor_encoder,
            module.weight_network,
            module.fusion,
        ):
            gradients = [
                parameter.grad
                for parameter in component.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.0)

    def test_model_preserves_existing_active_mask_policy(self):
        model_config, _loss, _selector = build_configs(
            {
                "model": {"backbone": "hrnet"},
                "hrnet": {
                    "base_channels": 2,
                    "blocks_per_stage": 1,
                    "residual_blocks_per_branch": 1,
                },
                "range_aware_radar": _config().to_dict(),
                "pointpillars": {"enabled": False},
                "coarse_reconstruction": {},
            }
        )
        model = CoarseReconstructionModel(model_config).eval()
        faulty = torch.rand(1, 3, 40, 40)
        radar = torch.rand(1, 4, 40, 40)
        radar[:, 0] = 1.0
        repair = torch.zeros(1, 1, 40, 40)
        repair[:, :, 15:25, 15:25] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 12:28, 12:28] = 1
        halo *= 1 - repair
        with torch.no_grad():
            outputs = model(faulty, radar, repair, halo, halo)
        self.assertEqual(outputs["local_input"].shape, (1, 21, 40, 40))
        outside = 1 - outputs["active_mask"]
        self.assertEqual(
            float((outputs["local_radar_context"] * outside).abs().max()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
