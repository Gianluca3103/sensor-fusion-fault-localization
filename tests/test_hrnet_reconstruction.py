import unittest
from unittest.mock import patch

import torch

from models.two_stage_reconstruction_head import (
    BEVGridGeometry,
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    HRNetBackbone,
    HRNetConfig,
    PointPillarsConfig,
    build_configs,
    coarse_reconstruction_range_metrics,
)


def _backbone(base_channels=2):
    return HRNetBackbone(
        6,
        HRNetConfig(
            base_channels=base_channels,
            num_stages=4,
            blocks_per_stage=2,
            residual_blocks_per_branch=1,
        ),
    )


class HRNetBackboneTests(unittest.TestCase):
    def test_config_builds_single_hrnet_architecture(self):
        config, _loss, _selector = build_configs(
            {
                "hrnet": {
                    "base_channels": 16,
                    "num_stages": 4,
                    "blocks_per_stage": 2,
                    "residual_blocks_per_branch": 2,
                    "dropout": 0.1,
                },
                "pointpillars": {"enabled": True},
                "coarse_reconstruction": {},
            }
        )
        self.assertEqual(config.hrnet.branch_channels, (16, 32, 64, 128))
        self.assertEqual(config.hrnet.dropout, 0.1)
        self.assertEqual(config.local_input_channels, 130)

    def test_serialized_hrnet_config_round_trip(self):
        original = CoarseReconstructionConfig(
            hrnet=HRNetConfig(
                base_channels=4,
                num_stages=2,
                blocks_per_stage=1,
                residual_blocks_per_branch=1,
            )
        )
        config = CoarseReconstructionConfig.from_dict(original.to_dict())
        self.assertEqual(config.hrnet.branch_channels, (4, 8))
        self.assertFalse(config.pointpillars.enabled)

    def test_full_resolution_and_multiresolution_shapes(self):
        model = _backbone().eval()
        with torch.no_grad():
            output, debug = model(torch.randn(1, 6, 320, 320))
        expected = {
            "hrnet_stage_1_branch_0": (320, 320),
            "hrnet_stage_2_branch_0": (320, 320),
            "hrnet_stage_2_branch_1": (160, 160),
            "hrnet_stage_3_branch_0": (320, 320),
            "hrnet_stage_3_branch_1": (160, 160),
            "hrnet_stage_3_branch_2": (80, 80),
            "hrnet_stage_4_branch_0": (320, 320),
            "hrnet_stage_4_branch_1": (160, 160),
            "hrnet_stage_4_branch_2": (80, 80),
            "hrnet_stage_4_branch_3": (40, 40),
        }
        for name, size in expected.items():
            self.assertEqual(tuple(debug[name].shape[-2:]), size, name)
        self.assertEqual(output.shape, (1, 32, 320, 320))

    def test_odd_shapes_use_explicit_fusion_targets(self):
        model = _backbone().eval()
        with torch.no_grad():
            output, debug = model(torch.randn(1, 6, 65, 67))
        self.assertEqual(debug["hrnet_stage_4_branch_0"].shape[-2:], (65, 67))
        self.assertEqual(debug["hrnet_stage_4_branch_1"].shape[-2:], (33, 34))
        self.assertEqual(debug["hrnet_stage_4_branch_2"].shape[-2:], (17, 17))
        self.assertEqual(debug["hrnet_stage_4_branch_3"].shape[-2:], (9, 9))
        self.assertEqual(output.shape[-2:], (65, 67))

    def test_bidirectional_fusion_receives_gradients(self):
        model = _backbone().train()
        output, debug = model(torch.randn(1, 6, 32, 32))
        low = debug["hrnet_stage_4_branch_3"]
        low.retain_grad()
        output.square().mean().backward()
        self.assertIsNotNone(low.grad)
        self.assertGreater(float(low.grad.abs().sum()), 0.0)

    def test_no_pooling_layers(self):
        forbidden = (torch.nn.MaxPool2d, torch.nn.AvgPool2d)
        self.assertFalse(any(isinstance(m, forbidden) for m in _backbone().modules()))


class HRNetIntegrationTests(unittest.TestCase):
    def _direct_config(self, *, halo=True):
        config, _loss, _selector = build_configs(
            {
                "hrnet": {
                    "base_channels": 2,
                    "blocks_per_stage": 1,
                    "residual_blocks_per_branch": 1,
                },
                "pointpillars": {"enabled": False},
                "masks": {
                    "use_healthy_context_mask": True,
                    "use_halo_context": halo,
                },
                "coarse_reconstruction": {},
            }
        )
        return config

    def test_direct_bev_mode_preserves_diffusion_compatibility(self):
        config = self._direct_config()
        model = CoarseReconstructionModel(config).eval()
        faulty = torch.rand(1, 3, 32, 32)
        radar = torch.rand(1, 4, 32, 32)
        repair = torch.zeros(1, 1, 32, 32)
        repair[:, :, 8:24, 8:24] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 6:26, 6:26] = 1
        halo *= 1 - repair
        with torch.no_grad():
            outputs = model(faulty, radar, repair, halo, halo)
        self.assertEqual(outputs["local_input"].shape, (1, 9, 32, 32))
        self.assertEqual(outputs["coarse_lidar_bev"].shape, faulty.shape)

    def test_disabling_halo_removes_halo_context(self):
        model = CoarseReconstructionModel(self._direct_config(halo=False)).eval()
        faulty = torch.rand(1, 3, 32, 32)
        radar = torch.rand(1, 4, 32, 32)
        repair = torch.zeros(1, 1, 32, 32)
        repair[:, :, 10:20, 10:20] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 6:24, 6:24] = 1
        halo *= 1 - repair
        with torch.no_grad():
            outputs = model(faulty, radar, repair, halo, halo)
        self.assertEqual(int(outputs["halo_mask"].count_nonzero()), 0)
        self.assertTrue(torch.equal(outputs["active_mask"], repair))
        self.assertEqual(int(outputs["local_context_mask"].count_nonzero()), 0)

    def test_pointpillars_erasure_heads_and_outside_invariant(self):
        pointpillars = PointPillarsConfig(enabled=True, output_channels=2)
        config = CoarseReconstructionConfig(
            lidar_channels=2,
            radar_channels=2,
            pointpillars=pointpillars,
            hrnet=HRNetConfig(
                base_channels=2,
                blocks_per_stage=1,
                residual_blocks_per_branch=1,
            ),
        )
        model = CoarseReconstructionModel(
            config,
            grid_geometry=BEVGridGeometry(0.0, 64.0, -32.0, 32.0),
        ).train()
        faulty = torch.rand(1, 3, 32, 32)
        radar = torch.rand(1, 4, 32, 32)
        lidar_features = torch.randn(1, 2, 32, 32)
        radar_features = torch.randn(1, 2, 32, 32)
        repair = torch.zeros(1, 1, 32, 32)
        repair[:, :, 8:24, 9:23] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 6:26, 7:25] = 1
        halo *= 1 - repair
        with patch.object(
            model,
            "_sensor_features",
            return_value=(lidar_features, radar_features, {}, {}),
        ):
            outputs = model(faulty, radar, repair, halo, halo)
        self.assertEqual(outputs["local_input"].shape, (1, 6, 32, 32))
        outside = 1.0 - repair
        self.assertTrue(
            torch.equal(outputs["coarse_lidar_bev"] * outside, faulty * outside)
        )
        outputs["replacement_raw"].square().mean().backward()
        gradient = model.replacement_head.head.weight.grad
        self.assertIsNotNone(gradient)
        for channel in range(3):
            self.assertGreater(float(gradient[channel].abs().sum()), 0.0)

    def test_range_metrics_report_required_bins(self):
        mask = torch.ones(1, 1, 8, 8)
        occupancy = torch.ones(1, 1, 8, 8)
        metrics = coarse_reconstruction_range_metrics(
            {
                "reconstruction_mask": mask,
                "occupancy_logits": torch.full((1, 1, 8, 8), 10.0),
            },
            torch.cat((occupancy, occupancy, occupancy), dim=1),
            x_range=(0.0, 64.0),
            y_range=(-32.0, 32.0),
        )
        for name in ("0_15m", "15_30m", "30_45m", "45_60m", "over_60m"):
            self.assertIn(f"range_{name}/occupancy_iou", metrics)


if __name__ == "__main__":
    unittest.main()
