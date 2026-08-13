import unittest
from unittest.mock import patch

import torch

from models.reconstruction_head import (
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
    def test_config_builds_hrnet_without_changing_other_backbones(self):
        config, _loss, _selector = build_configs(
            {
                "model": {"backbone": "hrnet"},
                "hrnet": {
                    "base_channels": 16,
                    "num_stages": 4,
                    "blocks_per_stage": 2,
                    "residual_blocks_per_branch": 2,
                    "dropout": 0.0,
                },
                "pointpillars": {"enabled": True},
                "coarse_reconstruction": {},
            }
        )
        self.assertEqual(config.backbone, "hrnet")
        self.assertEqual(config.hrnet.branch_channels, (16, 32, 64, 128))

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
        self.assertEqual(
            debug["hrnet_final_concatenated"].shape,
            (1, 30, 320, 320),
        )
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

    def test_bidirectional_fusion_and_all_modules_receive_gradients(self):
        model = _backbone().train()
        output, debug = model(torch.randn(1, 6, 32, 32))
        low = debug["hrnet_stage_4_branch_3"]
        low.retain_grad()
        debug["hrnet_stage_1_branch_0"].retain_grad()
        output.square().mean().backward(retain_graph=True)
        self.assertIsNotNone(low.grad)
        self.assertGreater(float(low.grad.abs().sum()), 0.0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

        model.zero_grad(set_to_none=True)
        _, second_debug = model(torch.randn(1, 6, 32, 32))
        high = second_debug["hrnet_stage_1_branch_0"]
        high.retain_grad()
        second_debug["hrnet_stage_4_branch_3"].square().mean().backward()
        self.assertIsNotNone(high.grad)
        self.assertGreater(float(high.grad.abs().sum()), 0.0)

    def test_no_pooling_layers(self):
        model = _backbone()
        forbidden = (torch.nn.MaxPool2d, torch.nn.AvgPool2d)
        self.assertFalse(
            any(isinstance(module, forbidden) for module in model.modules())
        )


class HRNetIntegrationTests(unittest.TestCase):
    def test_three_layer_radar_stem_has_exact_seven_cell_receptive_field(self):
        config, _loss, _selector = build_configs(
            {
                "model": {"backbone": "hrnet"},
                "hrnet": {
                    "base_channels": 2,
                    "blocks_per_stage": 1,
                    "residual_blocks_per_branch": 1,
                    "radar_context_layers": 3,
                    "radar_context_channels": 8,
                },
                "pointpillars": {"enabled": False},
                "coarse_reconstruction": {},
            }
        )
        model = CoarseReconstructionModel(config)
        encoder = model.radar_context_encoder
        convolutions = [
            module
            for module in encoder.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        self.assertEqual(len(convolutions), 3)
        self.assertEqual(encoder.effective_receptive_field_cells, 7)
        for convolution in convolutions:
            self.assertEqual(convolution.kernel_size, (3, 3))
            self.assertEqual(convolution.stride, (1, 1))
            self.assertEqual(convolution.padding, (1, 1))
            self.assertEqual(convolution.dilation, (1, 1))

        for convolution in convolutions:
            torch.nn.init.ones_(convolution.weight)
        radar = torch.zeros(1, 4, 15, 15)
        radar[:, 0, 7, 7] = 1.0
        with torch.no_grad():
            context = encoder(radar)
        support = context.abs().sum(dim=1)[0] > 0
        rows, cols = support.nonzero(as_tuple=True)
        self.assertEqual((int(rows.min()), int(rows.max())), (4, 10))
        self.assertEqual((int(cols.min()), int(cols.max())), (4, 10))
        self.assertEqual(int(support.sum()), 49)

    def test_handcrafted_bev_inputs_do_not_require_pointpillars(self):
        config, _loss, _selector = build_configs(
            {
                "model": {"backbone": "hrnet"},
                "hrnet": {
                    "base_channels": 2,
                    "blocks_per_stage": 1,
                    "residual_blocks_per_branch": 1,
                },
                "pointpillars": {"enabled": False},
                "coarse_reconstruction": {},
            }
        )
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

        self.assertFalse(config.pointpillars.enabled)
        self.assertEqual(config.lidar_channels, 3)
        self.assertEqual(config.radar_channels, 4)
        self.assertEqual(outputs["local_input"].shape, (1, 9, 32, 32))
        self.assertEqual(outputs["coarse_lidar_bev"].shape, faulty.shape)

    def test_input_erasure_output_heads_and_outside_mask_invariant(self):
        pointpillars = PointPillarsConfig(enabled=True, output_channels=2)
        config = CoarseReconstructionConfig(
            backbone="hrnet",
            lidar_channels=2,
            radar_channels=2,
            pointpillars=pointpillars,
            hrnet=HRNetConfig(
                base_channels=2,
                blocks_per_stage=1,
                residual_blocks_per_branch=1,
            ),
        )
        geometry = BEVGridGeometry(0.0, 64.0, -32.0, 32.0)
        model = CoarseReconstructionModel(config, grid_geometry=geometry).train()
        faulty = torch.rand(1, 3, 32, 32)
        radar = torch.rand(1, 4, 32, 32)
        lidar_features = torch.randn(1, 2, 32, 32)
        radar_features = torch.randn(1, 2, 32, 32)
        repair = torch.zeros(1, 1, 32, 32)
        repair[:, :, 8:24, 9:23] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 6:26, 7:25] = 1
        halo *= 1 - repair
        healthy = halo.clone()

        sensor_result = (lidar_features, radar_features, {}, {})
        with patch.object(model, "_sensor_features", return_value=sensor_result):
            outputs = model(faulty, radar, repair, healthy, halo)

        self.assertEqual(outputs["local_input"].shape, (1, 6, 32, 32))
        self.assertEqual(outputs["occupancy_logits"].shape, (1, 1, 32, 32))
        self.assertEqual(outputs["predicted_density"].shape, (1, 1, 32, 32))
        self.assertEqual(outputs["predicted_height"].shape, (1, 1, 32, 32))
        self.assertEqual(
            float((outputs["local_lidar_context"] * repair).abs().max()),
            0.0,
        )
        outside = 1.0 - repair
        self.assertTrue(
            torch.equal(outputs["coarse_lidar_bev"] * outside, faulty * outside)
        )
        outputs["replacement_raw"].square().mean().backward()
        head_gradient = model.replacement_head.head.weight.grad
        self.assertIsNotNone(head_gradient)
        for channel in range(3):
            self.assertGreater(float(head_gradient[channel].abs().sum()), 0.0)

    def test_range_metrics_report_all_required_bins(self):
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
            self.assertIn(f"range_{name}/occupancy_precision", metrics)
            self.assertIn(f"range_{name}/occupancy_recall", metrics)
            self.assertIn(f"range_{name}/occupancy_f1", metrics)
            self.assertIn(f"range_{name}/occupancy_iou", metrics)


if __name__ == "__main__":
    unittest.main()
