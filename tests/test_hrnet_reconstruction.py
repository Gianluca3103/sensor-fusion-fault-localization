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
from models.two_stage_reconstruction_head.reconstruction_crop import (
    ReconstructionCropExtractor,
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
    def _direct_config(self, *, halo=True, minimum_context_crop_size=0):
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
                "coarse_reconstruction": {
                    "minimum_context_crop_size": minimum_context_crop_size,
                    "context_crop_pad_multiple": 8,
                },
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

    def test_tiny_interior_region_expands_to_80_without_changing_masks(self):
        extractor = ReconstructionCropExtractor(pad_multiple=8, minimum_size=80)
        repair = torch.zeros(1, 1, 320, 320)
        halo = torch.zeros_like(repair)
        repair[:, :, 150:154, 160:166] = 1
        halo[:, :, 146:158, 156:170] = 1
        original_repair = repair.clone()
        original_halo = halo.clone()
        crops = extractor.extract({"repair": repair, "halo": halo}, repair, halo)
        self.assertEqual(crops.crop_heights.tolist(), [80])
        self.assertEqual(crops.crop_widths.tolist(), [80])
        self.assertEqual(tuple(crops.valid_mask.shape[-2:]), (80, 80))
        self.assertTrue(torch.equal(repair, original_repair))
        self.assertTrue(torch.equal(halo, original_halo))

    def test_edge_regions_shift_inward_and_preserve_minimum(self):
        extractor = ReconstructionCropExtractor(pad_multiple=8, minimum_size=80)
        for row_slice, column_slice, expected in (
            (slice(0, 4), slice(0, 5), (0, 80, 0, 80)),
            (slice(316, 320), slice(315, 320), (240, 320, 240, 320)),
        ):
            repair = torch.zeros(1, 1, 320, 320)
            repair[:, :, row_slice, column_slice] = 1
            crops = extractor.extract({"repair": repair}, repair, torch.zeros_like(repair))
            self.assertEqual(tuple(crops.boxes[0].tolist()), expected)
            self.assertEqual(crops.crop_heights.tolist(), [80])
            self.assertEqual(crops.crop_widths.tolist(), [80])

    def test_technical_padding_is_invalid_and_stride_eight_compatible(self):
        extractor = ReconstructionCropExtractor(pad_multiple=8, minimum_size=81)
        repair = torch.zeros(1, 1, 320, 320)
        repair[:, :, 100:105, 100:105] = 1
        crops = extractor.extract({"repair": repair}, repair, torch.zeros_like(repair))
        self.assertEqual(tuple(crops.valid_mask.shape[-2:]), (88, 88))
        self.assertEqual(int(crops.valid_mask.sum()), 81 * 81)
        self.assertEqual(int(crops.valid_mask[:, :, 81:, :].sum()), 0)
        self.assertEqual(int(crops.valid_mask[:, :, :, 81:].sum()), 0)

    def test_modalities_share_identical_crop_coordinates(self):
        extractor = ReconstructionCropExtractor(pad_multiple=8, minimum_size=80)
        coordinates = torch.arange(320 * 320).reshape(1, 1, 320, 320).float()
        repair = torch.zeros(1, 1, 320, 320)
        repair[:, :, 20:30, 40:50] = 1
        crops = extractor.extract(
            {"lidar": coordinates, "radar": coordinates + 7},
            repair,
            torch.zeros_like(repair),
        )
        valid = crops.valid_mask.bool()
        self.assertTrue(torch.equal(
            crops.tensors["radar"][valid] - crops.tensors["lidar"][valid],
            torch.full_like(crops.tensors["lidar"][valid], 7),
        ))

    def test_batch_uses_shared_technical_shape_with_per_sample_validity(self):
        extractor = ReconstructionCropExtractor(pad_multiple=8, minimum_size=80)
        repair = torch.zeros(2, 1, 320, 320)
        repair[0, :, 10:14, 10:14] = 1
        repair[1, :, 100:191, 120:217] = 1
        crops = extractor.extract(
            {"repair": repair}, repair, torch.zeros_like(repair)
        )
        self.assertEqual(crops.crop_heights.tolist(), [80, 91])
        self.assertEqual(crops.crop_widths.tolist(), [80, 97])
        self.assertEqual(tuple(crops.valid_mask.shape), (2, 1, 96, 104))
        self.assertEqual(int(crops.valid_mask[0].sum()), 80 * 80)
        self.assertEqual(int(crops.valid_mask[1].sum()), 91 * 97)

    def test_contextual_coarse_forward_has_stride8_deepest_and_outside_invariant(self):
        model = CoarseReconstructionModel(
            self._direct_config(minimum_context_crop_size=80)
        ).train()
        faulty = torch.rand(1, 3, 320, 320)
        radar = torch.rand(1, 4, 320, 320)
        repair = torch.zeros(1, 1, 320, 320)
        repair[:, :, 150:154, 150:154] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 145:160, 145:160] = 1
        halo *= 1 - repair
        outputs = model(faulty, radar, repair, halo, halo)
        self.assertEqual(tuple(outputs["local_input"].shape[-2:]), (80, 80))
        self.assertEqual(
            tuple(outputs["hrnet_stage_4_branch_3"].shape[-2:]), (10, 10)
        )
        self.assertTrue(torch.equal(
            outputs["coarse_lidar_bev"] * (1 - repair), faulty * (1 - repair)
        ))
        outputs["coarse_lidar_bev"].square().mean().backward()
        self.assertIsNotNone(model.replacement_head.head.weight.grad)


if __name__ == "__main__":
    unittest.main()
