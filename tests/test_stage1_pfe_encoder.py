"""Checks for the new Stage-I PFE backbone and checkpoint boundary."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.radar_lidar_stage1.config import Stage1Config, VoxelGrid
from models.radar_lidar_stage1.evaluate import evaluate_pair
from models.radar_lidar_stage1.model import RadarLidarStage1
from models.radar_lidar_stage1.metrics import Stage1MetricAccumulator
from models.radar_lidar_stage1 import pfe_spconv
from models.radar_lidar_stage1.pfe_spconv import spconv
from models.radar_lidar_stage1.sparse import SparseConv3d, SparseSites
from models.radar_lidar_stage1.train import config_from_dict


ROOT = Path(__file__).resolve().parents[1]


class PFEEncoderTests(unittest.TestCase):
    def test_surface_config_uses_fine_grid_and_pfe(self):
        config = config_from_dict(json.loads(
            (ROOT / "configs/radar_lidar_stage1_surface.json").read_text()))
        self.assertEqual(config.encoder_backend, "pfe_spconv")
        self.assertEqual(config.grid.size_xyz, (0.2, 0.2, 0.25))
        self.assertEqual(config.growth_scales, ())
        self.assertTrue(config.bottleneck_cross_attention_only)
        self.assertEqual(config.corr_scale_weights, (0., 0., 0., 1.))

    def test_old_checkpoints_keep_native_backend(self):
        old = Stage1Config().as_dict()
        old.pop("encoder_backend")
        self.assertEqual(config_from_dict(old).encoder_backend, "native")
        self.assertFalse(config_from_dict(old).bottleneck_cross_attention_only)

    def test_cross_modal_comparison_only_uses_bottleneck(self):
        grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (1., 1., 1.))
        config = replace(Stage1Config(grid=grid, channels=(8, 16, 32, 64)),
                         growth_scales=(), bottleneck_cross_attention_only=True,
                         corr_scale_weights=(0., 0., 0., 1.))
        model = RadarLidarStage1(config).eval()
        xyz = torch.tensor([[1., 1., 1.], [2., 3., 3.], [3., 5., 6.],
                            [6., 3., 9.], [7., 7., 12.], [9., 11., 4.],
                            [12., 4., 6.], [14., 14., 14.]])
        radar = torch.cat((xyz, torch.ones(8, 4)), dim=1)[None]
        lidar = torch.cat((xyz + .1, torch.ones(8, 1)), dim=1)[None]
        valid = torch.ones((1, 8), dtype=torch.bool)
        losses, diagnostics = model.forward_train(radar, valid, lidar, valid,
                                                  return_intermediates=True)
        self.assertEqual(list(diagnostics["intermediates"]["attention"]), ["s4"])
        self.assertTrue(all(isinstance(model.correspondence[i], torch.nn.Identity)
                            for i in range(3)))
        self.assertTrue(all(float(losses[f"loss/corr_s{i}"].detach()) == 0.
                            for i in range(1, 4)))
        metrics = Stage1MetricAccumulator(config)
        metrics.update(model, radar, valid, lidar, valid,
                       intermediates=diagnostics["intermediates"])
        report = metrics.finish()
        self.assertFalse(report["scales"]["s1"]["correspondence_enabled"])
        self.assertTrue(report["scales"]["s4"]["correspondence_enabled"])
        self.assertEqual(report["correspondence"]["query_count"],
                         report["scales"]["s4"]["query_count"])
        paired = evaluate_pair(model, radar, valid, lidar, valid)
        self.assertIsNone(paired["s1"]["recall_at_1"])
        self.assertEqual(paired["s4"]["matched_count"],
                         report["scales"]["s4"]["valid_corr_query_count"])

    def test_pfe_voxel_features_keep_sensor_attributes(self):
        grid = VoxelGrid((0., 0., 0.), (8., 8., 8.), (1., 1., 1.))
        valid = torch.ones((1, 2), dtype=torch.bool)
        for modality, width in (("radar", 7), ("lidar", 4)):
            encoder = pfe_spconv.PFEPointVoxelEncoder(grid, modality).eval()
            points = torch.zeros((1, 2, width))
            points[0, :, :3] = torch.tensor([[1., 1., 1.], [3., 3., 3.]])
            changed = points.clone()
            changed[0, 0, 3:] = 5.
            with torch.no_grad():
                original, _ = encoder(points, valid)
                updated, _ = encoder(changed, valid)
            self.assertTrue(torch.equal(original.coords, updated.coords))
            self.assertGreater(float((original.features - updated.features).abs().sum()), 0.)

    def test_missing_spconv_fails_before_training(self):
        if spconv is not None:
            self.skipTest("spconv is available")
        config = replace(Stage1Config(), encoder_backend="pfe_spconv",
                         growth_scales=())
        with self.assertRaisesRegex(RuntimeError, "requires spconv"):
            RadarLidarStage1(config)

    def test_four_scale_adapter_and_gradients_with_reference_sparse_ops(self):
        """Exercise Stage-I glue even when the external CUDA wheel is absent."""
        class ReferenceTensor:
            def __init__(self, features, indices, spatial_shape, batch_size):
                self.features, self.indices = features, indices
                self.spatial_shape, self.batch_size = spatial_shape, batch_size

            def replace_feature(self, features):
                return ReferenceTensor(features, self.indices, self.spatial_shape,
                                       self.batch_size)

        class ReferenceConv(torch.nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                         padding=0, bias=False, indice_key=None):
                super().__init__()
                self.conv = SparseConv3d(in_channels, out_channels, stride=stride)

            def forward(self, tensor):
                sites = SparseSites(tensor.indices.long(), tensor.features,
                                    tuple(tensor.spatial_shape))
                result = self.conv(sites)
                return ReferenceTensor(result.features, result.coords.int(),
                                       result.shape_zyx, tensor.batch_size)

        grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (1., 1., 1.))
        config = replace(Stage1Config(grid=grid, channels=(8, 16, 32, 64)),
                         encoder_backend="pfe_spconv", growth_scales=())
        xyz = torch.tensor([[1., 1., 1.], [2., 3., 3.], [3., 5., 6.],
                            [6., 3., 9.], [7., 7., 12.], [9., 11., 4.],
                            [12., 4., 6.], [14., 14., 14.]])
        radar = torch.cat((xyz, torch.ones(8, 4)), dim=1)[None].repeat(2, 1, 1)
        lidar = torch.cat((xyz + .1, torch.ones(8, 1)), dim=1)[None].repeat(2, 1, 1)
        valid = torch.ones((2, 8), dtype=torch.bool)
        reference = SimpleNamespace(SubMConv3d=ReferenceConv,
                                    SparseConv3d=ReferenceConv,
                                    SparseConvTensor=ReferenceTensor)
        with patch.object(pfe_spconv, "spconv", reference):
            model = RadarLidarStage1(config).train()
            radar_output = model.forward_radar(radar, valid)
            lidar_levels, _ = model.lidar_teacher(lidar, valid)
            self.assertEqual(list(radar_output.features), ["s1", "s2", "s3", "s4"])
            for index, lidar_level in enumerate(lidar_levels):
                radar_level = radar_output.features[f"s{index + 1}"]
                self.assertEqual(radar_level.stride, 2 ** index)
                self.assertEqual(lidar_level.stride, 2 ** index)
                self.assertEqual(radar_level.features.shape[1], config.channels[index])
                self.assertEqual(lidar_level.features.shape[1], config.channels[index])
            losses, _ = model.forward_train(radar, valid, lidar, valid)
            self.assertTrue(torch.isfinite(losses["loss/total"]))
            losses["loss/total"].backward()
            self.assertGreater(float(model.radar_only.backbone.stem.conv.kernel.grad.abs().sum()), 0.)
            # This tiny scene has no contrastive negatives, so separately
            # check that teacher features remain differentiable.
            sum(level.features.square().mean() for level in lidar_levels).backward()
            self.assertGreater(float(model.lidar_teacher.backbone.stem.conv.kernel.grad.abs().sum()), 0.)

    @unittest.skipUnless(spconv is not None and torch.cuda.is_available(),
                         "requires the actual spconv CUDA backend")
    def test_both_sensor_encoders_emit_four_sparse_scales_with_gradients(self):
        grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (1., 1., 1.))
        config = replace(Stage1Config(grid=grid, channels=(16, 32, 64, 128)),
                         encoder_backend="pfe_spconv", growth_scales=(),
                         surface_proposals_per_site=0)
        model = RadarLidarStage1(config).cuda().train()
        xyz = torch.tensor([[1., 1., 1.], [2., 3., 3.], [3., 5., 6.],
                            [6., 3., 9.], [7., 7., 12.], [9., 11., 4.],
                            [12., 4., 6.], [14., 14., 14.]])
        radar = torch.cat((xyz, torch.ones(8, 4)), dim=1)[None].repeat(2, 1, 1).cuda()
        lidar = torch.cat((xyz + .1, torch.ones(8, 1)), dim=1)[None].repeat(2, 1, 1).cuda()
        valid = torch.ones((2, 8), dtype=torch.bool, device="cuda")
        radar_levels = model.radar_only(radar, valid).features
        lidar_levels, _ = model.lidar_teacher(lidar, valid)
        for index in range(4):
            radar_level = radar_levels[f"s{index + 1}"]
            lidar_level = lidar_levels[index]
            self.assertEqual(radar_level.stride, 2 ** index)
            self.assertEqual(lidar_level.stride, 2 ** index)
            self.assertEqual(radar_level.features.shape[1], config.channels[index])
            self.assertEqual(lidar_level.features.shape[1], config.channels[index])
            self.assertGreater(len(radar_level.coords), 0)
            self.assertGreater(len(lidar_level.coords), 0)
        loss = sum(level.features.square().mean() for level in lidar_levels)
        loss += sum(level.features[:, 0].mean() for level in radar_levels.values())
        loss.backward()
        self.assertGreater(float(model.radar_only.backbone.stem.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.lidar_teacher.backbone.stem.weight.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
