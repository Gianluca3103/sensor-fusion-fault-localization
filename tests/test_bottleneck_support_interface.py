"""The new deployed Stage-I boundary has two outputs and no LiDAR input."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.radar_lidar_stage1 import pfe_spconv
from models.radar_lidar_stage1.bottleneck_support import (
    BottleneckSupportOutput, BottleneckSupportStage1)
from models.radar_lidar_stage1.config import Stage1Config, VoxelGrid
from models.radar_lidar_stage1.sparse import SparseConv3d, SparseSites
from models.radar_lidar_stage1.train import load_radar_only
from models.radar_lidar_stage2.candidate_domain import make_candidates
from models.radar_lidar_stage2.candidate_domain import CandidateDomain
from models.radar_lidar_stage2.config import Stage2Config
from models.radar_lidar_stage2.stage1_conditioning import Stage1Conditioning
from models.radar_lidar_stage2_diffusion.cache import collate_cached, prepare_frame
from models.radar_lidar_stage2_diffusion.config import DiffusionConfig
from scripts.visualize_stage1_confidence_cloud import load_encoder


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


class BottleneckInterfaceTests(unittest.TestCase):
    def test_dropout_regularizes_training_heads_but_not_inference(self):
        grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (1., 1., 1.))
        config = replace(Stage1Config(grid=grid, channels=(8, 16, 32, 64)),
                         encoder_backend="pfe_spconv", growth_scales=(),
                         bottleneck_cross_attention_only=True,
                         bottleneck_support_only=True,
                         corr_scale_weights=(0., 0., 0., 1.),
                         confidence_weight=0., attention_dim=16,
                         bottleneck_dropout=0.1)
        xyz = torch.tensor([[1., 1., 1.], [2., 3., 3.], [3., 5., 6.],
                            [6., 3., 9.], [7., 7., 12.], [9., 11., 4.],
                            [12., 4., 6.], [14., 14., 14.]])
        radar = torch.cat((xyz, torch.ones(8, 4)), dim=1)[None]
        valid = torch.ones((1, 8), dtype=torch.bool)
        reference = SimpleNamespace(SubMConv3d=ReferenceConv,
                                    SparseConv3d=ReferenceConv,
                                    SparseConvTensor=ReferenceTensor)
        with patch.object(pfe_spconv, "spconv", reference):
            model = BottleneckSupportStage1(config)
            self.assertEqual(model.radar_only.dropout.p, 0.1)
            self.assertEqual(model.correspondence.dropout.p, 0.1)
            model.train()
            torch.manual_seed(1)
            first = model.forward_radar(radar, valid)
            torch.manual_seed(2)
            second = model.forward_radar(radar, valid)
            self.assertFalse(torch.equal(first.confidence_region.features,
                                         second.confidence_region.features))
            model.eval()
            with torch.no_grad():
                first = model.forward_radar(radar, valid)
                second = model.forward_radar(radar, valid)
            torch.testing.assert_close(first.confidence_region.features,
                                       second.confidence_region.features)
            model.train()
            lidar = torch.cat((xyz + .1, torch.ones(8, 1)), dim=1)[None]
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4,
                                           weight_decay=5e-4)
            losses, _ = model.forward_train(radar, valid, lidar, valid)
            self.assertTrue(torch.isfinite(losses["loss/total"]))
            losses["loss/total"].backward()
            optimizer.step()

    def test_training_inference_and_stage2_boundary(self):
        grid = VoxelGrid((0., 0., 0.), (16., 16., 16.), (1., 1., 1.))
        config = replace(Stage1Config(grid=grid, channels=(8, 16, 32, 64)),
                         encoder_backend="pfe_spconv", growth_scales=(),
                         bottleneck_cross_attention_only=True,
                         bottleneck_support_only=True,
                         corr_scale_weights=(0., 0., 0., 1.),
                         confidence_weight=0., attention_dim=16)
        xyz = torch.tensor([[1., 1., 1.], [2., 3., 3.], [3., 5., 6.],
                            [6., 3., 9.], [7., 7., 12.], [9., 11., 4.],
                            [12., 4., 6.], [14., 14., 14.]])
        radar = torch.cat((xyz, torch.ones(8, 4)), dim=1)[None]
        lidar = torch.cat((xyz + .1, torch.ones(8, 1)), dim=1)[None]
        valid = torch.ones((1, 8), dtype=torch.bool)
        reference = SimpleNamespace(SubMConv3d=ReferenceConv,
                                    SparseConv3d=ReferenceConv,
                                    SparseConvTensor=ReferenceTensor)
        with patch.object(pfe_spconv, "spconv", reference):
            model = BottleneckSupportStage1(config).eval()
            output = model.forward_radar(radar, valid)
            self.assertIsInstance(output, BottleneckSupportOutput)
            self.assertEqual(set(vars(output)), {"bottleneck", "confidence_region"})
            self.assertEqual(output.bottleneck.features.shape[1], config.attention_dim)
            self.assertEqual(output.bottleneck.stride, 8)
            self.assertEqual(output.confidence_region.stride, 2)
            self.assertTrue(bool(((output.confidence_region.features >= 0) &
                                  (output.confidence_region.features <= 1)).all()))
            losses, diagnostics = model.forward_train(radar, valid, lidar, valid)
            self.assertTrue(torch.isfinite(losses["loss/total"]))
            self.assertEqual(len(diagnostics["occupied"]), len(output.confidence_region.coords))
            losses["loss/total"].backward()
            self.assertIsNotNone(model.radar_only.support_head[-1].weight.grad)
            self.assertGreater(float(model.radar_only.support_head[-1].weight.grad.abs().sum()), 0.)

            domain = make_candidates(output, grid, confidence_threshold=0.0)
            self.assertEqual(domain.counts["seed_source"], "bottleneck_support_field")
            conditioner = Stage1Conditioning(config.conditioning_channels, Stage2Config()).eval()
            conditioned = conditioner(output, domain)
            self.assertEqual(conditioned.shape, (len(domain.coordinates), Stage2Config().channels[0]))

            batch = {"frame_id": ["00000"], "clean_lidar": lidar,
                     "clean_lidar_valid": valid,
                     "faulty_lidar": lidar[:, :0],
                     "faulty_lidar_valid": valid[:, :0],
                     "fault_region": [{"full_scan": True}]}
            cached = prepare_frame(output, grid, batch, DiffusionConfig())
            self.assertEqual(int(cached["source_count"]), 1)
            merged = collate_cached([cached], grid)
            self.assertEqual(len(merged["cache"]["sources"]), 1)
            self.assertEqual(len(merged["cache"]["indices"]), 1)
            data = merged["cache"]
            cached_domain = CandidateDomain(data["coordinates"], data["confidence"],
                                            grid, data["counts"])
            direct = conditioner(output, cached_domain)
            from_cache = conditioner.forward_cached(
                cached_domain, data["sources"], data["indices"], None,
                source_centers=data["source_centers"])
            torch.testing.assert_close(direct, from_cache, atol=1e-5, rtol=1e-5)

            with TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "radar_only.pth"
                torch.save(model.radar_only_state_dict(), checkpoint)
                restored = load_radar_only(checkpoint)
                torch.testing.assert_close(restored(radar, valid).bottleneck.features,
                                           output.bottleneck.features)
                viewer_model, _ = load_encoder(checkpoint, "cpu")
                torch.testing.assert_close(viewer_model(radar, valid).confidence_region.features,
                                           output.confidence_region.features)


if __name__ == "__main__":
    unittest.main()
