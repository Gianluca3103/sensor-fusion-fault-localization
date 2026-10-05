"""Focused checks for the inference boundary and absolute-depth objective."""

import math
import unittest
from unittest.mock import patch

import torch

from models.two_stage_reconstruction_head.cross_modal_encoders import EncoderGrid
from models.two_stage_reconstruction_head.diffusion_process.joint_relation_diffusion import (
    PairedRadarLidarAttention, RadarRelationDiffusion, RadarRelationEncoder,
    denormalize_depth, joint_relation_loss, normalize_depth,
    sample_joint_full_scan,
)
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.ray_depth_attention import (
    CleanLidarRelationshipTeacher,
)
from models.two_stage_reconstruction_head.ray_depth_queries import ray_tile_indices


class JointRelationDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.geometry = RangeGeometry(
            beam_elevations_rad=(-0.1, 0.0, 0.1), azimuth_bins=8,
            min_range_m=0.5, max_range_m=12.0,
            azimuth_span_rad=2 * math.pi)
        self.grid = EncoderGrid(
            minimum_xyz=(0.0, -8.0, -4.0),
            maximum_xyz=(16.0, 8.0, 4.0),
            voxel_size_xyz=(2.0, 2.0, 2.0))
        self.ray = torch.tensor(self.geometry.ray_directions()[1].copy())
        self.rows, self.cols = ray_tile_indices(
            self.geometry, row_start=1, row_stop=2)
        self.radar = torch.tensor([[[*(self.ray[0] * 4).tolist(),
                                     10.0, 0.0, 0.0, 0.0]]])
        self.radar_valid = torch.ones((1, 1), dtype=torch.bool)
        self.observed = torch.tensor([[[*(self.ray[2] * 5).tolist(), 7.0]]])
        self.observed_valid = torch.ones((1, 1), dtype=torch.bool)
        self.clean = torch.tensor([[[*(self.ray[0] * 10).tolist(), 4.0],
                                    [*(self.ray[2] * 5).tolist(), 7.0]]])
        self.clean_valid = torch.ones((1, 2), dtype=torch.bool)
        self.relation_model = RadarRelationEncoder(
            self.geometry, self.grid, width=8, history_scans=20)
        self.diffusion = RadarRelationDiffusion(
            self.geometry, relation_width=8, hidden=8, timesteps=8)

    def test_radar_only_relation_and_training_only_clean_attention(self):
        relation = self.relation_model(
            self.radar, self.radar_valid, self.rows, self.cols)
        self.assertEqual(relation.features.shape, (1, 8, 8))
        self.assertTrue(bool(relation.support[0, 0]))
        self.assertFalse(bool(relation.support[0, 4]))
        teacher = CleanLidarRelationshipTeacher(
            self.geometry, self.grid, width=8, history_scans=20)
        teacher.eval().requires_grad_(False)
        paired = PairedRadarLidarAttention(8)
        nearby_clean = self.clean.clone()
        nearby_clean[0, 0, :3] = self.ray[0] * 4.5
        loss = joint_relation_loss(
            relation, paired, teacher, nearby_clean, self.clean_valid, (1, 8))
        self.assertTrue(torch.isfinite(loss["loss"]))
        self.assertGreater(int(loss["paired_rays"]), 0)
        diffusion_loss = self.diffusion.training_loss(
            relation, self.observed, self.observed_valid,
            self.clean, self.clean_valid, (1, 8),
            timestep=torch.tensor([3]), noise=torch.ones((1, 1, 1, 8)))
        self.assertEqual(int(diffusion_loss["clean_hits"]), 1)
        self.assertTrue(torch.isfinite(diffusion_loss["loss"]))
        self.assertGreaterEqual(float(diffusion_loss["metric_depth"].detach()), 0.0)
        self.assertNotIn("intensity", diffusion_loss)
        (loss["loss"] + diffusion_loss["loss"]).backward()
        self.assertIsNotNone(self.diffusion.denoiser.output[-1].weight.grad)
        self.assertIsNotNone(self.diffusion.return_head[-1].weight.grad)
        self.assertIsNotNone(self.relation_model.output[1].weight.grad)
        self.assertIsNotNone(paired.hit_head.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_absolute_depth_target_is_not_limited_to_three_meters(self):
        self.assertAlmostEqual(float(denormalize_depth(
            normalize_depth(torch.tensor(10.0), self.geometry),
            self.geometry)), 10.0, places=4)
        relation = self.relation_model(
            self.radar, self.radar_valid, self.rows, self.cols)
        self.assertGreater(float(self.clean[0, 0, :3].norm() -
                                 relation.depth_hint_m[0, 0]), 3.0)
        loss = self.diffusion.training_loss(
            relation, self.observed, self.observed_valid,
            self.clean, self.clean_valid, (1, 8),
            timestep=torch.tensor([2]), noise=torch.zeros((1, 1, 1, 8)))
        self.assertEqual(int(loss["clean_hits"]), 1)

    def test_return_loss_does_not_read_noised_clean_depth(self):
        relation = self.relation_model(
            self.radar, self.radar_valid, self.rows, self.cols)
        shape = (1, 1, 1, 8)
        first = self.diffusion.training_loss(
            relation, self.observed, self.observed_valid,
            self.clean, self.clean_valid, (1, 8),
            timestep=torch.tensor([1]), noise=torch.zeros(shape))
        second = self.diffusion.training_loss(
            relation, self.observed, self.observed_valid,
            self.clean, self.clean_valid, (1, 8),
            timestep=torch.tensor([6]), noise=torch.ones(shape))
        torch.testing.assert_close(first["return"], second["return"])
        self.assertNotEqual(float(first["metric_depth"].detach()),
                            float(second["metric_depth"].detach()))

    def test_five_metre_depth_error_has_substantial_weight(self):
        relation = self.relation_model(
            self.radar, self.radar_valid, self.rows, self.cols)

        def predict_five_metres(noisy, _static, timestep):
            alpha = self.diffusion.schedule.sqrt_alpha_bars[timestep].view(
                -1, 1, 1, 1)
            sigma = self.diffusion.schedule.sqrt_one_minus_alpha_bars[
                timestep].view(-1, 1, 1, 1)
            predicted = normalize_depth(torch.full_like(noisy, 5), self.geometry)
            epsilon = (noisy - alpha * predicted) / sigma
            return epsilon, torch.zeros_like(noisy), torch.zeros_like(noisy)

        with patch.object(self.diffusion.denoiser, "forward",
                          side_effect=predict_five_metres):
            loss = self.diffusion.training_loss(
                relation, self.observed, self.observed_valid,
                self.clean, self.clean_valid, (1, 8),
                timestep=torch.tensor([3]), noise=torch.zeros((1, 1, 1, 8)))
        self.assertGreater(
            float((self.diffusion.metric_depth_weight *
                   loss["metric_depth"]).detach()),
            10 * float((0.25 * loss["depth"]).detach()))

    def test_empty_radar_preserves_observed_lidar_without_additions(self):
        empty = self.radar[:, :0]
        valid = self.radar_valid[:, :0]
        self.relation_model.eval()
        self.diffusion.eval()
        with patch.object(self.diffusion.denoiser, "forward",
                          side_effect=AssertionError("Denoiser must be skipped")):
            cloud = sample_joint_full_scan(
                self.relation_model, self.diffusion,
                empty, valid, self.observed, self.observed_valid,
                tile_rows=1, tile_cols=8, steps=2)
        torch.testing.assert_close(cloud, self.observed[0])

    def test_full_scan_adds_only_supported_rays_and_preserves_original(self):
        self.relation_model.eval()
        self.diffusion.eval()
        def accept_supported(noisy, _static, _timestep):
            return torch.zeros_like(noisy), torch.full_like(noisy, 10), torch.zeros_like(noisy)
        with patch.object(self.diffusion.denoiser, "forward",
                          side_effect=accept_supported), \
                patch.object(self.diffusion.return_head, "forward",
                             side_effect=lambda static: torch.full_like(
                                 static[:, :1], 10)):
            cloud = sample_joint_full_scan(
                self.relation_model, self.diffusion,
                self.radar, self.radar_valid,
                self.observed, self.observed_valid,
                tile_rows=1, tile_cols=8, steps=2)
        self.assertGreater(len(cloud), len(self.observed[0]))
        torch.testing.assert_close(cloud[:1], self.observed[0])
        self.assertTrue(bool((cloud[1:, :3].norm(dim=-1) >= 0.5 - 1e-4).all()))
        self.assertTrue(bool((cloud[1:, :3].norm(dim=-1) <= 12 + 1e-4).all()))
        self.assertTrue(bool((cloud[1:, 3] == 0).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_joint_step(self):
        device = torch.device("cuda")
        relation_model = self.relation_model.to(device)
        diffusion = self.diffusion.to(device)
        teacher = CleanLidarRelationshipTeacher(
            self.geometry, self.grid, width=8, history_scans=20).to(device)
        teacher.eval().requires_grad_(False)
        paired = PairedRadarLidarAttention(8).to(device)
        radar = self.radar.to(device)
        radar_valid = self.radar_valid.to(device)
        observed = self.observed.to(device)
        observed_valid = self.observed_valid.to(device)
        clean = self.clean.to(device)
        clean_valid = self.clean_valid.to(device)
        relation = relation_model(radar, radar_valid,
                                  self.rows.to(device), self.cols.to(device))
        reconstruction = diffusion.training_loss(
            relation, observed, observed_valid, clean, clean_valid, (1, 8))
        alignment = joint_relation_loss(
            relation, paired, teacher, clean, clean_valid, (1, 8))
        total = reconstruction["loss"] + alignment["loss"]
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()


if __name__ == "__main__":
    unittest.main()
