import math
import unittest
from unittest.mock import patch

import torch

from models.two_stage_reconstruction_head.diffusion_process.ray_view_diffusion import (
    RadarGatedRayDiffusion, calibrate_reliability_threshold,
    project_lidar_tile, sample_full_scan,
)
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry, angular_indices
from models.two_stage_reconstruction_head.ray_depth_attention import RayDepthBlueprint
from models.two_stage_reconstruction_head.ray_depth_queries import (
    RayDepthQueries, ray_tile_indices,
)


class RadarGatedRayDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.geometry = RangeGeometry(
            beam_elevations_rad=(-0.1, 0.0, 0.1), azimuth_bins=8,
            min_range_m=0.5, max_range_m=12.0,
            azimuth_span_rad=2 * math.pi,
        )
        self.rows, self.cols = ray_tile_indices(
            self.geometry, row_start=1, row_stop=2, batch_size=1,
        )
        direction = torch.tensor(self.geometry.ray_directions()[1].copy())
        self.directions = direction
        self.observed = torch.tensor([[[*(direction[2] * 4).tolist(), 7.0],
                                        [20.0, 0.0, 0.0, 3.0]]])
        self.observed_valid = torch.tensor([[True, True]])
        self.clean = torch.tensor([[[*(direction[0] * 4.5).tolist(), 5.0],
                                     [*(direction[2] * 4).tolist(), 7.0],
                                     [*(direction[4] * 10).tolist(), 9.0]]])
        self.clean_valid = torch.tensor([[True, True, True]])
        self.model = RadarGatedRayDiffusion(
            self.geometry, blueprint_width=8, hidden=8,
            timesteps=8, max_correction_m=3.0,
        )

    def blueprint(self, radar_columns=(0, 2, 4)):
        depths = torch.tensor([4.0, 8.0]).reshape(1, 1, 2).expand(1, 8, 2).clone()
        valid = torch.ones((1, 8, 2), dtype=torch.bool)
        source = torch.zeros((1, 8, 2), dtype=torch.long)
        queries = RayDepthQueries(self.rows, self.cols,
            self.directions.reshape(1, 8, 3), depths, valid, source)
        features = torch.randn(1, 8, 2, 8)
        evidence = torch.zeros((1, 8, 2, 3))
        evidence[..., 2] = 1.0
        for col in radar_columns:
            evidence[0, col, 0] = torch.tensor([0.9, 0.0, 0.1])
        logits = torch.tensor([3.0, 0.0, -1.0]).reshape(1, 1, 3).expand(1, 8, 3).clone()
        return RayDepthBlueprint(queries, features, evidence, logits,
                                 torch.zeros_like(depths))

    def test_projection_keeps_nearest_return_and_intensity(self):
        double = torch.cat((self.clean, torch.tensor([[[*(self.directions[0] * 6).tolist(),
                                                      99.0]]])), 1)
        depth, intensity, hit = project_lidar_tile(
            self.geometry, self.rows, self.cols,
            double, torch.ones((1, len(double[0])), dtype=torch.bool),
        )
        self.assertTrue(hit[0, 0])
        self.assertAlmostEqual(depth[0, 0].item(), 4.5, places=3)
        self.assertAlmostEqual(intensity[0, 0].item(), 5.0, places=3)
        self.assertFalse(hit[0, 1])

    def test_training_supervises_only_input_derived_radar_proposals(self):
        blueprint = self.blueprint()
        condition = self.model.prepare_condition(
            blueprint, self.observed, self.observed_valid, (1, 8))
        self.assertEqual(torch.nonzero(condition.proposal_mask[0, 0].reshape(-1)).flatten().tolist(),
                         [0, 4])
        self.assertTrue(condition.observed_mask[0, 0, 0, 2])
        losses = self.model.training_loss(
            blueprint, self.observed, self.observed_valid,
            self.clean, self.clean_valid, (1, 8),
            timestep=torch.tensor([3]), noise=torch.ones((1, 1, 1, 8)),
        )
        self.assertEqual(int(losses["radar_supported_rays"]), 2)
        self.assertEqual(int(losses["correctable_rays"]), 1)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(self.model.reliability[0].weight.grad)
        self.assertIsNotNone(self.model.denoiser.output[-1].weight.grad)

    def test_sampling_cannot_add_outside_radar_or_replace_observed(self):
        blueprint = self.blueprint()
        self.model.eval()
        result = self.model.sample(
            blueprint, self.observed, self.observed_valid, (1, 8),
            steps=3, reliability_threshold=0.0, return_threshold=0.0,
        )
        self.assertEqual(torch.nonzero(result.added_mask[0].reshape(-1)).flatten().tolist(), [0, 4])
        self.assertAlmostEqual(float(result.depth_m[0, 0, 2]), 4.0, places=5)
        self.assertEqual(float(result.intensity[0, 0, 2]), 7.0)
        self.assertFalse(bool(result.added_mask[0, 0, 2]))
        self.assertTrue(bool((result.depth_m[result.added_mask] >= 0.5).all()))
        self.assertTrue(bool((result.depth_m[result.added_mask] <= 12).all()))
        additions = result.added_points(0)
        self.assertEqual(additions.shape, (2, 4))
        merged = result.merge_with_observed(self.observed, self.observed_valid, 0)
        torch.testing.assert_close(merged[:2], self.observed[0])

    def test_no_radar_support_returns_only_original_points(self):
        result = self.model.sample(
            self.blueprint(radar_columns=()), self.observed, self.observed_valid,
            (1, 8), steps=3, reliability_threshold=0.0, return_threshold=0.0,
        )
        self.assertFalse(bool(result.added_mask.any()))
        self.assertEqual(result.added_points(0).shape, (0, 4))
        torch.testing.assert_close(
            result.merge_with_observed(self.observed, self.observed_valid, 0),
            self.observed[0],
        )

    def test_reliability_calibration_handles_ties_and_abstention(self):
        scores = torch.tensor([0.95, 0.9, 0.9, 0.4])
        labels = torch.tensor([True, True, False, False])
        calibrated = calibrate_reliability_threshold(
            scores, labels, minimum_precision=0.6, minimum_predictions=2)
        self.assertEqual(calibrated["accepted"], 3)
        self.assertAlmostEqual(calibrated["threshold"], 0.9, places=5)
        abstained = calibrate_reliability_threshold(
            scores, labels, minimum_precision=0.9, minimum_predictions=2)
        self.assertEqual(abstained["threshold"], 1.0)
        self.assertEqual(abstained["accepted"], 0)

    def test_reverse_steps_keep_noise_zero_outside_support(self):
        blueprint = self.blueprint()
        seen = []
        original = self.model.denoiser.forward
        def record(noisy, static, timestep):
            seen.append(noisy.detach().clone())
            return original(noisy, static, timestep)
        with patch.object(self.model.denoiser, "forward", side_effect=record):
            self.model.sample(blueprint, self.observed, self.observed_valid,
                              (1, 8), steps=4,
                              reliability_threshold=0.0, return_threshold=0.0)
        self.assertEqual(len(seen), 4)
        outside = torch.ones((1, 1, 1, 8), dtype=torch.bool)
        outside[..., 0] = False
        outside[..., 4] = False
        for state in seen:
            self.assertTrue(bool((state[outside] == 0).all()))

    def test_full_scan_stitches_each_ray_once_and_encodes_once(self):
        geometry = RangeGeometry(
            beam_elevations_rad=(-0.1, 0.1), azimuth_bins=9,
            min_range_m=0.5, max_range_m=12.0,
            azimuth_span_rad=2 * math.pi,
        )
        diffusion = RadarGatedRayDiffusion(
            geometry, blueprint_width=8, hidden=8, timesteps=8).eval()

        class DummyFusion:
            max_candidates = 2048

            def __call__(self, _geometry, queries, _radar, _observed, _raw):
                batch, rays, slots = queries.depths_m.shape
                features = torch.zeros(batch, rays, slots, 8)
                evidence = torch.zeros(batch, rays, slots, 3)
                evidence[..., 0] = 0.9
                evidence[..., 2] = 0.1
                logits = torch.empty(batch, rays, slots + 1)
                logits[..., :slots] = 3
                logits[..., -1] = -1
                return RayDepthBlueprint(queries, features, evidence, logits,
                                         torch.zeros_like(queries.depths_m))

        class DummyBlueprint:
            training = False
            fusion = DummyFusion()

            def __init__(self):
                self.geometry = geometry
                self.encoder_calls = 0

            def encoders(self, *_args):
                self.encoder_calls += 1
                return {"radar": None, "observed_lidar": None}

        def queries(_geometry, rows, cols, *_args):
            directions = torch.tensor(
                geometry.ray_directions().copy())[rows, cols]
            shape = (*rows.shape, 1)
            return RayDepthQueries(rows, cols, directions,
                                   torch.full(shape, 4.0),
                                   torch.ones(shape, dtype=torch.bool),
                                   torch.zeros(shape, dtype=torch.long))

        model = DummyBlueprint()
        observed = torch.tensor([[[*(geometry.ray_directions()[0, 2] * 4).tolist(), 7.0],
                                  [20.0, 0.0, 0.0, 3.0]]])
        radar = torch.zeros(1, 1, 7)
        with patch("models.two_stage_reconstruction_head.diffusion_process."
                   "ray_view_diffusion.propose_ray_depth_queries", side_effect=queries), \
             patch("models.two_stage_reconstruction_head.diffusion_process."
                   "ray_view_diffusion.project_lidar_tile",
                   wraps=project_lidar_tile) as projection:
            merged = sample_full_scan(
                model, diffusion, radar, torch.ones(1, 1, dtype=torch.bool),
                observed, torch.ones(1, 2, dtype=torch.bool),
                tile_rows=1, tile_cols=4, steps=3,
                reliability_threshold=0.0, return_threshold=0.0,
            )
        self.assertEqual(projection.call_count, 1)
        self.assertEqual(model.encoder_calls, 1)
        torch.testing.assert_close(merged[:2], observed[0])
        self.assertEqual(len(merged), 19)  # two originals plus 17 unique missing rays
        row, col, _, valid = angular_indices(merged[2:, :3].numpy(), geometry)
        self.assertTrue(valid.all())
        self.assertEqual(len(set(zip(row.tolist(), col.tolist()))), 17)

        abstained = sample_full_scan(
            model, diffusion, radar, torch.ones(1, 1, dtype=torch.bool),
            observed, torch.ones(1, 2, dtype=torch.bool),
            tile_rows=1, tile_cols=4, reliability_threshold=1.0,
        )
        torch.testing.assert_close(abstained, observed[0])
        self.assertEqual(model.encoder_calls, 1)


if __name__ == "__main__":
    unittest.main()
