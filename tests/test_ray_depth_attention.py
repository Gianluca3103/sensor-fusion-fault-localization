import math
import unittest

import torch

from models.two_stage_reconstruction_head.cross_modal_encoders import EncoderGrid
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.ray_depth_attention import RayDepthBlueprintModel
from models.two_stage_reconstruction_head.ray_depth_queries import (
    propose_ray_depth_queries, ray_tile_indices,
)
from models.two_stage_reconstruction_head.ray_depth_training import (
    clean_first_return_targets, ray_depth_blueprint_loss,
)


class RayDepthAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.geometry = RangeGeometry(
            beam_elevations_rad=(-0.1, 0.0, 0.1), azimuth_bins=8,
            min_range_m=0.5, max_range_m=12.0,
            azimuth_span_rad=2 * math.pi,
        )
        self.grid = EncoderGrid((-12, -12, -2), (12, 12, 2), (2, 2, 1))
        self.rows, self.cols = ray_tile_indices(
            self.geometry, row_start=1, row_stop=2,
            col_start=0, col_stop=4,
        )
        direction = torch.tensor(self.geometry.ray_directions()[1, 0].copy())
        self.xyz = direction * 4.0
        self.radar = torch.tensor([[
            [*self.xyz.tolist(), 12.0, 3.0, 2.0, 0.0],
            [*(self.xyz * 1.1).tolist(), 9.0, 2.0, 1.0, -1.0],
        ]])
        self.radar_valid = torch.tensor([[True, True]])
        self.lidar = torch.tensor([[[*(self.xyz * 1.02).tolist(), 8.0]]])
        self.lidar_valid = torch.tensor([[True]])
        self.model = RayDepthBlueprintModel(
            self.geometry, self.grid, width=8, heads=2,
            neighbors=4, chunk_size=8,
        )

    def test_proposals_are_sensor_grounded_and_survive_lidar_dropout(self):
        queries = propose_ray_depth_queries(
            self.geometry, self.rows, self.cols, self.radar,
            self.radar_valid, self.lidar, torch.zeros_like(self.lidar_valid),
        )
        self.assertEqual(queries.depths_m.shape, (1, 4, 5))
        self.assertTrue(queries.valid[0, 0, 0])
        self.assertAlmostEqual(queries.depths_m[0, 0, 0].item(), 4.0, places=3)
        self.assertFalse(queries.valid[0, 0, 2])
        self.assertTrue(queries.valid[0, 0, -1])

    def test_null_is_selected_when_both_sensors_are_empty(self):
        self.model.eval()
        with torch.no_grad():
            result = self.model(
                self.radar, torch.zeros_like(self.radar_valid),
                self.lidar, torch.zeros_like(self.lidar_valid),
                self.rows, self.cols,
            )
        self.assertEqual(result.features.shape, (1, 4, 5, 8))
        self.assertEqual(result.first_return_logits.shape, (1, 4, 6))
        torch.testing.assert_close(
            result.evidence_weights[result.queries.valid],
            torch.tensor([0.0, 0.0, 1.0]).expand(
                int(result.queries.valid.sum()), -1),
        )
        self.assertFalse(torch.isnan(result.first_return_logits).any())
        _, exists = result.predicted_first_return()
        self.assertFalse(bool(exists.any()))

    def test_teacher_cannot_change_blueprint_and_is_rejected_at_inference(self):
        self.model.train()
        bare = self.model(self.radar, self.radar_valid, self.lidar,
                          self.lidar_valid, self.rows, self.cols)
        taught = self.model(self.radar, self.radar_valid, self.lidar,
                            self.lidar_valid, self.rows, self.cols,
                            clean_lidar=self.lidar * 1.3,
                            clean_valid=self.lidar_valid)
        self.assertIsNotNone(taught.clean_teacher)
        torch.testing.assert_close(bare.features, taught.features)
        torch.testing.assert_close(bare.first_return_logits,
                                   taught.first_return_logits)
        self.model.eval()
        with self.assertRaisesRegex(ValueError, "training supervision"):
            self.model(self.radar, self.radar_valid, self.lidar,
                       self.lidar_valid, self.rows, self.cols,
                       clean_lidar=self.lidar, clean_valid=self.lidar_valid)

    def test_radar_metadata_affects_blueprint_and_gradients_flow(self):
        self.model.eval()
        baseline = self.model(self.radar, self.radar_valid, self.lidar,
                              self.lidar_valid, self.rows, self.cols)
        changed_radar = self.radar.clone()
        changed_radar[0, 0, 3:6] = torch.tensor([-20.0, -7.0, -8.0])
        changed = self.model(changed_radar, self.radar_valid, self.lidar,
                             self.lidar_valid, self.rows, self.cols)
        self.assertFalse(torch.allclose(baseline.features, changed.features))
        baseline.first_return_logits.sum().backward()
        self.assertIsNotNone(self.model.fusion.blocks[0].radar[0].query.weight.grad)

    def test_wrong_radar_geometry_removes_local_support(self):
        self.model.eval()
        no_lidar = torch.zeros_like(self.lidar_valid)
        with torch.no_grad():
            aligned = self.model(self.radar, self.radar_valid, self.lidar,
                                 no_lidar, self.rows, self.cols)
            shifted = self.radar.clone()
            other_direction = torch.tensor(self.geometry.ray_directions()[1, 3].copy())
            shifted[..., :3] = other_direction * torch.tensor([4.0, 4.4])[None, :, None]
            mismatched = self.model(shifted, self.radar_valid, self.lidar,
                                    no_lidar, self.rows, self.cols)
        self.assertTrue(aligned.queries.valid[0, 0, 0])
        self.assertFalse(mismatched.queries.valid[0, 0, 0])
        self.assertTrue(bool((aligned.evidence_weights[0, 0, :, 0] > 0).any()))
        self.assertTrue(bool((mismatched.evidence_weights[0, 0, :, 2] == 1).any()))

    def test_clean_projection_and_training_loss(self):
        depth, present = clean_first_return_targets(
            self.geometry, self.rows, self.cols,
            torch.cat((self.lidar, self.lidar * 1.1), dim=1),
            torch.tensor([[True, True]]),
        )
        self.assertTrue(present[0, 0])
        self.assertAlmostEqual(depth[0, 0].item(), 4.08, places=2)
        self.assertFalse(bool(present[0, 1:].any()))
        self.model.train()
        result = self.model(self.radar, self.radar_valid, self.lidar,
                            self.lidar_valid, self.rows, self.cols,
                            clean_lidar=self.lidar, clean_valid=self.lidar_valid)
        losses = ray_depth_blueprint_loss(
            result, self.geometry, self.lidar, self.lidar_valid,
            grid=self.grid, teacher_weight=0.1,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["coverage"]), 0)
        losses["loss"].backward()
        self.assertIsNotNone(self.model.fusion.return_head.weight.grad)

    def test_batched_different_sensor_availability(self):
        rows, cols = ray_tile_indices(
            self.geometry, row_start=1, row_stop=2,
            col_start=0, col_stop=4, batch_size=2,
        )
        radar = self.radar.expand(2, -1, -1).clone()
        lidar = self.lidar.expand(2, -1, -1).clone()
        radar_valid = torch.tensor([[True, True], [False, False]])
        lidar_valid = torch.tensor([[False], [True]])
        result = self.model(radar, radar_valid, lidar, lidar_valid, rows, cols)
        self.assertEqual(result.first_return_logits.shape, (2, 4, 6))
        self.assertEqual(result.features.shape, (2, 4, 5, 8))
        self.assertTrue(bool((result.evidence_weights[0, 0, :, 0] > 0).any()))
        self.assertTrue(bool((result.evidence_weights[1, 0, :, 1] > 0).any()))


if __name__ == "__main__":
    unittest.main()
