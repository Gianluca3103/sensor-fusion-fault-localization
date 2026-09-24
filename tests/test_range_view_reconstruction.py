import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.two_stage_reconstruction_head.range_view.geometry import (
    RangeGeometry, angular_indices, backproject, project_lidar, transform_points, transform_radar_to_lidar,
)
from models.two_stage_reconstruction_head.range_view.radar import project_aligned_radar
from models.two_stage_reconstruction_head.range_view.targets import build_range_targets
from models.two_stage_reconstruction_head.range_view.merge import MergeConfig, merge_reconstruction
from models.two_stage_reconstruction_head.range_view.model import (
    CircularHorizontalConv, RangeModelConfig, RangeViewReconstructor,
)
from models.two_stage_reconstruction_head.range_view.loss import range_edit_loss
from models.two_stage_reconstruction_head.range_view.metrics import evaluate_xyz
from scripts.visualize_range_view_inputs import _preview_range


class RangeViewTests(unittest.TestCase):
    def setUp(self):
        self.geometry = RangeGeometry(
            beam_elevations_rad=(-0.1, 0.1), azimuth_bins=16,
            min_range_m=0.1, max_range_m=50.0,
            azimuth_span_rad=2 * np.pi,
        )

    def point(self, row, col, distance, intensity=0.5):
        return np.r_[backproject(np.asarray(row), np.asarray(col),
                                 np.asarray(distance), self.geometry), intensity].astype(np.float32)

    def test_xyz_range_xyz_round_trip_and_nearest_visible_return(self):
        points = np.stack([self.point(0, 0, 8), self.point(1, 15, 12), self.point(0, 0, 4)])
        projected = project_lidar(points, self.geometry)
        self.assertEqual(projected.collision_count[0, 0], 2)
        self.assertEqual(projected.nearest_original_index[0, 0], 2)
        self.assertAlmostEqual(float(projected.range_m[0, 0]), 4, places=5)
        rows, cols = np.nonzero(projected.valid)
        recovered = backproject(rows, cols, projected.range_m[rows, cols], self.geometry)
        winners = points[projected.nearest_original_index[rows, cols], :3]
        self.assertLess(float(np.max(np.linalg.norm(recovered - winners, axis=1))), 1e-4)

    def test_azimuth_seam_wrap_and_partial_field_of_view(self):
        near_zero = np.asarray([[5.0, 0.0001, -0.5]], dtype=np.float32)
        near_two_pi = np.asarray([[5.0, -0.0001, -0.5]], dtype=np.float32)
        self.assertEqual(int(angular_indices(near_zero, self.geometry)[1][0]), 0)
        self.assertEqual(int(angular_indices(near_two_pi, self.geometry)[1][0]), 15)
        front = RangeGeometry((-0.1, 0.1), 16, 0.1, 50.0, np.pi)
        self.assertFalse(bool(angular_indices(np.asarray([[-5.0, 0.0, -0.5]]), front)[3][0]))

    def test_radar_transform_and_aggregation(self):
        transform = np.eye(4)
        transform[0, 3] = 2
        radar = np.asarray([[2, 0, 0, 4, 1], [2, 0, 0, 8, 3]], dtype=np.float32)
        aligned = transform_radar_to_lidar(radar, transform)
        self.assertAlmostEqual(float(aligned[0, 0]), 4)
        np.testing.assert_array_equal(transform_points(radar, transform)[:, 3:], radar[:, 3:])
        features = project_aligned_radar(aligned, self.geometry)
        self.assertEqual(int((features[0] > 0).sum()), 1)
        row, col = np.nonzero(features[0])
        self.assertAlmostEqual(float(features[3, row[0], col[0]]), 6)
        self.assertAlmostEqual(float(features[4, row[0], col[0]]), 2)

    def test_add_keep_delete_replace_targets(self):
        clean = np.stack([self.point(0, 0, 5), self.point(0, 2, 8), self.point(0, 4, 10)])
        faulty = np.stack([self.point(0, 0, 5), self.point(0, 3, 7), self.point(0, 4, 15)])
        targets = build_range_targets(project_lidar(faulty, self.geometry),
                                      project_lidar(clean, self.geometry),
                                      faulty, clean, np.asarray([0, -1, 2]))
        self.assertTrue(targets.keep[0, 0])
        self.assertTrue(targets.add[0, 2])
        self.assertTrue(targets.delete[0, 3])
        self.assertTrue(targets.replace[0, 4])
        self.assertTrue(targets.add[0, 4] and targets.delete[0, 4])

    def test_identity_append_only_generated_provenance_and_conservative_delete(self):
        original = np.stack([self.point(0, 0, 5), self.point(0, 4, 10)])
        projection = project_lidar(original, self.geometry)
        zero = np.zeros(self.geometry.shape, dtype=np.float32)
        identity = merge_reconstruction(original, projection, self.geometry, zero,
                                        np.ones_like(zero) * 5, zero)
        np.testing.assert_array_equal(identity.points, original)
        add = zero.copy(); add[0, 0] = 0.9; add[1, 8] = 0.8
        ranges = np.ones_like(zero) * 7
        delete = zero.copy(); delete[0, 0] = 0.998; delete[0, 4] = 0.9995
        append = merge_reconstruction(original, projection, self.geometry, add,
                                      ranges, delete, config=MergeConfig(False, 0.999, 0.5))
        np.testing.assert_array_equal(append.points[:len(original)], original)
        self.assertEqual(len(append.generated_points), 2)
        self.assertEqual(append.same_ray_original_and_generated, 1)
        self.assertTrue(np.all(append.output_is_generated[-2:]))
        np.testing.assert_allclose(append.generated_points[1, :3], self.point(1, 8, 7)[:3], atol=1e-5)
        conservative = merge_reconstruction(original, projection, self.geometry, add,
                                            ranges, delete, config=MergeConfig(True, 0.999, 0.5))
        np.testing.assert_array_equal(conservative.deleted_original_indices, [1])
        np.testing.assert_array_equal(conservative.points[0], original[0])
        pruned = merge_reconstruction(original, projection, self.geometry, add,
                                      ranges, delete, config=MergeConfig(False, 0.999, 0.95))
        np.testing.assert_array_equal(pruned.points, original)

    def test_model_circular_seam_outputs_and_masked_losses(self):
        torch.manual_seed(4)
        convolution = CircularHorizontalConv(1, 1)
        x = torch.randn(1, 1, 2, 16)
        self.assertTrue(torch.allclose(convolution(torch.roll(x, 1, -1)),
                                       torch.roll(convolution(x), 1, -1), atol=1e-6))
        model = RangeViewReconstructor(RangeModelConfig(hidden_channels=8, max_range_m=50))
        result = model(torch.zeros(1, 10, 2, 16))
        self.assertEqual(result["add_probability"].shape, (1, 2, 16))
        self.assertTrue(bool(torch.all(result["add_range_m"] > 0)))
        target = {key: torch.zeros(1, 2, 16) for key in
                  ("add", "add_range_m", "delete", "delete_valid", "clean_valid", "clean_range_m")}
        losses = range_edit_loss(result, target)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(float(losses["range_loss"].detach()), 0.0)

    def test_no_original_points_have_undefined_preservation_rate(self):
        clean = np.stack([self.point(0, 0, 5)])
        faulty = np.empty((0, 4), dtype=np.float32)
        faulty_projection = project_lidar(faulty, self.geometry)
        targets = build_range_targets(
            faulty_projection, project_lidar(clean, self.geometry),
            faulty, clean, np.empty(0, dtype=np.int64),
        )
        empty_map = np.zeros(self.geometry.shape, dtype=np.float32)
        merged = merge_reconstruction(
            faulty, faulty_projection, self.geometry, empty_map,
            np.ones_like(empty_map) * 5, empty_map,
        )
        sample = SimpleNamespace(
            targets=targets, faulty_points=faulty, clean_points=clean,
            faulty_source_ids=np.empty(0, dtype=np.int64),
        )
        metrics = evaluate_xyz(sample, merged)
        self.assertTrue(np.isnan(metrics["healthy_original_preservation_rate"]))
        self.assertTrue(np.isnan(metrics["false_original_delete_rate"]))

    def test_multi_original_ray_cannot_be_deleted_by_ray_score(self):
        original = np.stack([self.point(0, 0, 5), self.point(0, 0, 6)])
        projection = project_lidar(original, self.geometry)
        add = np.zeros(self.geometry.shape, dtype=np.float32)
        delete = add.copy(); delete[0, 0] = 1.0
        merged = merge_reconstruction(original, projection, self.geometry, add,
                                      np.ones_like(add) * 7, delete,
                                      config=MergeConfig(True, 0.999, 0.5))
        np.testing.assert_array_equal(merged.points, original)

    def test_forward_only_merge_blocks_rear_additions(self):
        original = np.stack([self.point(0, 0, 5)])
        projection = project_lidar(original, self.geometry)
        add = np.ones(self.geometry.shape, dtype=np.float32)
        delete = np.zeros_like(add)
        merged = merge_reconstruction(
            original, projection, self.geometry, add, np.ones_like(add) * 7,
            delete, config=MergeConfig(forward_only=True),
        )
        self.assertTrue(bool(np.all(merged.points[:, 0] >= 0)))
        self.assertLess(len(merged.generated_points), add.size)
        with self.assertRaisesRegex(ValueError, "front-filtered"):
            merge_reconstruction(
                np.stack([self.point(0, 8, 5)]),
                project_lidar(np.stack([self.point(0, 8, 5)]), self.geometry),
                self.geometry, add, np.ones_like(add) * 7, delete,
                config=MergeConfig(forward_only=True),
            )

    def test_angular_preview_keeps_nearest_range_in_cell(self):
        points = np.asarray([[5, 0, 0, 1], [10, 0, 0, 1]], dtype=np.float32)
        image = _preview_range(points, np.linspace(-90, 90, 11),
                               np.linspace(-10, 10, 11))
        self.assertAlmostEqual(float(np.nanmin(image)), 5.0)
        self.assertEqual(int(np.isfinite(image).sum()), 1)


if __name__ == "__main__":
    unittest.main()
