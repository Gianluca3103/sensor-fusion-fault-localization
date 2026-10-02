import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from models.two_stage_reconstruction_head.range_view.geometry import (
    RangeGeometry, angular_indices, backproject, project_lidar, transform_points, transform_radar_to_lidar,
)
from models.two_stage_reconstruction_head.range_view.radar import (
    filter_radar_below_lidar, filter_radar_floor_band, project_aligned_radar,
)
from models.two_stage_reconstruction_head.range_view.targets import build_range_targets
from models.two_stage_reconstruction_head.range_view.merge import MergeConfig, merge_reconstruction
from models.two_stage_reconstruction_head.range_view.model import (
    CircularHorizontalConv, RangeModelConfig, RangeViewReconstructor,
)
from models.two_stage_reconstruction_head.range_view.loss import range_edit_loss
from models.two_stage_reconstruction_head.range_view.metrics import evaluate_xyz
from models.two_stage_reconstruction_head.range_view.data import (
    RangeViewDataset, load_range_sample, rotate_points_yaw,
)
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

    def test_online_yaw_rotates_all_modalities_without_changing_provenance(self):
        clean = np.asarray([[5, 0, -0.5, 0.7]], dtype=np.float32)
        radar = np.asarray([[5, 0, -0.5, 3.0, 2.0]], dtype=np.float32)
        angle = np.deg2rad(30)
        rotated = rotate_points_yaw(radar, angle)
        np.testing.assert_allclose(rotated[0, :3], [5 * np.cos(angle), 5 * np.sin(angle), -0.5], atol=1e-6)
        np.testing.assert_array_equal(rotated[:, 3:], radar[:, 3:])
        np.testing.assert_array_equal(radar[0, :2], [5, 0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npz"
            np.savez(path, faulty_source_ids=np.asarray([0], dtype=np.int64))
            aligned = SimpleNamespace(
                lidar_points=clean.copy(), radar_points=radar.copy(),
                metadata={"range_view_full_scan": True, "dataset": "View-of-Delft"},
            )
            with patch("models.two_stage_reconstruction_head.range_view.data.load_aligned_point_inputs",
                       return_value=aligned), patch(
                "models.two_stage_reconstruction_head.range_view.data.load_clean_lidar_from_metadata",
                return_value=clean.copy(),
            ):
                sample = load_range_sample(path, Path(temporary), self.geometry,
                                           yaw_rotation_rad=angle)
            np.testing.assert_allclose(sample.faulty_points[:, :3],
                                       rotate_points_yaw(clean, angle)[:, :3], atol=1e-6)
            np.testing.assert_array_equal(sample.faulty_source_ids, [0])
            self.assertTrue(bool(sample.targets.healthy_original[0]))
            self.assertAlmostEqual(sample.metadata["online_yaw_deg"], 30)
            self.assertEqual(int((sample.radar_features[0] > 0).sum()), 1)

    def test_dataset_draws_new_yaw_only_when_enabled(self):
        with patch("models.two_stage_reconstruction_head.range_view.data.load_range_sample") as load:
            load.return_value.tensors.return_value = {}
            augmented = RangeViewDataset([Path("a.npz")], Path("radar"), self.geometry,
                                         online_yaw_deg=5)
            with patch("numpy.random.uniform", side_effect=[-5.0, 5.0]):
                augmented[0]
                augmented[0]
            angles = [call.kwargs["yaw_rotation_rad"] for call in load.call_args_list]
            np.testing.assert_allclose(angles, np.deg2rad([-5, 5]))
            load.reset_mock()
            unaugmented = RangeViewDataset([Path("a.npz")], Path("radar"), self.geometry)
            unaugmented[0]
            self.assertEqual(load.call_args.kwargs["yaw_rotation_rad"], 0)

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

    def test_radar_floor_band_removes_points_within_ten_centimeters_of_minimum(self):
        radar = np.asarray([
            [4, 0, -1.00, 5, 0],
            [5, 0, -0.94, 6, 0],
            [6, 0, -0.89, 7, 0],
            [7, 0, -0.50, 8, 0],
        ], dtype=np.float32)
        filtered, floor = filter_radar_floor_band(radar, 0.10)
        self.assertAlmostEqual(floor, -1.0)
        np.testing.assert_array_equal(filtered, radar[2:])
        unchanged, no_floor = filter_radar_floor_band(radar, 0.0)
        np.testing.assert_array_equal(unchanged, radar)
        self.assertIsNone(no_floor)

    def test_radar_below_lidar_uses_faulty_input_and_keeps_equal_height(self):
        radar = np.asarray([
            [4, 0, -1.8, 5, 0], [5, 0, -1.65, 6, 0], [6, 0, -1.4, 7, 0],
        ], dtype=np.float32)
        faulty = np.asarray([[5, 0, -1.65, 0.5], [8, 0, -0.5, 0.4]], dtype=np.float32)
        filtered, minimum = filter_radar_below_lidar(radar, faulty)
        self.assertAlmostEqual(minimum, float(faulty[0, 2]))
        np.testing.assert_array_equal(filtered, radar[1:])
        unchanged, no_minimum = filter_radar_below_lidar(
            radar, np.empty((0, 4), dtype=np.float32),
        )
        np.testing.assert_array_equal(unchanged, radar)
        self.assertIsNone(no_minimum)

    def test_loaded_sample_records_radar_floor_filter_without_changing_lidar(self):
        clean = np.asarray([[5, 0, -1.5, 0.7]], dtype=np.float32)
        radar = np.asarray([[5, 0, -1.0, 3, 0], [5, 0, -0.95, 4, 0],
                            [5, 0, -0.5, 5, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npz"
            np.savez(path, faulty_source_ids=np.asarray([0], dtype=np.int64))
            aligned = SimpleNamespace(
                lidar_points=clean.copy(), radar_points=radar.copy(),
                metadata={"range_view_full_scan": True, "dataset": "View-of-Delft"},
            )
            with patch("models.two_stage_reconstruction_head.range_view.data.load_aligned_point_inputs",
                       return_value=aligned), patch(
                "models.two_stage_reconstruction_head.range_view.data.load_clean_lidar_from_metadata",
                return_value=clean.copy(),
            ):
                sample = load_range_sample(path, Path(temporary), self.geometry,
                                           radar_floor_band_m=0.10,
                                           use_ray_encoding=True)
            np.testing.assert_array_equal(sample.faulty_points, clean)
            np.testing.assert_array_equal(sample.clean_points, clean)
            self.assertEqual(sample.features.shape[0], 13)
            np.testing.assert_allclose(np.moveaxis(sample.features[-3:], 0, -1),
                                       self.geometry.ray_directions())
            self.assertEqual(sample.metadata["radar_floor_removed_points"], 2)
            self.assertAlmostEqual(sample.metadata["radar_floor_reference_z_m"], -1.0)
            np.testing.assert_array_equal(sample.radar_points, radar[2:])

    def test_model_uses_optional_ray_encoding_without_changing_legacy_inputs(self):
        encoded = RangeViewReconstructor(RangeModelConfig(
            hidden_channels=8, max_range_m=50, use_ray_encoding=True))
        legacy = RangeViewReconstructor(RangeModelConfig(
            hidden_channels=8, max_range_m=50))
        self.assertEqual(encoded.input_channels, 13)
        self.assertEqual(legacy.input_channels, 10)
        with torch.no_grad():
            output = encoded(torch.zeros(1, 13, 2, 16))
        self.assertEqual(tuple(output["add_range_m"].shape), (1, 2, 16))

    def test_loaded_sample_uses_faulty_not_clean_lidar_for_radar_cutoff(self):
        clean = np.asarray([[5, 0, -1.5, 0.7], [5, 0, -0.5, 0.7]], dtype=np.float32)
        faulty = clean[1:].copy()
        radar = np.asarray([[5, 0, -1.0, 3, 0], [5, 0, -0.5, 4, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npz"
            np.savez(path, faulty_source_ids=np.asarray([1], dtype=np.int64))
            aligned = SimpleNamespace(
                lidar_points=faulty, radar_points=radar,
                metadata={"range_view_full_scan": True, "dataset": "View-of-Delft"},
            )
            with patch("models.two_stage_reconstruction_head.range_view.data.load_aligned_point_inputs",
                       return_value=aligned), patch(
                "models.two_stage_reconstruction_head.range_view.data.load_clean_lidar_from_metadata",
                return_value=clean,
            ):
                sample = load_range_sample(path, Path(temporary), self.geometry)
                legacy = load_range_sample(path, Path(temporary), self.geometry,
                                           filter_radar_by_lidar_min=False)
        self.assertAlmostEqual(sample.metadata["radar_lidar_min_z_m"], -0.5)
        self.assertEqual(sample.metadata["radar_below_lidar_removed_points"], 1)
        self.assertEqual(legacy.metadata["radar_below_lidar_removed_points"], 0)
        self.assertGreater(float(legacy.radar_features[1].sum()),
                           float(sample.radar_features[1].sum()))

    def test_add_keep_delete_replace_targets(self):
        clean = np.stack([self.point(0, 0, 5), self.point(0, 2, 8), self.point(0, 4, 10)])
        faulty = np.stack([self.point(0, 0, 5), self.point(0, 3, 7), self.point(0, 4, 15)])
        targets = build_range_targets(project_lidar(faulty, self.geometry),
                                      project_lidar(clean, self.geometry),
                                      faulty, clean, np.asarray([0, -1, 2]))
        self.assertTrue(targets.keep[0, 0])
        self.assertTrue(targets.add[0, 2])
        self.assertAlmostEqual(float(targets.add_intensity[0, 2]), 0.5)
        self.assertEqual(float(targets.add_intensity[0, 0]), 0.0)
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

    def test_intensity_head_is_masked_to_add_targets(self):
        model = RangeViewReconstructor(RangeModelConfig(
            hidden_channels=8, max_range_m=50, predict_intensity=True,
        ))
        result = model(torch.zeros(1, 10, 2, 16))
        self.assertEqual(result["add_intensity"].shape, (1, 2, 16))
        self.assertTrue(bool(torch.all(result["add_intensity"] >= 0)))
        target = {key: torch.zeros(1, 2, 16) for key in (
            "add", "add_range_m", "add_intensity", "delete", "delete_valid",
            "clean_valid", "clean_range_m",
        )}
        target["add"][0, 0, 3] = 1
        target["add_intensity"][0, 0, 3] = 2
        loss = range_edit_loss(result, target)
        self.assertTrue(bool(torch.isfinite(loss["intensity_loss"])))
        self.assertGreater(float(loss["intensity_loss"].detach()), 0)
        loss["loss"].backward()
        self.assertIsNotNone(model.intensity_head.weight.grad)
        self.assertGreater(float(model.intensity_head.weight.grad.abs().sum()), 0)
        target["add"].zero_()
        self.assertEqual(float(range_edit_loss(result, target)["intensity_loss"]), 0)

    def test_generated_intensity_preserves_original_intensity(self):
        original = np.stack([self.point(0, 0, 5, intensity=0.8)])
        projection = project_lidar(original, self.geometry)
        add = np.zeros(self.geometry.shape, dtype=np.float32)
        add[0, 2] = 1
        intensity = np.zeros_like(add)
        intensity[0, 2] = 0.35
        merged = merge_reconstruction(
            original, projection, self.geometry, add,
            np.full_like(add, 7), np.zeros_like(add), add_intensity=intensity,
        )
        self.assertAlmostEqual(float(merged.points[0, 3]), 0.8)
        self.assertAlmostEqual(float(merged.generated_points[0, 3]), 0.35)

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
        self.assertIn("reconstructed_chamfer_m", metrics)
        without_chamfer = evaluate_xyz(sample, merged, compute_chamfer=False)
        self.assertNotIn("faulty_chamfer_m", without_chamfer)
        self.assertNotIn("reconstructed_chamfer_m", without_chamfer)
        self.assertIn("reconstructed_f1_at_0.2m", without_chamfer)

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
