import unittest

import numpy as np

from Fault_Localization_Model.bev_utils import metric_to_grid
from Fault_Localization_Model.lidar_observability import (
    _compute_ray_observations_compiled,
    compute_observability_confidence,
    compute_ray_observations,
    create_observability_map,
)


class LidarObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.x_range = (0.0, 5.0)
        self.y_range = (0.0, 5.0)
        self.resolution = 1.0
        self.origin = (0.5, 0.5, 0.0)

    def observations(self, points, **kwargs):
        return compute_ray_observations(
            np.asarray(points, dtype=np.float32),
            self.origin,
            self.x_range,
            self.y_range,
            self.resolution,
            z_range=kwargs.get("z_range", (-2.0, 4.0)),
            num_z_bins=kwargs.get("num_z_bins", 6),
        )

    def test_straight_ray_stops_before_hit_and_never_traces_behind(self):
        result = self.observations([[3.5, 0.5, 0.0]])

        self.assertEqual(result["ray_count"][:, 0].tolist(), [0, 0, 1, 1, 1])
        self.assertTrue(result["hit_mask"][1, 0])
        self.assertEqual(int(result["ray_count"][1, 0]), 0)
        self.assertEqual(int(result["ray_count"][0, 0]), 0)

    def test_diagonal_ray_crosses_only_diagonal_cells(self):
        result = self.observations([[3.5, 3.5, 0.0]])
        expected = np.zeros((5, 5), dtype=np.uint32)
        expected[4, 0] = expected[3, 1] = expected[2, 2] = 1

        np.testing.assert_array_equal(result["ray_count"], expected)
        self.assertTrue(result["hit_mask"][1, 3])

    def test_varying_z_marks_the_bins_crossed_in_each_pillar(self):
        result = compute_ray_observations(
            np.asarray([[3.5, 0.5, 1.5]], dtype=np.float32),
            (0.5, 0.5, -1.0),
            self.x_range,
            self.y_range,
            self.resolution,
            z_range=(-2.0, 4.0),
            num_z_bins=6,
        )

        self.assertEqual(np.flatnonzero(result["observed_z_bins"][4, 0]).tolist(), [1])
        self.assertEqual(np.flatnonzero(result["observed_z_bins"][3, 0]).tolist(), [1, 2])
        self.assertEqual(np.flatnonzero(result["observed_z_bins"][2, 0]).tolist(), [2, 3])
        self.assertGreater(result["vertical_coverage"][2, 0], result["vertical_coverage"][4, 0])

    def test_duplicate_ray_increases_count_but_not_vertical_coverage(self):
        single = self.observations([[3.5, 0.5, 0.0]])
        duplicate = self.observations(
            [[3.5, 0.5, 0.0], [3.5, 0.5, 0.0]]
        )

        self.assertEqual(int(duplicate["ray_count"][3, 0]), 2)
        self.assertEqual(int(single["ray_count"][3, 0]), 1)
        np.testing.assert_array_equal(
            duplicate["vertical_coverage"], single["vertical_coverage"]
        )

    def test_rays_at_different_heights_increase_vertical_coverage(self):
        low = compute_ray_observations(
            np.asarray([[3.5, 0.5, 1.5]], dtype=np.float32),
            (0.5, 0.5, -1.0),
            self.x_range,
            self.y_range,
            self.resolution,
            z_range=(-2.0, 4.0),
            num_z_bins=6,
        )
        mixed = compute_ray_observations(
            np.asarray([[3.5, 0.5, 1.5], [3.5, 0.5, 3.5]], dtype=np.float32),
            (0.5, 0.5, -1.0),
            self.x_range,
            self.y_range,
            self.resolution,
            z_range=(-2.0, 4.0),
            num_z_bins=6,
        )

        self.assertGreater(
            mixed["vertical_coverage"][3, 0],
            low["vertical_coverage"][3, 0],
        )

    def test_outside_return_traces_only_the_segment_inside_the_bev(self):
        result = self.observations([[7.0, 0.5, 0.0]])

        self.assertEqual(result["ray_count"][:, 0].tolist(), [1, 1, 1, 1, 1])
        self.assertFalse(result["hit_mask"].any())

    def test_exact_and_near_grid_boundaries_terminate_without_double_counting(self):
        exact = compute_ray_observations(
            np.asarray([[4.5, 1.0, 0.0]], dtype=np.float32),
            (0.5, 1.0, 0.0),
            self.x_range,
            self.y_range,
            self.resolution,
        )
        below = compute_ray_observations(
            np.asarray([[4.5, 1.0 - 1.0e-5, 0.0]], dtype=np.float32),
            (0.5, 1.0 - 1.0e-5, 0.0),
            self.x_range,
            self.y_range,
            self.resolution,
        )

        self.assertEqual(int(exact["ray_count"][:, 1].sum()), 4)
        self.assertEqual(int(exact["ray_count"].sum()), 4)
        self.assertEqual(int(below["ray_count"][:, 0].sum()), 4)
        self.assertEqual(int(below["ray_count"].sum()), 4)

    def test_hit_indices_exactly_match_normal_bev_mapping(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.999999, 1.0, 0.0],
                [2.25, 3.75, 1.0],
                [4.99999, 4.99999, -1.0],
            ],
            dtype=np.float32,
        )
        result = self.observations(points)
        _xyz, rows, cols, valid, _height, _width = metric_to_grid(
            points,
            self.x_range,
            self.y_range,
            self.resolution,
        )
        expected = np.zeros((5, 5), dtype=bool)
        expected[rows, cols] = True

        self.assertTrue(valid.all())
        np.testing.assert_array_equal(result["hit_mask"], expected)

    def test_confidence_uses_saturating_support_and_stays_bounded(self):
        ray_count = np.asarray([[0, 1, 4]], dtype=np.uint32)
        coverage = np.asarray([[1.0, 0.5, 1.0]], dtype=np.float32)
        result = compute_observability_confidence(
            ray_count, coverage, ray_support_tau=2.0
        )

        expected_support = 1.0 - np.exp(-ray_count.astype(np.float32) / 2.0)
        np.testing.assert_allclose(result["ray_support"], expected_support)
        np.testing.assert_allclose(
            result["observability_confidence"], coverage * expected_support
        )
        self.assertGreaterEqual(float(result["observability_confidence"].min()), 0.0)
        self.assertLessEqual(float(result["observability_confidence"].max()), 1.0)

    @unittest.skipIf(
        _compute_ray_observations_compiled is None,
        "Numba is not installed",
    )
    def test_compiled_backend_exactly_matches_reference_backend(self):
        rng = np.random.default_rng(731)
        random_points = rng.uniform(
            low=(-1.0, -1.0, -3.0),
            high=(6.0, 6.0, 5.0),
            size=(500, 3),
        ).astype(np.float32)
        boundary_points = np.asarray(
            [
                [0.0, 0.0, -2.0],
                [1.0, 1.0, 0.0],
                [4.99999, 4.99999, 3.99999],
                [5.0, 2.5, 1.0],
                [2.5, 5.0, -1.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        points = np.concatenate((random_points, boundary_points), axis=0)
        arguments = (
            points,
            self.origin,
            self.x_range,
            self.y_range,
            self.resolution,
        )

        compiled = compute_ray_observations(
            *arguments,
            z_range=(-2.0, 4.0),
            num_z_bins=16,
            use_compiled=True,
        )
        reference = compute_ray_observations(
            *arguments,
            z_range=(-2.0, 4.0),
            num_z_bins=16,
            use_compiled=False,
        )

        for key in (
            "ray_count",
            "observed_z_bins",
            "vertical_coverage",
            "range_map",
            "hit_mask",
        ):
            np.testing.assert_array_equal(compiled[key], reference[key])


if __name__ == "__main__":
    unittest.main()
