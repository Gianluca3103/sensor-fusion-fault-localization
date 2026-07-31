from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from Fault_Localization_Model.fault_injector import (
    build_fault_plan,
    choose_samples,
    inject_fault,
    parse_fault_plan,
)


class FakeLidarCorruptions:
    @staticmethod
    def fov_filter(points, severity):
        return points[[0, 2]]


class FaultInjectorTests(unittest.TestCase):
    def test_parse_fault_plan(self):
        self.assertEqual(parse_fault_plan(["fog_sim:4", "rain_sim:5"]), [("fog_sim", 4), ("rain_sim", 5)])

    def test_parse_fault_plan_rejects_bad_items(self):
        with self.assertRaises(ValueError):
            parse_fault_plan(["fog_sim"])
        with self.assertRaises(ValueError):
            parse_fault_plan(["fog_sim:0"])
        with self.assertRaises(ValueError):
            parse_fault_plan(["not_a_fault:1"])

    def test_build_fault_plan_uses_defaults(self):
        plan = build_fault_plan(None, ["fog_sim", "fov_filter"], None, [("fog_sim", 4), ("fov_filter", 1)])
        self.assertEqual(plan, [("fog_sim", 4), ("fov_filter", 1)])

    def test_choose_samples_is_reproducible(self):
        bins = [Path("a.bin"), Path("b.bin"), Path("c.bin")]
        plan = [("fog_sim", 4), ("rain_sim", 5)]
        first = choose_samples(bins, 6, seed=7, plan=plan, shuffle=True)
        second = choose_samples(bins, 6, seed=7, plan=plan, shuffle=True)
        self.assertEqual(first, second)

    def test_choose_samples_draws_from_shuffled_candidate_pool(self):
        bins = [Path(f"{index}.bin") for index in range(20)]
        plan = [("fog_sim", 4)]
        samples = choose_samples(bins, 5, seed=7, plan=plan, shuffle=True)
        selected_bins = [sample[0] for sample in samples]
        self.assertNotEqual(selected_bins, bins[:5])

    def test_choose_samples_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            choose_samples([], 5, seed=7, plan=[("fog_sim", 4)])
        with self.assertRaises(ValueError):
            choose_samples([Path("a.bin")], 5, seed=7, plan=[])

    @patch("Fault_Localization_Model.fault_injector.apply_fog_simulator")
    def test_weather_replacement_gets_new_id_and_no_source(self, fog_simulator):
        clean = np.array(
            [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        fog_simulator.return_value = (
            np.column_stack([clean, np.array([1.0, 2.0], dtype=np.float32)]),
            {},
        )

        result, _ = inject_fault("fog_sim", clean, np.array([0, 1]), 5, Path("."), Path("."), 10, None)

        np.testing.assert_array_equal(result.source_ids, np.array([0, -1]))
        self.assertEqual(result.point_ids[0], 0)
        self.assertGreaterEqual(result.point_ids[1], 2)

    def test_fov_filter_applies_same_subset_to_ids(self):
        clean = np.array(
            [[1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0], [-1.0, 1.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        ids = np.array([10, 11, 12], dtype=np.int64)

        result, _ = inject_fault(
            "fov_filter",
            clean,
            ids,
            1,
            Path("."),
            Path("."),
            10,
            FakeLidarCorruptions,
        )

        self.assertEqual(len(result.points), len(result.source_ids))
        np.testing.assert_array_equal(result.point_ids, result.source_ids)
        np.testing.assert_array_equal(result.source_ids, np.array([10, 12]))

    def test_fov_filter_uses_literature_subset_deterministically(self):
        angles = np.deg2rad(np.arange(-180.0, 180.0, 2.0))
        clean = np.column_stack(
            [
                np.sin(angles),
                np.cos(angles),
                np.zeros_like(angles),
                np.ones_like(angles),
            ]
        ).astype(np.float32)
        ids = np.arange(len(clean), dtype=np.int64)

        first, first_meta = inject_fault(
            "fov_filter",
            clean,
            ids,
            1,
            Path("."),
            Path("."),
            10,
            FakeLidarCorruptions,
            rng_seed=100,
        )
        repeated, repeated_meta = inject_fault(
            "fov_filter",
            clean,
            ids,
            1,
            Path("."),
            Path("."),
            10,
            FakeLidarCorruptions,
            rng_seed=100,
        )
        different, different_meta = inject_fault(
            "fov_filter",
            clean,
            ids,
            1,
            Path("."),
            Path("."),
            10,
            FakeLidarCorruptions,
            rng_seed=101,
        )

        np.testing.assert_array_equal(first.source_ids, repeated.source_ids)
        np.testing.assert_array_equal(first.source_ids, different.source_ids)
        self.assertEqual(first_meta["fov_source"], repeated_meta["fov_source"])
        self.assertEqual(first_meta["fov_source"], different_meta["fov_source"])

    def test_old_laser_subset_preserves_source_ids(self):
        clean = np.array(
            [[float(index + 1), 1.0, 0.0, 1.0] for index in range(20)],
            dtype=np.float32,
        )
        ids = np.arange(20, dtype=np.int64)

        result, _ = inject_fault(
            "old_laser_degradation", clean, ids, 0, Path("."), Path("."), 10, None
        )

        self.assertEqual(len(result.points), len(result.source_ids))
        np.testing.assert_array_equal(result.point_ids, result.source_ids)
        self.assertTrue(set(result.source_ids).issubset(set(ids)))

    def test_old_laser_seed_controls_sample_diversity(self):
        clean = np.array(
            [
                [float(index + 1), 1.0, 0.0, float(index + 1)]
                for index in range(200)
            ],
            dtype=np.float32,
        )
        ids = np.arange(len(clean), dtype=np.int64)
        first, _ = inject_fault(
            "old_laser_degradation",
            clean,
            ids,
            0,
            Path("."),
            Path("."),
            10,
            None,
            rng_seed=50,
        )
        repeated, _ = inject_fault(
            "old_laser_degradation",
            clean,
            ids,
            0,
            Path("."),
            Path("."),
            10,
            None,
            rng_seed=50,
        )
        different, _ = inject_fault(
            "old_laser_degradation",
            clean,
            ids,
            0,
            Path("."),
            Path("."),
            10,
            None,
            rng_seed=51,
        )
        np.testing.assert_array_equal(first.source_ids, repeated.source_ids)
        self.assertFalse(np.array_equal(first.source_ids, different.source_ids))


if __name__ == "__main__":
    unittest.main()
