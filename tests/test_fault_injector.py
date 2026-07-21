from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "Fault_Localization_Model"
sys.path.insert(0, str(MODEL_DIR))

from fault_injector import build_fault_plan, choose_samples, inject_fault, parse_fault_plan


class FaultInjectorTests(unittest.TestCase):
    def test_parse_fault_plan(self):
        self.assertEqual(parse_fault_plan(["fog_sim:4", "rain_sim:5"]), [("fog_sim", 4), ("rain_sim", 5)])

    def test_parse_fault_plan_rejects_bad_items(self):
        with self.assertRaises(ValueError):
            parse_fault_plan(["fog_sim"])

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

    @patch("fault_injector.apply_fog_simulator")
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

        result, _ = inject_fault("fov_filter", clean, ids, 1, Path("."), Path("."), 10, None)

        self.assertEqual(len(result.points), len(result.source_ids))
        np.testing.assert_array_equal(result.point_ids, result.source_ids)
        self.assertTrue(set(result.source_ids).issubset(set(ids)))

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


if __name__ == "__main__":
    unittest.main()
