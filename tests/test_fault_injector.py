from pathlib import Path
import sys
import unittest

MODEL_DIR = Path(__file__).resolve().parents[1] / "Fault_Localization_Model"
sys.path.insert(0, str(MODEL_DIR))

from fault_injector import build_fault_plan, choose_samples, parse_fault_plan


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


if __name__ == "__main__":
    unittest.main()
