import unittest

from tools.benchmark_fine_diffusion_inference import _distribution, _percentile
from tools.summarize_fine_inference_benchmarks import _fairness_mismatches


class FineDiffusionInferenceBenchmarkTests(unittest.TestCase):
    def test_percentile_interpolates_ordered_values(self):
        self.assertEqual(_percentile([40.0, 10.0, 30.0, 20.0], 50.0), 25.0)
        self.assertEqual(_percentile([40.0, 10.0, 30.0, 20.0], 0.0), 10.0)
        self.assertEqual(_percentile([40.0, 10.0, 30.0, 20.0], 100.0), 40.0)

    def test_distribution_reports_latency_statistics(self):
        result = _distribution([1.0, 2.0, 3.0])
        self.assertEqual(result["mean_ms"], 2.0)
        self.assertEqual(result["p50_ms"], 2.0)
        self.assertEqual(result["min_ms"], 1.0)
        self.assertEqual(result["max_ms"], 3.0)

    def test_fairness_check_allows_only_sampling_step_difference(self):
        first = {
            "comparison_signature": {
                "sampling_steps": 1,
                "sample_fingerprint_sha256": "same",
                "fine_pointpillars_conditioning": True,
                "batch_size": 1,
            }
        }
        second = {
            "comparison_signature": {
                "sampling_steps": 3,
                "sample_fingerprint_sha256": "same",
                "fine_pointpillars_conditioning": True,
                "batch_size": 1,
            }
        }

        self.assertEqual(_fairness_mismatches([first, second]), [])

        second["comparison_signature"]["fine_pointpillars_conditioning"] = False
        self.assertEqual(
            _fairness_mismatches([first, second]),
            ["fine_pointpillars_conditioning"],
        )


if __name__ == "__main__":
    unittest.main()
