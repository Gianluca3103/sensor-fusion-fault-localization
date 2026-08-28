import unittest

from tools.benchmark_fine_diffusion_inference import _distribution, _percentile


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


if __name__ == "__main__":
    unittest.main()
