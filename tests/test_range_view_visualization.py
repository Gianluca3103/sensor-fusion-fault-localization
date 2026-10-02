"""Interactive range-view inspection uses consistent geometry and full PLY exports."""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from scripts.visualize_range_view_reconstruction import (
    _display_points, _render_comparison, _save_interactive_html, _save_ply,
    _shared_bounds,
)


class RangeViewVisualizationTests(unittest.TestCase):
    def test_common_bounds_and_plot_cap(self) -> None:
        faulty = np.asarray([[1, -2, 0], [2, 3, 1], [3, 2, -1]], dtype=np.float32)
        clean = np.asarray([[8, 5, 2]], dtype=np.float32)
        bounds = _shared_bounds(faulty, clean)
        self.assertLess(bounds[0][0], 1)
        self.assertGreater(bounds[0][1], 8)
        self.assertLess(bounds[1][0], -2)
        self.assertGreater(bounds[1][1], 5)
        self.assertEqual(len(_display_points(faulty, 2)), 2)
        self.assertEqual(len(_display_points(clean, 2)), 1)

    def test_exports_full_ply_and_both_figures(self) -> None:
        faulty = np.asarray([[5, -1, 0, 0.5], [6, 0, 1, 0.7]], dtype=np.float32)
        clean = np.asarray([[5, -1, 0, 0.5], [8, 1, 2, 0.4]], dtype=np.float32)
        generated = np.asarray([[8, 1, 2, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _save_ply(output / "cloud.ply", faulty)
            data = (output / "cloud.ply").read_bytes()
            header, payload = data.split(b"end_header\n", 1)
            self.assertIn(b"element vertex 2", header)
            np.testing.assert_allclose(np.frombuffer(payload, dtype="<f4").reshape(-1, 4),
                                       faulty)
            _render_comparison(
                output, faulty=faulty, clean=clean, original=faulty,
                generated=generated, sample_name="sample", fault="fog_sim",
                epoch=15, max_plot_points=2, show=True,
            )
            self.assertTrue((output / "sample_rotatable_3d.png").is_file())
            self.assertTrue((output / "sample_xy_xz_yz.png").is_file())
            _save_interactive_html(
                output / "sample_interactive.html", faulty=faulty, clean=clean,
                original=faulty, generated=generated, sample_name="sample",
                fault="fog_sim", epoch=15, max_plot_points=2,
                radar=np.asarray([[6, 1, 0, 4, 0]], dtype=np.float32),
            )
            page = (output / "sample_interactive.html").read_text(encoding="utf-8")
            self.assertIn("pointermove", page)
            self.assertIn("Reconstructed LiDAR", page)
            self.assertIn('id="show-radar"', page)
            self.assertIn('"radar":{"color":"#ffbf47","count":1', page)
            self.assertNotIn("https://", page)


if __name__ == "__main__":
    unittest.main()
