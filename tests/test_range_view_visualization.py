"""Interactive range-view inspection uses consistent geometry and full PLY exports."""

from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from scripts.visualize_range_view_reconstruction import (
    _display_points, _load_annotated_boxes, _radar_box_stats, _render_comparison,
    _save_interactive_html, _save_ply,
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
            self.assertIn('id="show-boxes"', page)
            self.assertIn('id="view-tabs"', page)
            self.assertIn("zoomAt(newDistance/oldDistance", page)
            self.assertIn("two fingers to pan", page)
            self.assertIn('"radar":{"color":"#ffbf47","count":1', page)
            self.assertNotIn("https://", page)

    def test_vod_box_corners_align_with_lidar_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            partition = Path(temporary) / "lidar" / "training"
            for name in ("velodyne", "label_2", "calib"):
                (partition / name).mkdir(parents=True)
            source = partition / "velodyne" / "00001.bin"
            source.touch()
            (partition / "calib" / "00001.txt").write_text(
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
            (partition / "label_2" / "00001.txt").write_text(
                "Car 0 0 0 0 0 0 0 2 2 2 5 0.5 0 0\n"
                "Pedestrian 0 0 0 0 0 0 0 2 1 1 -8 0 0 0\n",
                encoding="utf-8")
            boxes = _load_annotated_boxes({
                "dataset": "View-of-Delft", "source_relative_path": str(source)})
            self.assertEqual(len(boxes), 1)  # The rear box is outside this forward viewer.
            self.assertEqual(boxes[0]["name"], "Car")
            np.testing.assert_allclose(boxes[0]["corners"].min(axis=0), [4, -1.5, -1])
            np.testing.assert_allclose(boxes[0]["corners"].max(axis=0), [6, 0.5, 1])

            radar = np.asarray([[5, 0, 0, 0], [4, -1.5, -1, 0],
                                [7, 0, 0, 0]], dtype=np.float32)
            stats = _radar_box_stats(radar, boxes + boxes)
            self.assertEqual((stats["radar_returns"], stats["inside_boxes"],
                              stats["outside_boxes"]), (3, 2, 1))
            self.assertAlmostEqual(stats["inside_percent"], 200 / 3)
            self.assertAlmostEqual(stats["outside_percent"], 100 / 3)

            theta = np.pi / 4
            rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta), np.cos(theta), 0],
                                   [0, 0, 1]])
            rotated = {**boxes[0], "corners": boxes[0]["corners"] @ rotation.T}
            rotated_radar = np.asarray([[5, 0, 0], [7, 0, 0]]) @ rotation.T
            self.assertEqual(_radar_box_stats(rotated_radar, [rotated])["inside_boxes"], 1)

            cloud = np.asarray([[5, 0, 0, 0.5]], dtype=np.float32)
            html = Path(temporary) / "boxes.html"
            _save_interactive_html(
                html, faulty=cloud, clean=cloud, original=cloud,
                generated=cloud[:0], sample_name="00001", fault="fog_sim",
                epoch=80, max_plot_points=10, boxes=boxes, radar_stats=stats)
            page = html.read_text(encoding="utf-8")
            self.assertIn('"boxes":[{"name":"Car","color":"#4ce0ed"', page)
            self.assertIn('"radar_stats":{"radar_returns":3,"inside_boxes":2', page)
            self.assertIn('id="radar-coverage"', page)
            self.assertNotIn("__BOX_EDGES__", page)


if __name__ == "__main__":
    unittest.main()
