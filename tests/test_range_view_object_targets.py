import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry, project_lidar
from models.two_stage_reconstruction_head.range_view.loss import RangeLossConfig, range_edit_loss
from models.two_stage_reconstruction_head.range_view.object_targets import ground_return_mask, object_class_map
from models.two_stage_reconstruction_head.range_view.radar import radar_region_mask


class ObjectFocusedTargetsTests(unittest.TestCase):
    def test_ground_plane_target_excludes_raised_and_annotated_returns(self):
        x, y = np.meshgrid(np.linspace(2, 30, 30), np.linspace(-10, 10, 30))
        road = np.column_stack((x.ravel(), y.ravel(),
                                (-1.5 + 0.01 * x + 0.002 * y).ravel(),
                                np.ones(x.size))).astype(np.float32)
        raised = np.asarray([[8, 0, 1, 1]], dtype=np.float32)
        points = np.concatenate((road, raised))
        geometry = RangeGeometry(tuple(np.linspace(-0.7, 0.4, 128)), 512, 0.1, 50,
                                 2 * np.pi, max_beam_error_rad=0.02)
        projection = project_lidar(points, geometry)
        classes = np.zeros(geometry.shape, dtype=np.uint8)
        classes[projection.point_row[0], projection.point_col[0]] = 1
        ground = ground_return_mask(points, projection, classes)
        self.assertGreater(int(ground.sum()), 10)
        self.assertEqual(ground[projection.point_row[0], projection.point_col[0]], 0)
        self.assertEqual(ground[projection.point_row[-1], projection.point_col[-1]], 0)

    def test_missing_road_is_no_add_target_while_object_is_positive(self):
        logits = torch.zeros(1, 1, 3, requires_grad=True)
        prediction = {"add_logit": logits, "add_probability": torch.sigmoid(logits),
                      "add_range_m": torch.full((1, 1, 3), 5.0),
                      "delete_logit": torch.zeros(1, 1, 3)}
        target = {key: torch.zeros(1, 1, 3) for key in (
            "add", "add_range_m", "delete", "delete_valid", "clean_valid",
            "clean_range_m", "object_class", "radar_region", "ground_mask")}
        target["add"][:] = 1
        target["radar_region"][:] = 1
        target["object_class"][0, 0, 0] = 2
        target["ground_mask"][0, 0, 1] = 1
        loss = range_edit_loss(prediction, target, RangeLossConfig(
            radar_focused_objective=True, lambda_range=0, lambda_delete=0,
            lambda_free_space=0, lambda_intensity=0))
        loss["loss"].backward()
        self.assertGreater(abs(float(logits.grad[0, 0, 0])),
                           abs(float(logits.grad[0, 0, 2])))
        self.assertLess(float(logits.grad[0, 0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 0, 1]), 0)

    def test_vod_3d_boxes_label_clean_first_returns_only(self):
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
                "Pedestrian 0 0 0 0 0 0 0 2 1 1 8 2.5 0 0\n",
                encoding="utf-8")
            points = np.asarray([[5, 0, 0, 1], [8, 2, 0, 1], [12, 8, 0, 1]], dtype=np.float32)
            geometry = RangeGeometry((-0.1, 0.1), 32, 0.1, 50, 2 * np.pi)
            projection = project_lidar(points, geometry)
            classes = object_class_map(points, projection, {"source_relative_path": str(source)})
            self.assertEqual(classes[projection.point_row[0], projection.point_col[0]], 1)
            self.assertEqual(classes[projection.point_row[1], projection.point_col[1]], 2)
            self.assertEqual(classes[projection.point_row[2], projection.point_col[2]], 0)

    def test_radar_region_dilates_only_observed_radar_cells(self):
        radar = np.zeros((5, 9), dtype=np.float32)
        radar[2, 4] = 1
        region = radar_region_mask(radar, row_radius=1, col_radius=2)
        self.assertEqual(int(region.sum()), 15)
        self.assertEqual(region[2, 4], 1)
        self.assertEqual(region[0, 4], 0)

    def test_focused_loss_ignores_outside_radar_and_weights_object(self):
        logits = torch.full((1, 1, 3), -1.0, requires_grad=True)
        prediction = {
            "add_logit": logits,
            "add_probability": torch.sigmoid(logits),
            "add_range_m": torch.full((1, 1, 3), 5.0, requires_grad=True),
            "delete_logit": torch.zeros(1, 1, 3, requires_grad=True),
        }
        target = {key: torch.zeros(1, 1, 3) for key in (
            "add", "add_range_m", "delete", "delete_valid", "clean_valid",
            "clean_range_m", "object_class", "radar_region",
        )}
        target["add"][0, 0, :2] = 1
        target["add_range_m"][0, 0, :2] = 5
        target["clean_valid"][0, 0, :2] = 1
        target["clean_range_m"][0, 0, :2] = 5
        target["object_class"][0, 0, 0] = 2
        target["radar_region"][0, 0, :2] = 1
        focused = range_edit_loss(prediction, target, RangeLossConfig(
            radar_focused_objective=True, lambda_scanline=0.2))
        focused["loss"].backward()
        self.assertGreater(abs(float(logits.grad[0, 0, 0])), abs(float(logits.grad[0, 0, 1])))
        self.assertEqual(float(logits.grad[0, 0, 2]), 0)
        self.assertTrue(bool(torch.isfinite(focused["scanline_loss"])))


if __name__ == "__main__":
    unittest.main()
