import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel, parameter_breakdown
from PFS_Radar.radar_data import (
    CONTINENTAL_DTYPE,
    RADAR_CACHE_VERSION,
    historical_radar_frames,
    load_ground_truth_poses,
    pose_matrix,
    project_radar_bev,
    radar_cache_path,
    radar_cache_is_compatible,
    read_continental_bin,
    scene_session_root,
)
from PFS_Radar.train_pfs_radar import euclidean_dilate, localization_surrogate_loss


class PFSRadarTests(unittest.TestCase):
    def test_pose_matrix_uses_xyzw_quaternion_order(self):
        half_sqrt = np.sqrt(0.5)
        transform = pose_matrix(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0.0, 0.0, half_sqrt, half_sqrt]),
        )
        point = transform @ np.asarray([1.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(point[:3], [1.0, 3.0, 3.0]))

    def test_radar_projection_uses_expected_channels(self):
        points = np.asarray(
            [
                [10.0, 0.0, 0.0, -5.0, 10.0, 128.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 8.0, 10.0, 255.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        bev = project_radar_bev(
            points,
            np.eye(4),
            (0.0, 64.0),
            (-32.0, 32.0),
            0.2,
        )
        self.assertEqual(bev.shape, (4, 320, 320))
        self.assertEqual(bev[0].sum(), 1.0)
        self.assertTrue(np.isclose(bev[1].max(), 1.0))
        self.assertTrue(np.isclose(bev[2].max(), 8.0 / 30.0))
        self.assertTrue(np.isclose(bev[3].max(), 1.0))

    def test_malformed_radar_binary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "radar.bin"
            path.write_bytes(b"x" * (CONTINENTAL_DTYPE.itemsize + 1))
            with self.assertRaises(ValueError):
                read_continental_bin(path)

    def test_radar_cache_and_source_roots_include_session(self):
        metadata = {
            "scene": "Scene01",
            "session": "01_Day",
            "timestamp": "123",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "Scene01" / "01_Day"
            source.mkdir(parents=True)
            self.assertEqual(scene_session_root(root, metadata), source)
            self.assertEqual(
                radar_cache_path(root / "cache", metadata),
                root / "cache" / "Scene01" / "01_Day" / "123.npz",
            )

        with self.assertRaises(ValueError):
            radar_cache_path(
                Path("cache"),
                {**metadata, "scene": "../outside"},
            )

    def test_unsorted_ground_truth_timestamps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.txt"
            path.write_text(
                "2 0 0 0 0 0 0 1\n1 0 0 0 0 0 0 1\n",
                encoding="utf-8",
            )
            load_ground_truth_poses.cache_clear()
            with self.assertRaises(ValueError):
                load_ground_truth_poses(str(path))

    def test_historical_stack_never_selects_future_radar(self):
        timestamps = (100_000_000, 140_000_000)
        paths = ("100000000.bin", "140000000.bin")
        with patch(
            "PFS_Radar.radar_data.scene_radar_resources",
            return_value=(timestamps, paths, np.eye(4)),
        ):
            selected, delta_ms = historical_radar_frames(
                Path("scene"),
                lidar_timestamp=130_000_000,
                frame_count=1,
                max_delta_ms=50.0,
            )
        self.assertEqual(selected, [Path("100000000.bin")])
        self.assertEqual(delta_ms, -30.0)

    def test_radar_cache_compatibility_checks_stack_and_shape(self):
        sample_metadata = {
            "scene": "Scene01",
            "timestamp": "123",
            "x_range": [0.0, 4.0],
            "y_range": [-2.0, 2.0],
            "resolution": 1.0,
        }
        cache_metadata = {
            "scene": "Scene01",
            "lidar_timestamp": "123",
            "radar_delta_ms": 10.0,
            "radar_frame_count": 20,
            "requested_radar_frame_count": 20,
            "x_range": [0.0, 4.0],
            "y_range": [-2.0, 2.0],
            "resolution": 1.0,
            "max_abs_velocity": 30.0,
            "cache_format_version": RADAR_CACHE_VERSION,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez_compressed(
                path,
                radar_bev=np.zeros((4, 4, 4), dtype=np.float16),
                metadata_json=np.asarray(json.dumps(cache_metadata)),
            )
            self.assertTrue(
                radar_cache_is_compatible(
                    path,
                    sample_metadata,
                    max_delta_ms=30.0,
                    max_abs_velocity=30.0,
                    radar_frame_count=20,
                    require_full_stack=True,
                )
            )
            self.assertFalse(
                radar_cache_is_compatible(
                    path,
                    sample_metadata,
                    radar_frame_count=1,
                )
            )

    def test_model_shapes_and_parameter_accounting(self):
        model = PFSRadarReliabilityModel(base_channels=4)
        model.eval()
        with torch.no_grad():
            output = model(
                torch.zeros(1, 3, 64, 64),
                torch.zeros(1, 4, 64, 64),
                return_features=True,
            )
        self.assertEqual(output["logits"].shape, (1, 1, 64, 64))
        self.assertEqual(output["pfs_reliability"].shape, (1, 1, 4, 4))
        breakdown = parameter_breakdown(model)
        self.assertEqual(
            breakdown["total"],
            sum(value for key, value in breakdown.items() if key != "total"),
        )

    def test_production_model_parameter_count_is_unchanged(self):
        model = PFSRadarReliabilityModel(base_channels=16)
        self.assertEqual(
            parameter_breakdown(model),
            {
                "lidar_encoder": 1_179_760,
                "radar_encoder": 1_179_904,
                "fusion": 131_584,
                "pfs": 3_351_686,
                "radar_skip_fusion": 44_000,
                "decoder": 762_817,
                "total": 6_649_751,
            },
        )

    def test_localization_loss_penalizes_broad_predictions(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 7:9, 7:9] = 1.0
        correct = torch.full_like(target, -6.0)
        correct[:, :, 7:9, 7:9] = 6.0
        broad = torch.full_like(target, 2.0)

        correct_loss = localization_surrogate_loss(
            correct,
            target,
            tolerance_m=1.0,
            x_cell_size_m=1.0,
            y_cell_size_m=1.0,
        )
        broad_loss = localization_surrogate_loss(
            broad,
            target,
            tolerance_m=1.0,
            x_cell_size_m=1.0,
            y_cell_size_m=1.0,
        )
        self.assertGreater(broad_loss, correct_loss)

    def test_localization_loss_has_finite_gradients(self):
        logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, :, 4:8, 4:8] = 1.0

        loss = localization_surrogate_loss(
            logits,
            target,
            tolerance_m=1.0,
            x_cell_size_m=1.0,
            y_cell_size_m=1.0,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_euclidean_dilation_respects_twenty_centimeter_radius(self):
        values = torch.zeros(1, 1, 5, 5)
        values[:, :, 2, 2] = 1.0
        dilated = euclidean_dilate(
            values,
            tolerance_m=0.20,
            x_cell_size_m=0.20,
            y_cell_size_m=0.20,
        )

        self.assertEqual(dilated[0, 0, 2, 2], 1.0)
        self.assertEqual(dilated[0, 0, 1, 2], 1.0)
        self.assertEqual(dilated[0, 0, 2, 1], 1.0)
        self.assertEqual(dilated[0, 0, 1, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
