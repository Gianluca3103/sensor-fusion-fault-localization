import unittest
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import PFS_Radar_v2.radar_data as radar_data
from PFS_Radar.radar_data import radar_cache_path
from PFS_Radar_v2.pose import pose_velocity
from PFS_Radar_v2.radar_data import DopplerConfig, compensate_doppler, project_radar_bev


class PFSRadarV2Tests(unittest.TestCase):
    def test_cache_contract_requires_aligned_pointpillars_radar_points(self):
        config = DopplerConfig()
        metadata = {
            "cache_format_version": radar_data.RADAR_CACHE_VERSION,
            "policy": radar_data.POLICY_NAME,
            "channels": radar_data.CHANNELS,
            "sequence": "1",
            "radar_index": "00033",
            "lidar_index": "00001",
            "timestamp_ns": 123,
            "x_range": [0.0, 64.0],
            "y_range": [-32.0, 32.0],
            "resolution": 0.2,
            "doppler": asdict(config),
        }
        compatibility = dict(
            sequence="1",
            radar_index="33",
            lidar_index="1",
            timestamp=123,
            x_range=(0.0, 64.0),
            y_range=(-32.0, 32.0),
            resolution=0.2,
            doppler_config=config,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez_compressed(
                path,
                radar_bev=np.zeros((4, 320, 320), dtype=np.float16),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            self.assertFalse(
                radar_data.radar_cache_is_compatible(path, **compatibility)
            )
            np.savez_compressed(
                path,
                radar_bev=np.zeros((4, 320, 320), dtype=np.float16),
                radar_points=np.zeros((2, 5), dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            self.assertTrue(
                radar_data.radar_cache_is_compatible(path, **compatibility)
            )

    def test_shared_cache_lookup_supports_kradar_metadata(self):
        path = radar_cache_path(
            Path("cache"), {"sequence": "01", "radar_index": "33"}
        )
        self.assertEqual(path, Path("cache") / "1" / "00033.npz")

    def test_kradar_point_cache_reuses_a_frame_read(self):
        path = Path("rpc_00001.npy")
        expected = np.ones((2, 11), dtype=np.float32)
        radar_data.load_kradar_pc10p.cache_clear()
        with patch.object(radar_data.np, "load", return_value=expected) as reader:
            first = radar_data.load_kradar_pc10p(path)
            second = radar_data.load_kradar_pc10p(path)
        self.assertIs(first, second)
        reader.assert_called_once_with(path, allow_pickle=False)

    def test_kradar_wrapped_doppler_removes_velocity_alias(self):
        residual, sign, expected = compensate_doppler(
            np.asarray([[10.0, 0.0, 0.0, 1.0]]),
            np.asarray([7.0, 0.0, 0.0]),
            doppler_sign="1",
            doppler_period_mps=4.0,
        )
        self.assertEqual(sign, 1)
        self.assertTrue(np.allclose(expected, [-7.0]))
        self.assertTrue(np.allclose(residual, [0.0]))

    def test_kradar_label_pair_and_calibration_are_parsed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            label = root / "00033_00001.txt"
            label.write_text(
                "* idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)="
                "00033_00001_00002_00001_00004, timestamp=1643292946.710046076\n",
                encoding="utf-8",
            )
            calibration = root / "calib_radar_lidar.txt"
            calibration.write_text(
                "frame difference,x,y\n32,-2.54,0.3\n", encoding="utf-8"
            )
            radar_index, lidar_index, timestamp = radar_data.parse_label_frame(label)
            transform = radar_data.load_lidar_from_radar_transform(calibration)
        self.assertEqual((radar_index, lidar_index), ("00033", "00001"))
        self.assertEqual(timestamp, 1_643_292_946_710_046_076)
        self.assertTrue(np.allclose(transform[:3, 3], [2.54, -0.3, -0.7]))

    def test_pose_velocity_uses_neighboring_odometry_only_for_ego_doppler(self):
        timestamps = (0, 1_000_000_000, 2_000_000_000)
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = [0.0, 2.0, 4.0]
        velocity, yaw_rate = pose_velocity(timestamps, poses, timestamps[1])
        self.assertTrue(np.allclose(velocity, [2.0, 0.0, 0.0]))
        self.assertEqual(yaw_rate, 0.0)

    def test_ego_doppler_compensation_leaves_static_points_near_zero(self):
        points = np.asarray(
            [[10.0, 0.0, 0.0, -5.0], [0.0, 10.0, 0.0, 0.0]]
        )
        residual, sign, expected = compensate_doppler(
            points, np.asarray([5.0, 0.0, 0.0]), doppler_sign="auto"
        )
        self.assertEqual(sign, 1)
        self.assertTrue(np.allclose(expected, [-5.0, 0.0]))
        self.assertTrue(np.allclose(residual, 0.0))

    def test_auto_doppler_sign_detects_reversed_driver_convention(self):
        points = np.asarray(
            [[10.0, 0.0, 0.0, 5.0], [8.0, 0.0, 0.0, 5.0], [0.0, 10.0, 0.0, 0.0]]
        )
        residual, sign, _ = compensate_doppler(
            points, np.asarray([5.0, 0.0, 0.0]), doppler_sign="auto"
        )
        self.assertEqual(sign, -1)
        self.assertTrue(np.allclose(residual, 0.0))

    def test_projection_outputs_occupancy_power_speed_and_upper_height(self):
        points = np.asarray(
            [
                [2.0, 0.0, 1.0, 0.0, 2.0],
                [1.0, 1.0, 2.0, 2.0, 1.4],
            ],
            dtype=np.float32,
        )
        bev = project_radar_bev(
            points,
            np.asarray([0.0, 2.0], dtype=np.float32),
            np.asarray([False, True]),
            x_range=(0.0, 4.0),
            y_range=(-2.0, 2.0),
            resolution=1.0,
            config=DopplerConfig(),
        )
        self.assertEqual(bev.shape, (4, 4, 4))
        self.assertEqual(float(bev[0].sum()), 1.0)
        self.assertGreater(float(bev[1].max()), 0.0)
        self.assertGreater(float(bev[2].max()), 0.0)
        self.assertGreater(float(bev[3].max()), 0.0)
        self.assertTrue(np.all((bev >= 0.0) & (bev <= 1.0)))

    def test_single_frame_power_is_not_a_copy_of_static_occupancy(self):
        points = np.asarray(
            [[1.0, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 1.0, 0.0, 4.0]],
            dtype=np.float32,
        )
        bev = project_radar_bev(
            points,
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=bool),
            x_range=(0.0, 4.0),
            y_range=(-2.0, 2.0),
            resolution=1.0,
            config=DopplerConfig(),
        )
        occupied_power = bev[1][bev[0] > 0]
        self.assertEqual(float(bev[0].sum()), 2.0)
        self.assertEqual(np.unique(occupied_power).size, 2)
        self.assertFalse(np.array_equal(bev[0], bev[1]))

    def test_upper_height_ignores_implausible_elevation_returns(self):
        grid = radar_data._upper_height_grid(
            np.asarray([0, 0, 0]),
            np.asarray([1.0, 2.0, 20.0], dtype=np.float32),
            (1, 1),
        )
        self.assertAlmostEqual(float(grid[0, 0]), 5.0 / 8.0)


if __name__ == "__main__":
    unittest.main()
