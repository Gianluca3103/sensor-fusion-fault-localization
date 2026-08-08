import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import PFS_Radar_v2.radar_data as radar_data
from PFS_Radar.radar_data import radar_cache_path
from PFS_Radar_v2.radar_data import (
    AdaptiveStackConfig,
    ClusterObservation,
    DopplerTrackingConfig,
    ProcessedFrame,
    TrackState,
    associate_tracks,
    compensate_doppler,
    dbscan_labels,
    project_adaptive_bev,
    select_adaptive_indices,
)


class PFSRadarV2Tests(unittest.TestCase):
    def test_shared_cache_lookup_supports_kradar_metadata(self):
        path = radar_cache_path(
            Path("cache"), {"sequence": "01", "radar_index": "33"}
        )
        self.assertEqual(path, Path("cache") / "1" / "00033.npz")

    def test_kradar_point_cache_reuses_overlapping_frame_reads(self):
        path = Path("rpc_00001.npy")
        expected = np.ones((2, 11), dtype=np.float32)
        radar_data.load_kradar_pc10p.cache_clear()
        with patch.object(
            radar_data.np,
            "load",
            return_value=expected,
        ) as reader:
            first = radar_data.load_kradar_pc10p(path)
            second = radar_data.load_kradar_pc10p(path)

        self.assertIs(first, second)
        reader.assert_called_once_with(path, allow_pickle=False)

    def test_kradar_wrapped_doppler_removes_velocity_alias(self):
        period = 4.0
        points = np.asarray([[10.0, 0.0, 0.0, 1.0]])
        residual, sign, expected = compensate_doppler(
            points,
            np.asarray([7.0, 0.0, 0.0]),
            doppler_sign="1",
            doppler_period_mps=period,
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

    def test_adaptive_stack_has_no_frame_cap_by_default(self):
        timestamps = [index * 10_000_000 for index in range(50)]
        poses = np.repeat(np.eye(4)[None], len(timestamps), axis=0)
        selected = select_adaptive_indices(
            timestamps,
            poses,
            lidar_timestamp=timestamps[-1],
            config=AdaptiveStackConfig(max_age_s=1.0),
        )
        self.assertEqual(len(selected), 50)
        self.assertEqual(
            [row["timestamp"] for row in selected],
            timestamps,
        )

    def test_adaptive_stack_respects_an_explicit_frame_cap(self):
        timestamps = [index * 10_000_000 for index in range(50)]
        poses = np.repeat(np.eye(4)[None], len(timestamps), axis=0)
        selected = select_adaptive_indices(
            timestamps,
            poses,
            lidar_timestamp=timestamps[-1],
            config=AdaptiveStackConfig(max_frames=20, max_age_s=1.0),
        )
        self.assertEqual(
            [row["timestamp"] for row in selected],
            timestamps[-20:],
        )

    def test_adaptive_stack_stops_at_translation_gate(self):
        timestamps = [0, 100_000_000, 200_000_000, 300_000_000]
        poses = np.repeat(np.eye(4)[None], len(timestamps), axis=0)
        poses[:, 0, 3] = [0.0, 1.0, 2.0, 3.0]
        selected = select_adaptive_indices(
            timestamps,
            poses,
            lidar_timestamp=300_000_000,
            config=AdaptiveStackConfig(
                max_frames=20,
                max_age_s=1.0,
                max_translation_m=1.5,
                max_rotation_deg=5.0,
            ),
        )
        self.assertEqual([row["timestamp"] for row in selected], timestamps[-2:])
        self.assertGreater(selected[-1]["weight"], selected[0]["weight"])

    def test_adaptive_stack_stops_at_rotation_gate(self):
        timestamps = [0, 100_000_000, 200_000_000]
        poses = np.repeat(np.eye(4)[None], len(timestamps), axis=0)
        for pose, angle_deg in zip(poses, [0.0, 4.0, 8.0]):
            angle = np.deg2rad(angle_deg)
            pose[:2, :2] = [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        selected = select_adaptive_indices(
            timestamps,
            poses,
            lidar_timestamp=200_000_000,
            config=AdaptiveStackConfig(max_rotation_deg=5.0),
        )
        self.assertEqual(
            [row["timestamp"] for row in selected],
            timestamps[-2:],
        )

    def test_ego_doppler_compensation_leaves_static_points_near_zero(self):
        points = np.asarray(
            [
                [10.0, 0.0, 0.0, -5.0],
                [0.0, 10.0, 0.0, 0.0],
            ]
        )
        residual, sign, expected = compensate_doppler(
            points,
            np.asarray([5.0, 0.0, 0.0]),
            doppler_sign="auto",
        )
        self.assertEqual(sign, 1)
        self.assertTrue(np.allclose(expected, [-5.0, 0.0]))
        self.assertTrue(np.allclose(residual, 0.0))

    def test_auto_doppler_sign_detects_reversed_driver_convention(self):
        points = np.asarray(
            [
                [10.0, 0.0, 0.0, 5.0],
                [8.0, 0.0, 0.0, 5.0],
                [0.0, 10.0, 0.0, 0.0],
            ]
        )
        residual, sign, _ = compensate_doppler(
            points,
            np.asarray([5.0, 0.0, 0.0]),
            doppler_sign="auto",
        )
        self.assertEqual(sign, -1)
        self.assertTrue(np.allclose(residual, 0.0))

    def test_dbscan_clusters_only_dynamic_candidates(self):
        xy = np.asarray(
            [[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 5.0], [0.1, 0.1]]
        )
        labels = dbscan_labels(
            xy,
            np.asarray([True, True, True, True, False]),
            eps_m=0.5,
            min_samples=2,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(labels[4], -1)

    def test_cluster_association_builds_a_causal_track(self):
        observations = [
            [
                ClusterObservation(
                    frame_index=0,
                    timestamp=0,
                    label=0,
                    point_indices=np.asarray([0, 1]),
                    centroid=np.asarray([0.0, 0.0]),
                    doppler_velocity=np.asarray([1.0, 0.0]),
                )
            ],
            [
                ClusterObservation(
                    frame_index=1,
                    timestamp=1_000_000_000,
                    label=0,
                    point_indices=np.asarray([0, 1]),
                    centroid=np.asarray([1.0, 0.0]),
                    doppler_velocity=np.asarray([1.0, 0.0]),
                )
            ],
        ]
        tracks = associate_tracks(
            observations,
            association_distance_m=1.0,
            velocity_smoothing=0.5,
        )
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].hits, 2)
        self.assertTrue(np.allclose(tracks[0].velocity, [1.0, 0.0]))
        self.assertEqual(observations[0][0].track_id, observations[1][0].track_id)

    def test_projection_outputs_occupancy_power_speed_and_upper_height(self):
        frame = ProcessedFrame(
            timestamp=0,
            points=np.asarray(
                [
                    [2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 2.0, 2.0, 1.4, 1.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            doppler_residual_mps=np.asarray([0.0, 2.0], dtype=np.float32),
            dynamic_mask=np.asarray([False, True]),
            cluster_labels=np.asarray([-1, 0], dtype=np.int32),
            weight=1.0,
            doppler_sign=1,
            sensor_speed_mps=0.0,
            yaw_rate_dps=0.0,
        )
        observation = ClusterObservation(
            frame_index=0,
            timestamp=0,
            label=0,
            point_indices=np.asarray([1]),
            centroid=np.asarray([1.0, 1.0]),
            doppler_velocity=np.asarray([1.0, 0.0]),
            track_id=0,
        )
        tracks = {
            0: TrackState(
                track_id=0,
                position=np.asarray([2.0, 1.0]),
                velocity=np.asarray([1.0, 0.0]),
                last_timestamp=1_000_000_000,
                hits=2,
            )
        }
        bev = project_adaptive_bev(
            [frame],
            [[observation]],
            tracks,
            lidar_timestamp=1_000_000_000,
            x_range=(0.0, 4.0),
            y_range=(-2.0, 2.0),
            resolution=1.0,
            tracking_config=DopplerTrackingConfig(),
        )
        self.assertEqual(bev.shape, (4, 4, 4))
        self.assertEqual(float(bev[0].sum()), 1.0)
        self.assertGreater(float(bev[1].max()), 0.0)
        self.assertGreater(float(bev[2].max()), 0.0)
        self.assertGreater(float(bev[3].max()), 0.0)
        self.assertTrue(np.all((bev >= 0.0) & (bev <= 1.0)))

    def test_single_frame_power_is_not_a_copy_of_static_occupancy(self):
        frame = ProcessedFrame(
            timestamp=0,
            points=np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0, 0.0, 4.0],
                ],
                dtype=np.float32,
            ),
            doppler_residual_mps=np.zeros(2, dtype=np.float32),
            dynamic_mask=np.zeros(2, dtype=bool),
            cluster_labels=np.full(2, -1, dtype=np.int32),
            weight=1.0,
            doppler_sign=1,
            sensor_speed_mps=0.0,
            yaw_rate_dps=0.0,
        )
        bev = project_adaptive_bev(
            [frame],
            [[]],
            {},
            lidar_timestamp=0,
            x_range=(0.0, 4.0),
            y_range=(-2.0, 2.0),
            resolution=1.0,
            tracking_config=DopplerTrackingConfig(),
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
