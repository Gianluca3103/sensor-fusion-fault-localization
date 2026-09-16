import tempfile
import unittest
import json
from unittest.mock import patch
from pathlib import Path
import numpy as np

from Fault_Localization_Model.hercules_dataset import (
    CONTINENTAL_DTYPE, discover_hercules_frames, load_hercules_lidar,
    load_frame_radar, sensor_pose,
)
from models.two_stage_reconstruction_head.coarse_dataset import radar_cache_path
from Fault_Localization_Model.hercules_tracking import compensate_doppler


class HerculesDatasetTests(unittest.TestCase):
    def test_exact_pose_and_jittered_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'poses.txt'
            path.write_text('1000000000 1 0 0 0 0 0 1\n'
                            '1124000000 2 0 0 0 0 0 1\n'
                            '2024000000 3 0 0 0 0 0 1\n')
            pose, velocity = sensor_pose(path, 1124000000)
            self.assertEqual(pose[0, 3], 2)
            self.assertAlmostEqual(velocity[0], 1/.124)
            pose, _ = sensor_pose(path, 1062000000)
            self.assertAlmostEqual(pose[0, 3], 1.5)
            with self.assertRaisesRegex(ValueError, '900.000 ms'):
                sensor_pose(path, 1500000000)
            with self.assertRaisesRegex(ValueError, '124.000 ms'):
                sensor_pose(path, 1062000000, max_gap_s=.1)

    def make_session(self, root):
        session = root / 'day' / 'session'
        aeva = session / 'LiDAR' / 'Aeva'
        radar = session / 'Radar' / 'Continental'
        aeva.mkdir(parents=True)
        radar.mkdir(parents=True)
        for i in range(10):
            row = np.zeros((1, 29), dtype=np.uint8)
            row[:, :16] = np.array([[5., 0., 0., 42.]], dtype='<f4').view(np.uint8)
            row.tofile(aeva / f'{1_000_000_000+i*10_000_000}.bin')
        for timestamp in (1_000_000_000, 1_010_000_000, 1_100_000_000):
            row = np.zeros(1, dtype=CONTINENTAL_DTYPE)
            row['x'], row['range'], row['rcs'] = 5, 5, 30
            row.tofile(radar / f'{timestamp}.bin')
        identity = '1 0 0 0 0 1 0 0 0 0 1 0'
        (session / 'IMU_LiDAR.txt').write_text('Tr_lidar_to_imu: '+identity)
        (session / 'Continental_LiDAR.txt').write_text('Tr_lidar_to_radar: '+identity)
        for name in ('Aeva_gt.txt', 'Continental_gt.txt'):
            (session / name).write_text('1000000000 0 0 0 0 0 0 1\n1100000000 0 0 0 0 0 0 1\n')
        return session

    def test_current_contract_and_causal_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            frames = discover_hercules_frames(root, 'train')
            self.assertEqual(len(frames), 7)
            self.assertEqual(len(discover_hercules_frames(root, 'val')), 1)
            self.assertEqual(len(discover_hercules_frames(root, 'test')), 2)
            self.assertEqual(load_hercules_lidar(frames[1].lidar_path).tolist(), [[5., 0., 0., 42.]])
            _, radar, _ = load_frame_radar(frames[1], {
                'hercules_radar_frames': 20, 'hercules_temporal_radius': .75,
            })
            self.assertEqual(radar.shape, (2, 7))
            self.assertEqual(radar[:, 6].tolist(), [-1., 0.])
            self.assertEqual(radar_cache_path(root, {'dataset': 'HeRCULES', 'split': 'train', 'frame_id': '1'}), root / 'train' / '00001.npz')

    def test_malformed_records_and_pose_extrapolation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.make_session(root)
            bad = root / 'bad.bin'
            bad.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                load_hercules_lidar(bad)
            with self.assertRaises(ValueError):
                sensor_pose(session / 'Aeva_gt.txt', 900_000_000)

    def test_configurable_causal_radar_age(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            frame = discover_hercules_frames(root, 'train')[6]
            config = {'hercules_radar_frames': 20, 'hercules_temporal_radius': .75}
            with self.assertRaisesRegex(ValueError, '50.00 ms old'):
                load_frame_radar(frame, config)
            config['hercules_max_radar_age_ms'] = 75
            load_frame_radar(frame, config)
            self.assertEqual(config['_hercules_alignment']['newest_radar_age_ms'], 50)
            self.assertTrue(all(row['timestamp_ns'] <= int(frame.lidar_path.stem)
                                for row in config['_hercules_alignment']['alignment_rows']))
            config['hercules_max_radar_age_ms'] = float('nan')
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                load_frame_radar(frame, config)

    def test_v2_auto_doppler_sign(self):
        points = np.array([[5., 0., 0., 2.], [6., 0., 0., 2.]])
        residual, sign, _ = compensate_doppler(points, np.array([2., 0., 0.]))
        self.assertEqual(sign, -1)
        np.testing.assert_allclose(residual, 0.)

    def test_v2_translation_gate_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.make_session(root)
            (session / 'Continental_gt.txt').write_text('1000000000 0 0 0 0 0 0 1\n1100000000 10 0 0 0 0 0 1\n')
            frame = discover_hercules_frames(root, 'train')[1]
            config = {'hercules_radar_frames': 0, 'hercules_temporal_radius': .75,
                      'hercules_stack': {'max_translation_m': .5}}
            _, points, _ = load_frame_radar(frame, config)
            self.assertEqual(len(config['_hercules_alignment']['alignment_rows']), 1)
            self.assertEqual(points[:, 6].tolist(), [0.])
            self.assertEqual(config['_hercules_point_weights'].tolist(), [1.])

    def test_v2_tracks_advance_dynamic_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            frame = discover_hercules_frames(root, 'train')[1]
            def native(path):
                x = 5. if int(path.stem) == 1_000_000_000 else 5.02
                return np.array([[x, 0., 0., 2., x, 30., 0., 0.],
                                 [x, .1, 0., 2., x, 30., 0., 0.]], dtype=np.float32)
            config = {'hercules_radar_frames': 20, 'hercules_temporal_radius': .75}
            with patch('Fault_Localization_Model.hercules_dataset.load_continental', side_effect=native):
                _, points, _ = load_frame_radar(frame, config)
            self.assertEqual(config['_hercules_alignment']['confirmed_tracks'], 1)
            self.assertEqual(config['_hercules_alignment']['motion_compensated_points'], 4)
            self.assertGreater(float(points[0, 0]), 5.)
            self.assertAlmostEqual(float(points[-1, 0]), 5.02, places=5)

    def test_ego_translation_alignment_and_cache_contract(self):
        from Fault_Localization_Model.create_vod_reconstruction_dataset import _write_radar_cache
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = self.make_session(root)
            pose_text = '1000000000 0 0 0 0 0 0 1\n1100000000 .2 0 0 0 0 0 1\n'
            (session / 'Aeva_gt.txt').write_text(pose_text)
            (session / 'Continental_gt.txt').write_text(pose_text)
            frame = discover_hercules_frames(root, 'train')[1]
            config = {'dataset': 'HeRCULES', 'hercules_radar_frames': 20,
                      'hercules_temporal_radius': .75, 'radar_cache_root': str(root / 'cache'),
                      'x_min': 0., 'x_max': 64., 'y_min': -32., 'y_max': 32.,
                      'resolution': .2, 'bev_channel_profile': 'baseline'}
            raw, points, _ = load_frame_radar(frame, config)
            self.assertAlmostEqual(float(points[0, 0]), 4.98, places=5)
            self.assertAlmostEqual(float(points[-1, 0]), 5., places=5)
            path = _write_radar_cache(frame, raw, points, config)
            with np.load(path, allow_pickle=False) as cache:
                self.assertEqual(cache['radar_points'].shape[1], 5)
                self.assertEqual(len(cache['radar_points']), len(cache['radar_point_weights']))
                self.assertTrue(np.isfinite(cache['radar_bev']).all())
                metadata = json.loads(str(cache['metadata_json'].item()))
                self.assertEqual(metadata['dataset'], 'HeRCULES')
                self.assertTrue(metadata['hercules_alignment']['policy'].startswith('hercules_v2'))

    def test_artifacts_feed_both_current_pointpillars_encoders(self):
        import torch
        from models.two_stage_reconstruction_head.coarse_dataset import load_bev_triplet
        from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry, PointPillarsEncoder
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            frame = discover_hercules_frames(root, 'train')[1]
            _, radar, _ = load_frame_radar(frame, {'hercules_radar_frames': 20, 'hercules_temporal_radius': .75})
            cache = root / 'cache' / 'train'
            cache.mkdir(parents=True)
            np.savez(cache / '00001.npz', radar_bev=np.zeros((4, 320, 320), dtype=np.float32),
                     radar_points=radar[:, [0, 1, 2, 3, 5]])
            sample = root / 'sample.npz'
            np.savez(sample, clean_rgb=np.zeros((320, 320, 3), dtype=np.uint8),
                     faulty_rgb=np.zeros((320, 320, 3), dtype=np.uint8),
                     faulty_lidar_points=load_hercules_lidar(frame.lidar_path),
                     metadata_json=json.dumps({'dataset': 'HeRCULES', 'split': 'train', 'frame_id': '1'}))
            item = load_bev_triplet(sample, root / 'cache', include_pointpillars_inputs=True)
            geometry = BEVGridGeometry(0., 64., -32., 32., 320, 320)
            with torch.no_grad():
                for key, channels in [('faulty_lidar_points', 4), ('radar_points', 5)]:
                    encoder = PointPillarsEncoder(geometry, raw_channels=channels,
                        output_channels=8, max_points_per_pillar=32, max_pillars=None).eval()
                    features, _ = encoder([item[key]])
                    self.assertEqual(tuple(features.shape), (1, 8, 320, 320))
                    self.assertTrue(torch.isfinite(features).all())


if __name__ == '__main__':
    unittest.main()
