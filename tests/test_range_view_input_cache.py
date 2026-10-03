import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from models.two_stage_reconstruction_head.range_view.data import RangeViewDataset
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.range_view.input_cache import (
    cache_manifest, cache_settings, load_cached_tensors, validate_cache, write_cached_sample,
)


class RangeViewInputCacheTests(unittest.TestCase):
    def test_object_target_cache_matches_online_and_tracks_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            partition = root / "lidar" / "training"
            for name in ("velodyne", "label_2", "calib"):
                (partition / name).mkdir(parents=True)
            clean_path = partition / "velodyne" / "00001.bin"
            np.asarray([[5, 0, 0, 0.7], [8, 2, 0, 0.2]], dtype=np.float32).tofile(clean_path)
            (partition / "label_2" / "00001.txt").write_text(
                "Car 0 0 0 0 0 0 0 2 2 2 5 0.5 0 0\n", encoding="utf-8")
            (partition / "calib" / "00001.txt").write_text(
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
            sample = root / "samples" / "train" / "00001.npz"
            sample.parent.mkdir(parents=True)
            metadata = {"dataset": "View-of-Delft", "range_view_full_scan": True,
                        "split": "train", "frame_id": "00001",
                        "source_relative_path": str(clean_path)}
            np.savez(sample, metadata_json=json.dumps(metadata),
                     faulty_lidar_points=np.empty((0, 4), dtype=np.float32),
                     faulty_source_ids=np.empty(0, dtype=np.int64))
            radar_root = root / "radar"
            (radar_root / "train").mkdir(parents=True)
            np.savez(radar_root / "train" / "00001.npz",
                     radar_points=np.asarray([[5, 0, 0, 1, 2]], dtype=np.float32))
            geometry = RangeGeometry((-0.1, 0.1), 32, 0.1, 50, 2 * np.pi)
            settings = cache_settings(geometry, require_lidar_intensity=True,
                                      include_object_targets=True,
                                      radar_region_row_radius=1,
                                      radar_region_col_radius=2)
            cache_root = root / "cache"
            self.assertTrue(write_cached_sample(sample, radar_root, geometry, cache_root, settings))
            manifest = cache_manifest([sample], radar_root, settings)
            (cache_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            validate_cache(cache_root, [sample], radar_root, settings)
            direct = RangeViewDataset([sample], radar_root, geometry,
                                      require_lidar_intensity=True, include_object_targets=True,
                                      radar_region_row_radius=1, radar_region_col_radius=2)[0]
            cached = RangeViewDataset([sample], radar_root, geometry,
                                      require_lidar_intensity=True, include_object_targets=True,
                                      radar_region_row_radius=1, radar_region_col_radius=2,
                                      input_cache_root=cache_root)[0]
            for key in direct:
                np.testing.assert_array_equal(direct[key].numpy(), cached[key].numpy())
            self.assertEqual(int(cached["object_class"].sum()), 1)
            with (partition / "label_2" / "00001.txt").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaises(ValueError):
                validate_cache(cache_root, [sample], radar_root, settings)

    def test_cached_tensors_match_online_projection_and_invalidate_on_source_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "samples" / "train" / "00001.npz"
            sample.parent.mkdir(parents=True)
            radar_root = root / "radar"
            (radar_root / "train").mkdir(parents=True)
            clean_path = root / "clean.bin"
            clean = np.asarray([[5, 0, -0.5, 0.7], [7, 1, 0.3, 0.2]], dtype=np.float32)
            clean.tofile(clean_path)
            metadata = {"dataset": "View-of-Delft", "range_view_full_scan": True,
                        "split": "train", "frame_id": "00001",
                        "source_relative_path": str(clean_path)}
            np.savez(sample, metadata_json=json.dumps(metadata),
                     faulty_lidar_points=clean[:1],
                     faulty_source_ids=np.asarray([0], dtype=np.int64))
            np.savez(radar_root / "train" / "00001.npz",
                     radar_points=np.asarray([[6, 0, -0.3, 1, 2]], dtype=np.float32))
            geometry = RangeGeometry(beam_elevations_rad=(-0.1, 0.1), azimuth_bins=16,
                                     min_range_m=0.1, max_range_m=50,
                                     azimuth_span_rad=2 * np.pi)
            settings = cache_settings(geometry, require_lidar_intensity=True)
            cache_root = root / "cache"
            self.assertTrue(write_cached_sample(sample, radar_root, geometry, cache_root, settings))
            self.assertFalse(write_cached_sample(sample, radar_root, geometry, cache_root, settings))
            manifest = cache_manifest([sample], radar_root, settings)
            (cache_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            validate_cache(cache_root, [sample], radar_root, settings)
            direct = RangeViewDataset([sample], radar_root, geometry,
                                      require_lidar_intensity=True, use_ray_encoding=True)[0]
            cached = RangeViewDataset([sample], radar_root, geometry,
                                      require_lidar_intensity=True, use_ray_encoding=True,
                                      input_cache_root=cache_root)[0]
            for key in direct:
                np.testing.assert_array_equal(direct[key].numpy(), cached[key].numpy())
            self.assertEqual(tuple(load_cached_tensors(cache_root, sample, geometry)["features"].shape),
                             (10, 2, 16))
            altered = cache_settings(geometry, require_lidar_intensity=False)
            with self.assertRaises(ValueError):
                validate_cache(cache_root, [sample], radar_root, altered)
            with sample.open("ab") as handle:
                handle.write(b" ")
            with self.assertRaises(ValueError):
                validate_cache(cache_root, [sample], radar_root, settings)


if __name__ == "__main__":
    unittest.main()
