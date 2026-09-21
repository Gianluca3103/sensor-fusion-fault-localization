import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from voxelization import (
    HardVoxelizer,
    ModalityVoxelConfig,
    VoxelGridConfig,
    VoxelizationConfig,
)
from voxelization.cache import (
    InvalidVoxelCacheError,
    cache_metadata,
    load_voxel_cache,
    write_voxel_cache,
)


LIDAR_FIELDS = ("x", "y", "z", "reflectivity")


class Voxelization3DTests(unittest.TestCase):
    def setUp(self):
        self.grid = VoxelGridConfig()
        self.voxelizer = HardVoxelizer(self.grid)

    def test_a_boundary_mapping(self):
        near_max = np.nextafter(
            np.asarray(self.grid.maxs_xyz, dtype=np.float32),
            np.asarray(self.grid.mins_xyz, dtype=np.float32),
        )
        points = np.asarray(
            [
                [0.0, -32.0, -3.0, 1.0],
                [0.1999, -31.8001, -2.7501, 2.0],
                [0.2, -31.8, -2.75, 3.0],
                [*near_max, 4.0],
                [64.0, 0.0, 0.0, 5.0],
                [1.0, 32.0, 0.0, 6.0],
                [1.0, 0.0, 5.0, 7.0],
                [-0.0001, 0.0, 0.0, 8.0],
            ],
            dtype=np.float32,
        )
        coordinates, valid = self.voxelizer.point_indices(points)
        self.assertTrue(np.array_equal(valid, [1, 1, 1, 1, 0, 0, 0, 0]))
        self.assertTrue(np.array_equal(coordinates[0], [0, 0, 0]))
        self.assertTrue(np.array_equal(coordinates[1], [0, 0, 0]))
        self.assertTrue(np.array_equal(coordinates[2], [1, 1, 1]))
        self.assertTrue(np.array_equal(coordinates[3], [31, 319, 319]))

    def test_b_voxel_count_and_occupancy(self):
        points = np.asarray(
            [[0.01, -31.99, -2.99, 1], [0.02, -31.98, -2.98, 2], [1, 0, 0, 3]],
            dtype=np.float32,
        )
        result = self.voxelizer.voxelize(points, LIDAR_FIELDS)
        self.assertEqual(result.occupied_voxel_count, 2)
        self.assertTrue(np.array_equal(result.original_num_points, [2, 1]))
        self.assertEqual(result.valid_point_count, 3)

    def test_c_centroid_offsets_sum_to_zero_without_truncation(self):
        points = np.asarray(
            [[1.01, 2.01, 0.01, 1], [1.09, 2.05, 0.09, 2], [1.15, 2.11, 0.12, 3]],
            dtype=np.float32,
        )
        result = self.voxelizer.voxelize(points, LIDAR_FIELDS)
        centroid_offsets = result.voxel_points[0, :3, 4:7]
        self.assertTrue(np.allclose(centroid_offsets.sum(axis=0), 0, atol=1e-6))

    def test_d_index_to_center_round_trip(self):
        points = np.asarray(
            [[0.01, -31.99, -2.99, 1], [63.99, 31.99, 4.99, 2], [10.1, 3.2, 1.2, 3]],
            dtype=np.float32,
        )
        coordinates, valid = self.voxelizer.point_indices(points)
        centers = self.voxelizer.voxel_centers(coordinates[valid])
        half = np.asarray(self.grid.voxel_size) / 2
        self.assertTrue(np.all(np.abs(points[valid, :3] - centers) <= half + 1e-6))
        round_trip, round_valid = self.voxelizer.point_indices(
            np.c_[centers, np.zeros(len(centers), dtype=np.float32)]
        )
        self.assertTrue(round_valid.all())
        self.assertTrue(np.array_equal(round_trip, coordinates[valid]))

    def test_e_lidar_and_radar_are_separate(self):
        lidar = np.asarray([[1, 2, 0, 0.5]], dtype=np.float32)
        radar = np.asarray([[1, 2, 0, 12, -3]], dtype=np.float32)
        lidar_result = self.voxelizer.voxelize(lidar, LIDAR_FIELDS)
        radar_result = self.voxelizer.voxelize(
            radar, ("x", "y", "z", "rcs", "compensated_radial_velocity")
        )
        self.assertEqual(lidar_result.voxel_points.shape[-1], 10)
        self.assertEqual(radar_result.voxel_points.shape[-1], 11)
        self.assertEqual(lidar_result.raw_feature_names[-1], "reflectivity")
        self.assertEqual(radar_result.raw_feature_names[-1], "compensated_radial_velocity")

    def test_f_empty_cloud(self):
        result = self.voxelizer.voxelize(
            np.empty((0, 4), dtype=np.float32), LIDAR_FIELDS
        )
        self.assertEqual(result.voxel_coords.shape, (0, 3))
        self.assertEqual(result.voxel_points.shape, (0, 0, 10))
        self.assertEqual(result.num_points.shape, (0,))

    def test_g_deterministic_truncation(self):
        points = np.asarray(
            [[1.01 + index * 0.001, 2.01, 0.01, index] for index in range(10)],
            dtype=np.float32,
        )
        voxelizer = HardVoxelizer(self.grid, max_points_per_voxel=4)
        first = voxelizer.voxelize(points, LIDAR_FIELDS)
        second = voxelizer.voxelize(points, LIDAR_FIELDS)
        self.assertTrue(np.array_equal(first.voxel_points, second.voxel_points))
        self.assertTrue(np.array_equal(first.voxel_points[0, :, 3], [0, 1, 2, 3]))
        self.assertEqual(first.truncated_point_count, 6)
        self.assertEqual(int(first.num_points[0]), 4)
        self.assertEqual(int(first.original_num_points[0]), 10)

    def test_h_cache_reproducibility_and_visualization_sanity(self):
        points = np.asarray(
            [[1.01, 2.01, 0.01, 1], [1.09, 2.05, 0.09, 2], [5, -4, 1, 3]],
            dtype=np.float32,
        )
        config = VoxelizationConfig(
            grid=self.grid,
            lidar=ModalityVoxelConfig(8),
            radar=ModalityVoxelConfig(None),
        )
        result = HardVoxelizer(self.grid, max_points_per_voxel=8).voxelize(
            points, LIDAR_FIELDS
        )
        centers = self.voxelizer.voxel_centers(result.voxel_coords)
        self.assertTrue(np.isfinite(centers).all())
        self.assertTrue(np.all(centers[:, 0] >= self.grid.x_range[0]))
        self.assertTrue(np.all(centers[:, 0] < self.grid.x_range[1]))
        metadata = cache_metadata(
            result, config, modality="lidar", source_path="source.bin",
            sample_path="sample.npz", split="val", frame_id="1", lidar_source="clean",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_voxel_cache(path, result, metadata, compression_level=1)
            loaded, loaded_metadata = load_voxel_cache(path, config, modality="lidar")
            self.assertTrue(np.array_equal(loaded.voxel_coords, result.voxel_coords))
            self.assertTrue(np.array_equal(loaded.voxel_points, result.voxel_points))
            self.assertEqual(loaded_metadata["config_hash"], config.fingerprint)
            changed = VoxelizationConfig(grid=VoxelGridConfig(voxel_size=(0.4, 0.2, 0.25)))
            with self.assertRaises(InvalidVoxelCacheError):
                load_voxel_cache(path, changed, modality="lidar")

    def test_rejects_nonfinite_and_reports_out_of_range(self):
        points = np.asarray(
            [[1, 2, 0, 1], [np.nan, 2, 0, 2], [65, 2, 0, 3]], dtype=np.float32
        )
        result = self.voxelizer.voxelize(points, LIDAR_FIELDS)
        self.assertEqual(result.nonfinite_point_count, 1)
        self.assertEqual(result.out_of_range_point_count, 1)
        self.assertEqual(result.valid_point_count, 1)


if __name__ == "__main__":
    unittest.main()
