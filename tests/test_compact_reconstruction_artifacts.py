import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.compact_reconstruction_artifacts import compact_file


class CompactReconstructionArtifactsTests(unittest.TestCase):
    def test_sample_compaction_retains_training_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            shape = (4, 3)
            required_maps = {
                "fault_heatmap": np.zeros(shape, dtype=np.float32),
                "reliability_map": np.ones(shape, dtype=np.float32),
                "faulty_counts": np.zeros(shape, dtype=np.float32),
                "missing_faulty_counts": np.zeros(shape, dtype=np.float32),
                "moved_faulty_counts": np.zeros(shape, dtype=np.float32),
                "added_faulty_counts": np.zeros(shape, dtype=np.float32),
            }
            np.savez_compressed(
                path,
                **required_maps,
                clean_rgb=np.zeros((*shape, 3), dtype=np.uint8),
                faulty_rgb=np.zeros((*shape, 3), dtype=np.uint8),
                faulty_lidar_points=np.zeros((2, 4), dtype=np.float32),
                observability_confidence=np.ones(shape, dtype=np.float16),
                metadata_json=np.asarray(json.dumps({"dataset": "HeRCULES"})),
                clean_point_ids=np.arange(1000, dtype=np.int64),
                observability_ray_count=np.ones(shape, dtype=np.uint32),
            )

            _, _, kind = compact_file(path, 6)

            self.assertEqual(kind, "sample")
            with np.load(path, allow_pickle=False) as archive:
                self.assertNotIn("clean_point_ids", archive.files)
                self.assertNotIn("observability_ray_count", archive.files)
                self.assertIn("faulty_lidar_points", archive.files)
                self.assertEqual(
                    json.loads(str(archive["metadata_json"].item()))[
                        "artifact_profile"
                    ],
                    "training",
                )

    def test_radar_compaction_retains_pointpillars_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "radar.npz"
            np.savez_compressed(
                path,
                radar_bev=np.zeros((4, 4, 3), dtype=np.float16),
                radar_points=np.zeros((7, 5), dtype=np.float32),
                radar_point_weights=np.ones(7, dtype=np.float32),
                metadata_json=np.asarray(json.dumps({"dataset": "HeRCULES"})),
                unused=np.arange(1000),
            )

            _, _, kind = compact_file(path, 6)

            self.assertEqual(kind, "radar")
            with np.load(path, allow_pickle=False) as archive:
                self.assertNotIn("unused", archive.files)
                self.assertEqual(archive["radar_points"].shape, (7, 5))
                self.assertEqual(archive["radar_bev"].shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
