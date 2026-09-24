import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from Fault_Localization_Model.hercules_dataset import CONTINENTAL_DTYPE, discover_hercules_frames
from models.two_stage_reconstruction_head.range_view.data import load_range_sample
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from scripts.create_hercules_range_view_dataset import (
    _check_output_roots, _exclude_split_boundary_history, _fill_split, _process,
)
from scripts.create_hercules_range_view_dataset import _candidate_tasks


class HerculesRangeViewGenerationTests(unittest.TestCase):
    def make_session(self, root: Path) -> None:
        session = root / "day" / "session"
        aeva = session / "LiDAR" / "Aeva"
        radar = session / "Radar" / "Continental"
        aeva.mkdir(parents=True)
        radar.mkdir(parents=True)
        for index in range(10):
            row = np.zeros((1, 29), dtype=np.uint8)
            row[:, :16] = np.asarray([[5.0, 0.0, 0.0, 2.0]], dtype="<f4").view(np.uint8)
            row.tofile(aeva / f"{1_000_000_000 + index * 10_000_000}.bin")
        for timestamp in (1_000_000_000, 1_010_000_000, 1_100_000_000):
            row = np.zeros(1, dtype=CONTINENTAL_DTYPE)
            row["x"], row["range"], row["rcs"] = 5, 5, 30
            row.tofile(radar / f"{timestamp}.bin")
        identity = "1 0 0 0 0 1 0 0 0 0 1 0"
        (session / "IMU_LiDAR.txt").write_text("Tr_lidar_to_imu: " + identity)
        (session / "Continental_LiDAR.txt").write_text("Tr_lidar_to_radar: " + identity)
        for name in ("Aeva_gt.txt", "Continental_gt.txt"):
            (session / name).write_text(
                "1000000000 0 0 0 0 0 0 1\n1100000000 0 0 0 0 0 0 1\n"
            )

    def test_generated_artifacts_feed_range_view_without_bev(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            frame = discover_hercules_frames(root, "train")[1]
            task = {
                "frame": {
                    "frame_id": frame.frame_id, "split": frame.split,
                    "lidar_path": frame.lidar_path, "radar_path": frame.radar_path,
                    "lidar_calibration_path": frame.lidar_calibration_path,
                    "radar_calibration_path": frame.radar_calibration_path,
                    "radar_variant": "hercules_range_view_test",
                },
                "fault": "total_loss", "severity": 1, "injection_seed": 123,
                "output_root": str(root / "samples"),
                "radar_cache_root": str(root / "radar_cache"),
                "signature": "test", "compression_level": 1,
                "radar_config": {
                    "hercules_radar_frames": 20, "hercules_temporal_radius": 0.75,
                    "hercules_max_radar_age_ms": 50.0,
                    "hercules_max_pose_gap_ms": 200.0,
                    "hercules_stack": {"max_frames": 20, "max_age_s": 1.0,
                                       "max_translation_m": 4.0, "max_rotation_deg": 5.0},
                    "hercules_tracking": {"doppler_sign": "auto"},
                },
            }
            result = _process(task)
            self.assertEqual(result["status"], "created")
            self.assertEqual(_process(task)["status"], "cached")
            _check_output_roots(root / "samples", root / "radar_cache",
                                {"train": 1, "val": 0, "test": 0}, "test")
            with self.assertRaisesRegex(ValueError, "different generation policy"):
                _check_output_roots(root / "samples", root / "radar_cache",
                                    {"train": 1, "val": 0, "test": 0}, "another-policy")
            with np.load(result["sample"], allow_pickle=False) as sample_archive:
                metadata = json.loads(str(sample_archive["metadata_json"].item()))
                self.assertTrue(metadata["range_view_full_scan"])
                self.assertNotIn("point_filter", metadata)
                self.assertEqual(sample_archive["faulty_lidar_points"].shape, (0, 4))
            with np.load(result["radar"], allow_pickle=False) as radar_archive:
                self.assertEqual(radar_archive["radar_points"].shape[1], 5)
                self.assertNotIn("radar_bev", radar_archive.files)
            geometry = RangeGeometry((-0.1, 0.1), 16, 0.1, 20.0, 2 * np.pi)
            loaded = load_range_sample(result["sample"], root / "radar_cache", geometry)
            self.assertEqual(len(loaded.clean_points), 1)
            self.assertEqual(len(loaded.faulty_points), 0)
            self.assertEqual(int(loaded.targets.add.sum()), 1)
            self.assertTrue(np.all(loaded.features[2] == 0),
                            "Aeva velocity must not be used as reflectivity")
            fov_task = dict(task, fault="fov_filter", severity=1)
            fov_result = _process(fov_task)
            self.assertEqual(fov_result["status"], "created")
            with np.load(fov_result["sample"], allow_pickle=False) as archive:
                self.assertEqual(archive["faulty_lidar_points"].shape[1], 4)
                self.assertEqual(len(archive["faulty_source_ids"]),
                                 len(archive["faulty_lidar_points"]))

    def test_skipped_synchronization_is_backfilled_without_extra_samples(self):
        tasks = [{"frame": {"split": "train"}, "id": index} for index in range(5)]

        def fake_process(task):
            if task["id"] == 1:
                return {"status": "skipped_synchronization", "frame_id": "1"}
            return {"status": "created", "sample": str(task["id"]), "frame_id": str(task["id"])}

        with patch("scripts.create_hercules_range_view_dataset._init_worker"), patch(
            "scripts.create_hercules_range_view_dataset._process", side_effect=fake_process
        ):
            result = _fill_split(tasks, 3, workers=1, progress_every=10)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["skipped_count"], 1)
        self.assertEqual([item["sample"] for item in result["samples"]], ["0", "2", "3"])

    def test_parallel_generation_stops_at_exact_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            frames = discover_hercules_frames(root, "train")
            common = {
                "output_root": str(root / "samples"),
                "radar_cache_root": str(root / "radar_cache"),
                "signature": "parallel-test", "compression_level": 1,
                "radar_config": {
                    "hercules_radar_frames": 20, "hercules_temporal_radius": 0.75,
                    "hercules_max_radar_age_ms": 50.0,
                    "hercules_max_pose_gap_ms": 200.0,
                    "hercules_stack": {"max_frames": 20, "max_age_s": 1.0,
                                       "max_translation_m": 4.0, "max_rotation_deg": 5.0},
                    "hercules_tracking": {"doppler_sign": "auto"},
                },
            }
            tasks = _candidate_tasks(frames, split="train", seed=7,
                                     fault_plan=[("total_loss", 1)], common=common)
            result = _fill_split(tasks, 3, workers=2, progress_every=3)
            self.assertEqual(result["count"], 3)
            self.assertEqual(len(list((root / "samples" / "train").glob("*.npz"))), 3)
            self.assertEqual(len(list((root / "radar_cache" / "train").glob("*.npz"))), 3)

    def test_validation_history_buffer_excludes_boundary_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_session(root)
            validation = discover_hercules_frames(root, "val")
            # This tiny fixture has just one validation frame, which cannot
            # safely accumulate a one-second within-split radar history.
            eligible, excluded = _exclude_split_boundary_history(
                validation, split="val", history_s=1.0)
            self.assertEqual(eligible, [])
            self.assertEqual(excluded, 1)
            training = discover_hercules_frames(root, "train")
            eligible, excluded = _exclude_split_boundary_history(
                training, split="train", history_s=1.0)
            self.assertEqual(len(eligible), len(training))
            self.assertEqual(excluded, 0)


if __name__ == "__main__":
    unittest.main()
