import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from Fault_Localization_Model.concurrency_utils import iter_bounded_futures
from Fault_Localization_Model.io_utils import (
    atomic_savez_compressed,
    atomic_torch_save,
    atomic_write_json,
)
from Fault_Localization_Model.model_blocks import resize_reliability_map
from Fault_Localization_Model.sample_utils import (
    InvalidSampleError,
    filter_paths_by_fault,
    require_disjoint_splits,
    validate_heatmap_array,
    validate_radar_array,
    validate_rgb_array,
)
from Fault_Localization_Model.visualization_utils import draw_cell_boundaries
from PFS.pfs_model import PFSReliabilityModel
from PFS.training_utils import (
    require_checkpoint_args_match,
    require_checkpoint_semantics,
)
from PFS_Radar.pfs_radar_model import PFSRadarReliabilityModel


def write_sample(path, source_relative_path):
    metadata = {
        "scene": "Scene01",
        "timestamp": Path(source_relative_path).stem,
        "source_relative_path": source_relative_path,
    }
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


class RobustnessTests(unittest.TestCase):
    def test_bounded_future_iterator_processes_every_task(self):
        tasks = list(range(17))
        with ThreadPoolExecutor(max_workers=3) as executor:
            completed = [
                (task, future.result())
                for future, task in iter_bounded_futures(
                    executor,
                    lambda value: value * value,
                    tasks,
                    max_pending=5,
                )
            ]
        self.assertEqual(
            sorted(completed),
            [(value, value * value) for value in tasks],
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.assertRaisesRegex(ValueError, "max_pending"):
                list(iter_bounded_futures(executor, int, [], max_pending=0))

    def test_atomic_numpy_and_json_writes_leave_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "sample.npz"
            checkpoint = root / "checkpoint.pt"
            summary = root / "summary.json"
            atomic_savez_compressed(
                archive,
                values=np.arange(5, dtype=np.float32),
            )
            atomic_write_json(summary, {"ok": True})
            atomic_torch_save({"value": torch.tensor([3.0])}, checkpoint)

            with np.load(archive, allow_pickle=False) as data:
                np.testing.assert_array_equal(
                    data["values"], np.arange(5, dtype=np.float32)
                )
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"ok": True})
            loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(float(loaded["value"][0]), 3.0)
            self.assertEqual(
                [path for path in root.iterdir() if path.name.startswith(".")],
                [],
            )

    def test_split_leakage_is_detected_by_source_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.npz"
            validation = root / "validation.npz"
            write_sample(train, "Scene01/LiDAR/Aeva/123.bin")
            write_sample(validation, "Scene01\\LiDAR\\Aeva\\123.bin")

            with self.assertRaisesRegex(ValueError, "split leakage"):
                require_disjoint_splits(
                    {"train": [train], "validation": [validation]}
                )

    def test_fault_filter_rejects_unknown_requested_names(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.npz"
            metadata = {
                "scene": "Scene01",
                "timestamp": "123",
                "fault": "fog_sim",
            }
            np.savez_compressed(
                sample,
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            with self.assertRaisesRegex(ValueError, "rain_typo"):
                filter_paths_by_fault(
                    [sample],
                    exclude_faults=["rain_typo"],
                    strict_fault_names=True,
                )

    def test_split_contract_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.npz"
            validation = root / "validation.npz"
            write_sample(train, "Scene01/LiDAR/Aeva/100.bin")
            write_sample(validation, "Scene01/LiDAR/Aeva/200.bin")

            with np.load(validation, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"]))
            metadata["generator_version"] = 2
            np.savez_compressed(
                validation,
                metadata_json=np.asarray(json.dumps(metadata)),
            )

            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                require_disjoint_splits(
                    {"train": [train], "validation": [validation]}
                )

    def test_array_contracts_reject_invalid_training_data(self):
        with self.assertRaises(InvalidSampleError):
            validate_rgb_array(
                np.zeros((4, 4), dtype=np.uint8),
                name="faulty_rgb",
                path="sample.npz",
            )
        with self.assertRaises(InvalidSampleError):
            validate_heatmap_array(
                np.asarray([[1.1]], dtype=np.float32),
                path="sample.npz",
            )
        with self.assertRaises(InvalidSampleError):
            validate_radar_array(
                np.zeros((3, 4, 4), dtype=np.float32),
                path="radar.npz",
            )

    def test_reliability_downsampling_uses_area_aggregation(self):
        values = torch.tensor(
            [[[[0.0, 0.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )
        resized = resize_reliability_map(values, (1, 1))
        self.assertAlmostEqual(float(resized.item()), 0.25)

    def test_visual_grid_boundaries_follow_fractional_cell_edges(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = draw_cell_boundaries(image, grid_size=3)
        line_color = np.asarray([18, 18, 18], dtype=np.uint8)
        self.assertTrue(np.all(result[4, :] == line_color))
        self.assertTrue(np.all(result[7, :] == line_color))
        self.assertTrue(np.all(result[:, 4] == line_color))
        self.assertTrue(np.all(result[:, 7] == line_color))
        self.assertTrue(np.all(result[3, 3] == 0))

    def test_clean_reference_does_not_update_batchnorm_statistics(self):
        faulty = torch.rand(2, 3, 32, 32)
        clean = torch.rand(2, 3, 32, 32)

        pfs = PFSReliabilityModel(base_channels=2)
        pfs.train()
        pfs_batch_norm = pfs.encoder.enc1.block[1]
        before = int(pfs_batch_norm.num_batches_tracked)
        pfs(faulty, clean_bev=clean, return_features=True)
        self.assertEqual(int(pfs_batch_norm.num_batches_tracked) - before, 1)

        radar_model = PFSRadarReliabilityModel(base_channels=2)
        radar_model.train()
        radar_batch_norm = radar_model.lidar_encoder.enc1.block[1]
        fusion_batch_norm = radar_model.fusion[1]
        radar_before = int(radar_batch_norm.num_batches_tracked)
        fusion_before = int(fusion_batch_norm.num_batches_tracked)
        outputs = radar_model(
            faulty,
            torch.rand(2, 4, 32, 32),
            clean_lidar_bev=clean,
            return_features=True,
        )
        self.assertEqual(
            int(radar_batch_norm.num_batches_tracked) - radar_before,
            1,
        )
        self.assertEqual(
            int(fusion_batch_norm.num_batches_tracked) - fusion_before,
            1,
        )
        self.assertEqual(
            outputs["clean_features"].shape,
            outputs["stabilized_features"].shape,
        )

    def test_resume_rejects_changed_behavioral_arguments(self):
        current = SimpleNamespace(learning_rate=1e-4, exclude_faults=["snow", "rain"])
        require_checkpoint_args_match(
            {"learning_rate": 1e-4, "exclude_faults": ["rain", "snow"]},
            current,
            ("learning_rate", "exclude_faults"),
        )
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            require_checkpoint_args_match(
                {"learning_rate": 2e-4},
                current,
                ("learning_rate",),
            )

    def test_resume_rejects_missing_or_changed_training_semantics(self):
        with self.assertRaisesRegex(ValueError, "predates"):
            require_checkpoint_semantics({}, 2, "test")
        require_checkpoint_semantics(
            {"training_semantics_version": 2},
            2,
            "test",
        )
        with self.assertRaisesRegex(ValueError, "version 1"):
            require_checkpoint_semantics(
                {"training_semantics_version": 1},
                2,
                "test",
            )

if __name__ == "__main__":
    unittest.main()
