import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.train_coarse_reconstruction import _split_paths as coarse_split_paths
from training.train_residual_diffusion import _split_paths as diffusion_split_paths


class TrainingDataSelectionTests(unittest.TestCase):
    def test_missing_radar_is_filtered_before_seeded_sampling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "data"
            radar_root = root / "radar"
            split_root = data_root / "train"
            split_root.mkdir(parents=True)

            missing_sample = None
            for index in range(6):
                timestamp = str(1000 + index)
                sample_path = split_root / f"sample_{index}.npz"
                metadata = {"scene": "Scene01", "timestamp": timestamp}
                np.savez_compressed(
                    sample_path,
                    metadata_json=np.array(json.dumps(metadata)),
                )
                if index == 5:
                    missing_sample = sample_path
                    continue
                radar_path = radar_root / "Scene01" / f"{timestamp}.npz"
                radar_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(radar_path, radar=np.zeros((1,), dtype=np.float32))

            with contextlib.redirect_stdout(io.StringIO()):
                coarse_first = coarse_split_paths(
                    data_root, radar_root, "train", 3, 17
                )
                coarse_second = coarse_split_paths(
                    data_root, radar_root, "train", 3, 17
                )
                diffusion = diffusion_split_paths(
                    data_root, radar_root, "train", 3, 17
                )

            self.assertEqual(coarse_first, coarse_second)
            self.assertEqual(coarse_first, diffusion)
            self.assertEqual(len(coarse_first), 3)
            self.assertNotIn(missing_sample, coarse_first)


if __name__ == "__main__":
    unittest.main()
