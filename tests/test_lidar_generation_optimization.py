import unittest

import numpy as np

from Fault_Localization_Model.create_grid_reliability_heatmaps import (
    canonical_maps_for_storage,
    chronological_source_batches,
    point_counts_grid,
)


def reference_point_counts_grid(
    points,
    x_min,
    x_max,
    y_min,
    y_max,
    grid_rows,
    grid_cols,
):
    counts = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    x_cell_size = (x_max - x_min) / grid_rows
    y_cell_size = (y_max - y_min) / grid_cols
    cols = np.floor((points[:, 1] - y_min) / y_cell_size).astype(np.int32)
    rows_from_bottom = np.floor(
        (points[:, 0] - x_min) / x_cell_size
    ).astype(np.int32)
    rows = grid_rows - 1 - rows_from_bottom
    valid = (
        (rows >= 0)
        & (rows < grid_rows)
        & (cols >= 0)
        & (cols < grid_cols)
    )
    np.add.at(counts, (rows[valid], cols[valid]), 1.0)
    return counts


class LidarGenerationOptimizationTests(unittest.TestCase):
    def test_storage_keeps_canonical_maps_and_drops_only_legacy_aliases(self):
        canonical = {
            "clean_point_counts": np.asarray([[1.0]], dtype=np.float32),
            "missing_faulty_counts": np.asarray([[2.0]], dtype=np.float32),
            "added_faulty_counts": np.asarray([[3.0]], dtype=np.float32),
        }
        maps = {
            **canonical,
            "correct_counts": canonical["clean_point_counts"],
            "missing_counts": canonical["missing_faulty_counts"],
            "wrong_counts": canonical["added_faulty_counts"],
            "fault_heatmap": np.asarray([[0.5]], dtype=np.float32),
        }

        stored = canonical_maps_for_storage(maps)

        self.assertTrue(set(canonical).issubset(stored))
        self.assertIn("fault_heatmap", stored)
        self.assertNotIn("correct_counts", stored)
        self.assertNotIn("missing_counts", stored)
        self.assertNotIn("wrong_counts", stored)

    def test_bincount_grid_is_exactly_equal_to_previous_accumulation(self):
        rng = np.random.default_rng(42)
        points = np.column_stack(
            (
                rng.uniform(-2.0, 66.0, 10_000),
                rng.uniform(-34.0, 34.0, 10_000),
                rng.normal(size=10_000),
                rng.uniform(size=10_000),
            )
        ).astype(np.float32)

        expected = reference_point_counts_grid(
            points,
            0.0,
            64.0,
            -32.0,
            32.0,
            320,
            320,
        )
        actual = point_counts_grid(
            points,
            0.0,
            64.0,
            -32.0,
            32.0,
            320,
            320,
        )

        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.array_equal(actual, expected))

    def test_chronological_batches_preserve_tasks_and_group_duplicates(self):
        tasks = [
            {"index": 0, "bin_path": "scene/z/30.bin", "injection_seed": 10},
            {"index": 1, "bin_path": "scene/a/20.bin", "injection_seed": 11},
            {"index": 2, "bin_path": "scene/a/10.bin", "injection_seed": 12},
            {"index": 3, "bin_path": "scene/a/20.bin", "injection_seed": 13},
        ]

        batches = chronological_source_batches(tasks, batch_size=2)
        flattened = [task for batch in batches for task in batch]

        self.assertEqual(
            sorted((task["index"], task["injection_seed"]) for task in flattened),
            sorted((task["index"], task["injection_seed"]) for task in tasks),
        )
        duplicate_batch = [
            batch
            for batch in batches
            if any(task["bin_path"] == "scene/a/20.bin" for task in batch)
        ]
        self.assertEqual(len(duplicate_batch), 1)
        self.assertEqual(
            [
                task["index"]
                for task in duplicate_batch[0]
                if task["bin_path"] == "scene/a/20.bin"
            ],
            [1, 3],
        )


if __name__ == "__main__":
    unittest.main()
