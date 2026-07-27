from pathlib import Path
import unittest
from unittest.mock import patch

from PFS_Radar.radar_data import RadarAlignmentUnavailableError
from PFS_Radar_v2.prepare_radar_cache import _build_batch, _scene_batches


class PrepareRadarCacheTests(unittest.TestCase):
    def test_scene_batches_are_chronological_and_scene_local(self):
        keyed_tasks = [
            (("scene_b", "session", 30), "b30"),
            (("scene_a", "session", 20), "a20"),
            (("scene_a", "session", 10), "a10"),
            (("scene_a", "session", 30), "a30"),
        ]

        batches = _scene_batches(keyed_tasks, batch_size=2)

        self.assertEqual(batches, [["a10", "a20"], ["a30"], ["b30"]])

    def test_build_batch_reports_each_outcome_without_losing_completed_work(self):
        tasks = [
            (Path("created.npz"),),
            (Path("skipped.npz"),),
            (Path("failed.npz"),),
        ]

        def build(task):
            if task[0].name == "skipped.npz":
                raise RadarAlignmentUnavailableError("no causal frame")
            if task[0].name == "failed.npz":
                raise ValueError("bad frame")

        with patch(
            "PFS_Radar_v2.prepare_radar_cache._build",
            side_effect=build,
        ):
            outcomes = _build_batch(tasks)

        self.assertEqual(outcomes[0][0], "created")
        self.assertEqual(outcomes[1][0], "skipped")
        self.assertIn("no causal frame", outcomes[1][2])
        self.assertEqual(outcomes[2][0], "failed")
        self.assertIn("ValueError: bad frame", outcomes[2][2])


if __name__ == "__main__":
    unittest.main()
