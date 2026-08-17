from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.evaluate_all_coarse_runs import _choose_radar_root


class EvaluateAllCoarseRunsTests(unittest.TestCase):
    def _cache(self, root: Path, name: str) -> Path:
        cache = root / name
        (cache / "test").mkdir(parents=True)
        (cache / "test" / "sample.npz").touch()
        return cache

    def test_selects_pointpillar_radar_cache_by_stack(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            radar3 = self._cache(root, "radar3")
            radar5 = self._cache(root, "radar5")
            args = Namespace(
                radar3_root=radar3,
                radar5_root=radar5,
                radar10_root=None,
                radar20_root=None,
            )

            selected, _ = _choose_radar_root(
                root / "coarse_vod_radar3_hrnet",
                {
                    "model": {
                        "radar_channels": 4,
                        "pointpillars": {"enabled": False},
                    }
                },
                args,
            )
            self.assertEqual(selected, radar3.resolve())

            selected, _ = _choose_radar_root(
                root / "coarse_vod_radar5_bev_hrnet",
                {
                    "model": {
                        "radar_channels": 4,
                        "pointpillars": {"enabled": False},
                    }
                },
                args,
            )
            self.assertEqual(selected, radar5.resolve())

            selected, _ = _choose_radar_root(
                root / "coarse_vod_radar5_pointpillars_hrnet",
                {
                    "model": {
                        "radar_channels": 64,
                        "pointpillars": {"enabled": True},
                    }
                },
                args,
            )
            self.assertEqual(selected, radar5.resolve())


if __name__ == "__main__":
    unittest.main()
