import multiprocessing
import unittest
from concurrent.futures import ProcessPoolExecutor
from Fault_Localization_Model.create_vod_reconstruction_dataset import _bounded_results


def _identity(value):
    return value


class GenerationWorkerTests(unittest.TestCase):
    def test_spawn_workers_return_all_tasks(self):
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            values = list(_bounded_results(pool, _identity, list(range(12)), 4))
        self.assertEqual(sorted(values), list(range(12)))

    def test_empty_tasks(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertEqual(list(_bounded_results(pool, _identity, [], 2)), [])
