import multiprocessing
import unittest
from concurrent.futures import ProcessPoolExecutor
from Fault_Localization_Model.create_vod_reconstruction_dataset import _bounded_results
from Fault_Localization_Model.create_vod_reconstruction_dataset import _chronological_tasks


def _identity(value):
    return value


class GenerationWorkerTests(unittest.TestCase):
    def test_ordering_preserves_faults_seeds_and_selected_frames(self):
        tasks = [{'frame': {'radar_path': scene, 'lidar_path': f'/lidar/{stamp}.bin'},
                  'fault': fault, 'severity': index+1, 'injection_seed': index*123}
                 for index, (scene, stamp, fault) in enumerate([
                     ('sceneB', 30, 'fog'), ('sceneA', 20, 'total'), ('sceneA', 10, 'fov')])]
        import copy
        original = copy.deepcopy(tasks)
        ordered = _chronological_tasks(tasks)
        self.assertEqual(tasks, original)
        self.assertEqual([t['injection_seed'] for t in ordered], [246, 123, 0])
        self.assertEqual(sorted(ordered, key=lambda t: t['injection_seed']),
                         sorted(original, key=lambda t: t['injection_seed']))

    def test_spawn_workers_return_all_tasks(self):
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            values = list(_bounded_results(pool, _identity, list(range(12)), 4))
        self.assertEqual(sorted(values), list(range(12)))

    def test_empty_tasks(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertEqual(list(_bounded_results(pool, _identity, [], 2)), [])
