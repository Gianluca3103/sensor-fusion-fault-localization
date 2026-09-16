import unittest
import tempfile
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from Fault_Localization_Model.geometric_reference import (
    GeometricReferenceConfig, DatasetReferenceSource, build_reference, load_reference,
)
from models.two_stage_reconstruction_head.geometric_reconstruction import (
    GeometricLossConfig, SoftGeometricReconstructionLoss, expected_xyz, point_metrics,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


class GeometryLossTests(unittest.TestCase):
    def setUp(self):
        self.geometry = BEVGridGeometry(0., 1., 0., 1., 5, 5)
        self.loss = SoftGeometricReconstructionLoss(GeometricLossConfig(enabled=True, halo_m=0.), self.geometry)
        self.bev = torch.zeros(1, 3, 5, 5)
        self.bev[:, 2] = 3/8
        self.bev[:, 0, 2, 2] = 1.
        self.mask = torch.ones(1, 1, 5, 5)
        self.reference = expected_xyz(self.bev[0], self.geometry)[12:13].detach()

    def test_identity_and_known_translation(self):
        identity = self.loss(self.bev, [self.reference], self.mask)
        self.assertAlmostEqual(float(identity['geometric_loss']), 0., places=6)
        shifted = self.reference+torch.tensor([0., 0., .2])
        result = self.loss(self.bev, [shifted], self.mask)
        self.assertAlmostEqual(float(result['geometric_coverage_loss']), .15, places=5)
        self.assertAlmostEqual(float(result['geometric_accuracy_loss']), .15, places=5)
        metrics = point_metrics(self.reference.numpy(), shifted.numpy())
        self.assertAlmostEqual(metrics['chamfer_mean_m'], .2, places=5)

    def test_occupancy_and_height_gradients_are_finite_nonzero(self):
        prediction = self.bev.clone()
        prediction[:, 0] = .2
        prediction[:, 2] += .02
        prediction.requires_grad_()
        self.loss(prediction, [self.reference], self.mask)['geometric_loss'].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(prediction.grad[:, 2].abs().sum()), 0.)

    def test_densification_hallucination_and_missing_geometry(self):
        reference = np.array([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]])
        sparse = reference[[0, 2]]
        dense = point_metrics(reference, reference)
        missing = point_metrics(sparse, reference)
        displaced = point_metrics(reference+np.array([0., .3, 0.]), reference)
        hallucinated = point_metrics(np.concatenate([reference, [[3., 3., 0.]]]), reference,
                                    observable=np.ones(4, dtype=bool))
        self.assertLess(dense['chamfer_mean_m'], displaced['chamfer_mean_m'])
        self.assertGreater(missing['ref_to_pred_mean_m'], dense['ref_to_pred_mean_m'])
        self.assertGreater(hallucinated['pred_to_ref_mean_m'], dense['pred_to_ref_mean_m'])
        self.assertGreater(hallucinated['hallucination_rate_0_2m'], 0.)
        unknown = point_metrics(np.array([[3., 3., 0.]]), reference)
        self.assertIsNone(unknown['hallucination_rate_0_2m'])
        self.assertEqual(unknown['unknown_predictions'], 1)

    def test_missing_and_hallucinated_prediction_losses(self):
        missing = self.bev.clone()
        missing[:, 0] = 0.
        hallucinated = self.bev.clone()
        hallucinated[:, 0, 0, 0] = 1.
        self.assertGreater(float(self.loss(missing, [self.reference], self.mask)['geometric_coverage_loss']), .4)
        self.assertGreater(float(self.loss(hallucinated, [self.reference], self.mask)['geometric_accuracy_loss']), 0.)

    def test_empty_sets_and_healthy_halo_has_no_gradients(self):
        empty = np.empty((0, 3))
        self.assertEqual(point_metrics(empty, empty)['f1_0.1m'], 1.)
        self.assertEqual(point_metrics(empty, self.reference.numpy())['f1_0.1m'], 0.)
        prediction = self.bev.clone().requires_grad_()
        mask = torch.zeros_like(self.mask)
        mask[:, :, 2, 2] = 1
        loss = SoftGeometricReconstructionLoss(replace(self.loss.config, halo_m=.4), self.geometry)
        loss(prediction, [self.reference+torch.tensor([0., 0., .1])], mask)['geometric_loss'].backward()
        healthy = mask.expand_as(prediction) == 0
        self.assertEqual(float(prediction.grad[healthy].abs().sum()), 0.)


class DenseReferenceTests(unittest.TestCase):
    def make_vod(self, root, labels=True):
        public = root / 'raw'
        train = public / 'lidar/training'
        for directory in ('velodyne', 'pose', 'calib', 'label_2'):
            (train / directory).mkdir(parents=True)
        (public / 'lidar/ImageSets').mkdir()
        (public / 'lidar/ImageSets/train.txt').write_text('00000\n00001\n00002\n00003\n')
        (public / 'lidar/ImageSets/val.txt').write_text('00004\n')
        for index in range(5):
            points = np.array([[10.-.1*index, .02*index, 0., 42.],
                               [20.+.4*index, 0., 0., 42.]], dtype=np.float32)
            points.tofile(train / 'velodyne' / f'{index:05d}.bin')
            pose = np.eye(4)
            pose[0, 3] = .1*index
            (train / 'pose' / f'{index:05d}.json').write_text(json.dumps({'odomToCamera': pose.flatten().tolist()}))
            (train / 'calib' / f'{index:05d}.txt').write_text('Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0')
            if labels:
                (train / 'label_2' / f'{index:05d}.txt').write_text(f'Car 0 0 0 0 0 0 0 2 2 2 {20.+.4*index} 0 0 0')
        data = root / 'samples'
        sample = data / 'train/frame.npz'
        sample.parent.mkdir(parents=True)
        np.savez(sample, metadata_json=json.dumps({'dataset': 'VoD', 'split': 'train', 'frame_id': '00002'}))
        config = GeometricReferenceConfig(enabled=True, cache_root=str(root / 'references'))
        return public, data, sample, config

    def test_actual_pose_alignment_boxes_and_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, data, sample, config = self.make_vod(root)
            source = DatasetReferenceSource(raw, 'VoD', 'train')
            path, _ = build_reference(sample, data, source, config)
            reference = load_reference(sample, data, config)
            static = reference[reference[:, 0] < 15]
            np.testing.assert_allclose(static[:, 0], 9.8, atol=1e-5)
            self.assertEqual(len(reference[reference[:, 0] > 15]), 1)
            with np.load(path) as cache:
                metadata = json.loads(str(cache['metadata_json'].item()))
            self.assertNotIn('00004', metadata['frames_used'])
            self.assertEqual(metadata['dynamic_strategy'], 'boxes')
            self.assertTrue(build_reference(sample, data, source, config)[1])
            with self.assertRaises(ValueError):
                load_reference(sample, data, replace(config, past_frames=1))

    def test_missing_annotations_fail_closed_and_no_reference_in_model_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, data, sample, config = self.make_vod(root, labels=False)
            source = DatasetReferenceSource(raw, 'VoD', 'train')
            path, _ = build_reference(sample, data, source, config)
            with np.load(path) as cache:
                self.assertEqual(json.loads(str(cache['metadata_json'].item()))['dynamic_strategy'], 'central_only')
                self.assertEqual(len(cache['reference_points']), len(cache['central_points']))
        from models.two_stage_reconstruction_head.coarse_reconstruction.train_coarse_reconstruction import _move_batch as coarse_move
        from models.two_stage_reconstruction_head.diffusion_process.train_fine_diffusion import _move_batch as fine_move
        batch = {key: torch.zeros(1, 1, 2, 2) for key in ('faulty_bev', 'radar_bev',
            'reconstruction_mask', 'healthy_context_mask', 'halo_mask', 'clean_bev')}
        batch['geometric_reference_points'] = (torch.ones(100, 3),)
        self.assertNotIn('geometric_reference_points', coarse_move(batch, torch.device('cpu')))
        self.assertNotIn('geometric_reference_points', fine_move(batch, torch.device('cpu')))


if __name__ == '__main__':
    unittest.main()
