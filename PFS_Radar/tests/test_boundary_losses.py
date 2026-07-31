import unittest

import torch

from PFS_Radar.boundary_losses import BoundaryWeightedBCELoss


class BoundaryWeightedBCELossTest(unittest.TestCase):
    def test_all_healthy_has_zero_boundary_loss(self):
        loss_fn = BoundaryWeightedBCELoss()
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        target = torch.zeros_like(logits)
        loss, diagnostics = loss_fn(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(diagnostics["boundary_strength_mean"]), 0.0)

    def test_all_faulty_has_zero_boundary_loss(self):
        loss_fn = BoundaryWeightedBCELoss()
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        target = torch.ones_like(logits)
        loss, diagnostics = loss_fn(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(float(diagnostics["boundary_strength_mean"]), 0.0)

    def test_single_faulty_cell_generates_boundary_and_confidence_reduces_weight(self):
        target = torch.zeros(1, 1, 9, 9)
        target[:, :, 4, 4] = 1.0
        logits = torch.zeros_like(target, requires_grad=True)

        ordinary = BoundaryWeightedBCELoss(use_evidence_confidence=False)
        confident = BoundaryWeightedBCELoss(
            use_evidence_confidence=True,
            evidence_n_ref=10.0,
        )
        ordinary_loss, ordinary_diag = ordinary(logits, target)
        low_evidence_loss, low_evidence_diag = confident(
            logits,
            target,
            evidence_count=torch.ones_like(target),
        )

        self.assertGreater(float(ordinary_diag["boundary_cell_fraction"]), 0.0)
        self.assertGreater(float(ordinary_loss.detach()), 0.0)
        self.assertGreater(
            float(ordinary_diag["boundary_weight_mean"]),
            float(low_evidence_diag["boundary_weight_mean"]),
        )
        self.assertTrue(torch.isfinite(low_evidence_loss))

    def test_large_homogeneous_region_has_boundary_not_interior(self):
        loss_fn = BoundaryWeightedBCELoss()
        target = torch.zeros(1, 1, 12, 12)
        target[:, :, 3:9, 3:9] = 1.0
        logits = torch.zeros_like(target)
        _loss, diagnostics = loss_fn(logits, target)
        self.assertGreater(float(diagnostics["boundary_cell_fraction"]), 0.0)
        local_fault = loss_fn._local_average(target)
        boundary_strength = (4.0 * local_fault * (1.0 - local_fault)).clamp(0.0, 1.0)
        self.assertAlmostEqual(float(boundary_strength[0, 0, 5, 5]), 0.0, places=6)
        self.assertGreater(float(boundary_strength[0, 0, 3, 5]), 0.0)

    def test_fragmented_target_generates_boundary_response(self):
        loss_fn = BoundaryWeightedBCELoss()
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, 1::2, 1::2] = 1.0
        logits = torch.zeros_like(target)
        loss, diagnostics = loss_fn(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(diagnostics["boundary_cell_fraction"]), 0.0)

    def test_soft_target_is_accepted_without_thresholding(self):
        loss_fn = BoundaryWeightedBCELoss()
        target = torch.full((1, 1, 8, 8), 0.5)
        logits = torch.zeros_like(target)
        loss, diagnostics = loss_fn(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        self.assertAlmostEqual(float(diagnostics["boundary_strength_mean"]), 1.0, places=6)

    def test_gradient_is_finite_and_nonzero_when_boundary_exists(self):
        loss_fn = BoundaryWeightedBCELoss()
        target = torch.zeros(1, 1, 8, 8)
        target[:, :, 2:6, 2:6] = 1.0
        logits = torch.zeros_like(target, requires_grad=True)
        loss, _diagnostics = loss_fn(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
