from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction import (
    _active_fraction_recommendation,
    _move_batch as move_coarse_batch,
    _run_epoch as run_coarse_epoch,
    _split_paths as coarse_split_paths,
    _summarize_active_fractions,
)
from models.reconstruction_head.diffusion_process.train_residual_diffusion import (
    _move_batch as move_diffusion_batch,
    _split_paths as diffusion_split_paths,
)
from models.reconstruction_head import MaskedBEVReconstructionLoss


class TrainingDataSelectionTests(unittest.TestCase):
    def test_validation_attention_is_requested_only_for_saved_batch(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention_requests = []

            def forward(
                self,
                faulty_bev,
                radar_bev,
                reconstruction_mask,
                healthy_context_mask,
                halo_mask,
                *,
                return_attention_weights=False,
            ):
                self.attention_requests.append(return_attention_weights)
                erased = faulty_bev * (1.0 - reconstruction_mask)
                outputs = {
                    "erased_lidar_bev": erased,
                    "replacement_raw": faulty_bev,
                    "coarse_lidar_bev": faulty_bev,
                    "reconstruction_mask": reconstruction_mask,
                    "healthy_context_mask": healthy_context_mask,
                    "halo_mask": halo_mask,
                }
                for name in (
                    "local_input",
                    "local_bottleneck",
                    "query_tokens",
                    "context_tokens",
                    "attention_context",
                    "fused_bottleneck",
                    "global_context_map",
                ):
                    outputs[name] = faulty_bev
                if return_attention_weights:
                    outputs["attention_weights"] = torch.zeros(1, 1, 1, 1)
                return outputs

        batch = {
            "clean_bev": torch.zeros(1, 3, 4, 4),
            "faulty_bev": torch.zeros(1, 3, 4, 4),
            "radar_bev": torch.zeros(1, 4, 4, 4),
            "reconstruction_mask": torch.ones(1, 1, 4, 4),
            "healthy_context_mask": torch.zeros(1, 1, 4, 4),
            "halo_mask": torch.zeros(1, 1, 4, 4),
        }
        model = RecordingModel()
        saved_batches = []
        active_fractions = []

        run_coarse_epoch(
            model,
            [batch, batch],
            MaskedBEVReconstructionLoss(),
            torch.device("cpu"),
            return_attention=True,
            conditioning_callback=lambda raw_batch, outputs: saved_batches.append(
                "attention_weights" in outputs
            ),
            active_fraction_samples=active_fractions,
        )

        self.assertEqual(model.attention_requests, [True, False])
        self.assertEqual(saved_batches, [True])
        self.assertEqual(active_fractions, [1.0, 1.0])

    def test_active_fraction_summary_and_recommendation(self):
        summary = _summarize_active_fractions([0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertAlmostEqual(summary["median"], 0.5)
        self.assertAlmostEqual(summary["p90"], 0.9)
        self.assertAlmostEqual(summary["maximum"], 1.0)
        self.assertIn("dense U-Net", _active_fraction_recommendation(summary))

        sparse_summary = _summarize_active_fractions([0.01, 0.05, 0.1])
        self.assertIn(
            "cropped dense", _active_fraction_recommendation(sparse_summary)
        )

    def test_existing_split_is_sampled_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "data"
            split_root = data_root / "train"
            split_root.mkdir(parents=True)

            for index in range(6):
                sample_path = split_root / f"sample_{index}.npz"
                np.savez_compressed(
                    sample_path,
                    value=np.asarray(index),
                )

            all_paths = coarse_split_paths(data_root, "train", None, 17)
            coarse_first = coarse_split_paths(data_root, "train", 3, 17)
            coarse_second = coarse_split_paths(data_root, "train", 3, 17)
            diffusion = diffusion_split_paths(data_root, "train", 3, 17)

            self.assertEqual(len(all_paths), 6)
            self.assertEqual(coarse_first, coarse_second)
            self.assertEqual(coarse_first, diffusion)
            self.assertEqual(len(coarse_first), 3)

    def test_cached_masks_are_cast_after_device_transfer(self):
        batch = {
            "clean_bev": torch.zeros(1, 3, 4, 4),
            "faulty_bev": torch.zeros(1, 3, 4, 4),
            "radar_bev": torch.zeros(1, 4, 4, 4),
            "reconstruction_mask": torch.zeros(1, 1, 4, 4, dtype=torch.uint8),
            "healthy_context_mask": torch.zeros(1, 1, 4, 4, dtype=torch.uint8),
            "halo_mask": torch.zeros(1, 1, 4, 4, dtype=torch.uint8),
        }

        coarse = move_coarse_batch(batch, torch.device("cpu"))
        diffusion = move_diffusion_batch(batch, torch.device("cpu"))

        for moved in (coarse, diffusion):
            self.assertEqual(moved["reconstruction_mask"].dtype, torch.float32)
            self.assertEqual(moved["healthy_context_mask"].dtype, torch.float32)
            self.assertEqual(moved["halo_mask"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
