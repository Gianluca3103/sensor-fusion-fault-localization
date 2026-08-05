import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from models.reconstruction_head import (
    BEVTripletDataset,
    FaultSelector,
    FaultSelectorConfig,
    ReconstructionBEVEncoders,
    RequiredCorrection,
    TripletBEVEncoders,
    mask_unreliable_lidar,
)


class ReconstructionEncoderTests(unittest.TestCase):
    def test_triplet_encoders_produce_matching_independent_pyramids(self):
        encoders = TripletBEVEncoders(
            base_channels=4,
            channel_multipliers=(1, 2, 4),
        ).eval()
        with torch.no_grad():
            output = encoders(
                torch.zeros(2, 3, 65, 63),
                torch.zeros(2, 4, 65, 63),
                torch.zeros(2, 3, 65, 63),
                torch.ones(2, 1, 65, 63),
                torch.ones(2, 1, 65, 63),
            )

        expected_shapes = ((2, 4, 65, 63), (2, 8, 33, 32), (2, 16, 17, 16))
        for modality in ("clean", "radar", "faulty", "lidar_trusted"):
            self.assertEqual(
                tuple(tuple(feature.shape) for feature in output[modality].features),
                expected_shapes,
            )
        self.assertEqual(
            tuple(output["good_data"].shape),
            (2, 32, 17, 16),
        )
        trusted_latent = output["lidar_trusted"].bottleneck
        radar_latent = output["radar"].bottleneck
        self.assertTrue(torch.equal(output["good_data"][:, :16], trusted_latent))
        self.assertTrue(torch.equal(output["good_data"][:, 16:], radar_latent))
        self.assertEqual(tuple(output["required_correction"].shape), (2, 16, 17, 16))
        self.assertEqual(tuple(output["latent_repair_mask"].shape), (2, 1, 17, 16))
        self.assertIsNot(encoders.clean, encoders.faulty)
        self.assertIsNot(
            next(encoders.clean.parameters()),
            next(encoders.faulty.parameters()),
        )
        self.assertIsNot(encoders.faulty, encoders.lidar_trusted)
        self.assertIsNot(
            next(encoders.faulty.parameters()),
            next(encoders.lidar_trusted.parameters()),
        )
        self.assertIs(TripletBEVEncoders, ReconstructionBEVEncoders)

    def test_triplet_encoder_rejects_misaligned_inputs(self):
        encoders = TripletBEVEncoders(base_channels=2, channel_multipliers=(1, 2))
        with self.assertRaisesRegex(ValueError, "must share a spatial shape"):
            encoders(
                torch.zeros(1, 3, 32, 32),
                torch.zeros(1, 4, 16, 32),
                torch.zeros(1, 3, 32, 32),
                torch.ones(1, 1, 32, 32),
                torch.ones(1, 1, 32, 32),
            )

    def test_required_correction_is_clean_minus_faulty_inside_repair_box(self):
        clean_latent = torch.ones(1, 2, 4, 4)
        faulty_latent = torch.full((1, 2, 4, 4), 0.25)
        clean = type("Encoding", (), {"bottleneck": clean_latent})()
        faulty = type("Encoding", (), {"bottleneck": faulty_latent})()
        repair_mask = torch.zeros(1, 1, 8, 8)
        repair_mask[:, :, :4, :4] = 1.0

        correction, latent_mask = RequiredCorrection()(
            clean,
            faulty,
            repair_mask,
        )

        self.assertTrue(torch.all(latent_mask[:, :, :2, :2] == 1.0))
        self.assertTrue(torch.all(latent_mask[:, :, 2:, :] == 0.0))
        self.assertTrue(torch.all(latent_mask[:, :, :, 2:] == 0.0))
        self.assertTrue(torch.all(correction[:, :, :2, :2] == 0.75))
        self.assertTrue(torch.all(correction[:, :, 2:, :] == 0.0))
        self.assertTrue(torch.all(correction[:, :, :, 2:] == 0.0))

    def test_trusted_lidar_removes_every_unreliable_cell(self):
        faulty = torch.arange(36, dtype=torch.float32).reshape(1, 3, 3, 4)
        reliability = torch.ones(1, 1, 3, 4)
        reliability[:, :, 0, 1] = 0.0
        reliability[:, :, 2, 3] = 0.75

        trusted = mask_unreliable_lidar(faulty, reliability)

        self.assertTrue(torch.all(trusted[:, :, 0, 1] == 0.0))
        self.assertTrue(torch.all(trusted[:, :, 2, 3] == 0.0))
        retained = reliability.expand_as(faulty) == 1.0
        self.assertTrue(torch.equal(trusted[retained], faulty[retained]))

    def test_dataset_loads_and_normalizes_existing_artifact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.npz"
            radar_root = root / "radar"
            radar_path = radar_root / "Scene01" / "01_Day" / "123.npz"
            radar_path.parent.mkdir(parents=True)
            metadata = {
                "scene": "Scene01",
                "session": "01_Day",
                "timestamp": "123",
            }
            np.savez_compressed(
                sample_path,
                clean_rgb=np.full((8, 6, 3), 255, dtype=np.uint8),
                faulty_rgb=np.zeros((8, 6, 3), dtype=np.uint8),
                fault_heatmap=np.full((8, 6), 0.25, dtype=np.float32),
                reliability_map=np.full((8, 6), 0.75, dtype=np.float32),
                faulty_counts=np.ones((8, 6), dtype=np.float32),
                added_faulty_counts=np.zeros((8, 6), dtype=np.int32),
                missing_faulty_counts=np.ones((8, 6), dtype=np.int32),
                moved_faulty_counts=np.zeros((8, 6), dtype=np.int32),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            np.savez_compressed(
                radar_path,
                radar_bev=np.full((4, 8, 6), 0.5, dtype=np.float32),
            )

            item = BEVTripletDataset(
                [sample_path], radar_root, resize_hw=(10, 12)
            )[0]

        self.assertEqual(item["clean_bev"].shape, (3, 10, 12))
        self.assertEqual(item["radar_bev"].shape, (4, 10, 12))
        self.assertEqual(item["faulty_bev"].shape, (3, 10, 12))
        self.assertEqual(item["lidar_trusted_bev"].shape, (3, 10, 12))
        self.assertEqual(item["fault_heatmap"].shape, (1, 10, 12))
        self.assertEqual(item["reliability_map"].shape, (1, 10, 12))
        self.assertEqual(item["faulty_counts"].shape, (1, 10, 12))
        self.assertEqual(item["missing_faulty_counts"].shape, (1, 10, 12))
        self.assertTrue(torch.all(item["clean_bev"] == 1.0))
        self.assertTrue(torch.all(item["radar_bev"] == 0.5))
        self.assertTrue(torch.all(item["faulty_bev"] == 0.0))
        self.assertTrue(torch.all(item["lidar_trusted_bev"] == 0.0))

    def test_fault_selector_rejects_isolated_cells_and_combines_major_regions(self):
        heatmap = np.zeros((40, 20), dtype=np.float32)
        heatmap[34:37, 2:4] = 0.8  # Six-cell near blob.
        heatmap[20:24, 8:11] = 0.7  # Larger middle-distance blob.
        heatmap[2:5, 14:17] = 1.0  # Nine-cell far blob.
        heatmap[39, 19] = 1.0  # Isolated closest cell; must be rejected.
        zeros = np.zeros_like(heatmap)
        missing = (heatmap > 0).astype(np.float32)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=5,
                max_blobs=2,
                merge_radius_cells=0,
                box_padding_cells=0,
                distance_bin_m=2.0,
                x_cell_size_m=0.2,
                min_repair_fault_fraction=0.0,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
            )
        ).select(
            heatmap,
            added_faulty_counts=zeros,
            missing_faulty_counts=missing,
            moved_faulty_counts=zeros,
        )

        self.assertEqual([blob.cell_count for blob in selection.selected_blobs], [27])
        self.assertEqual(selection.selected_fault_cell_count, 27)
        self.assertEqual(len(selection.rejected_small_blobs), 1)
        self.assertFalse(selection.reconstruction_mask[39, 19])
        self.assertTrue(selection.reconstruction_mask[35, 2])
        self.assertTrue(selection.reconstruction_mask[22, 9])
        self.assertTrue(selection.reconstruction_mask[3, 15])

    def test_fault_selector_prefers_larger_blob_within_distance_bin(self):
        heatmap = np.zeros((20, 20), dtype=np.float32)
        heatmap[15:17, 1:3] = 1.0
        heatmap[13:16, 10:13] = 0.6
        zeros = np.zeros_like(heatmap)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=1,
                max_blobs=1,
                merge_radius_cells=0,
                box_padding_cells=0,
                combine_gap_cells=0,
                distance_bin_m=10.0,
                x_cell_size_m=0.2,
                min_repair_fault_fraction=0.0,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
            )
        ).select(
            heatmap,
            added_faulty_counts=zeros,
            missing_faulty_counts=(heatmap > 0).astype(np.float32),
            moved_faulty_counts=zeros,
        )

        self.assertEqual(selection.selected_blobs[0].cell_count, 9)

    def test_fault_selector_excludes_added_only_cells_but_keeps_mixed_cells(self):
        heatmap = np.zeros((12, 12), dtype=np.float32)
        heatmap[8:11, 1:4] = 1.0
        heatmap[2:5, 7:10] = 0.8
        added = np.zeros_like(heatmap)
        missing = np.zeros_like(heatmap)
        moved = np.zeros_like(heatmap)
        added[8:11, 1:4] = 2
        added[2:5, 7:10] = 1
        missing[2:5, 7:10] = 1

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=1,
                max_blobs=None,
                merge_radius_cells=0,
                box_padding_cells=0,
                min_repair_fault_fraction=0.0,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
            )
        ).select(
            heatmap,
            added_faulty_counts=added,
            missing_faulty_counts=missing,
            moved_faulty_counts=moved,
        )

        self.assertEqual(selection.original_fault_cell_count, 18)
        self.assertEqual(selection.excluded_added_only_cell_count, 9)
        self.assertEqual(selection.thresholded_cell_count, 9)
        self.assertFalse(selection.reconstruction_mask[9, 2])
        self.assertTrue(selection.reconstruction_mask[3, 8])

    def test_fault_selector_merges_nearby_missing_fragments_into_one_rectangle(self):
        heatmap = np.zeros((40, 40), dtype=np.float32)
        heatmap[5:8, 5:8] = 1.0
        heatmap[5:8, 13:16] = 1.0
        heatmap[35, 35] = 1.0
        zeros = np.zeros_like(heatmap)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=5,
                max_blobs=1,
                merge_radius_cells=3,
                box_padding_cells=1,
                min_repair_fault_fraction=0.0,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
            )
        ).select(
            heatmap,
            added_faulty_counts=zeros,
            missing_faulty_counts=(heatmap > 0).astype(np.float32),
            moved_faulty_counts=zeros,
        )

        self.assertEqual(selection.selected_fault_cell_count, 18)
        self.assertEqual(selection.selected_blobs[0].bbox, (4, 4, 9, 17))
        self.assertEqual(selection.selected_cell_count, 5 * 13)
        self.assertEqual(len(selection.rejected_small_blobs), 1)

    def test_fault_selector_ignores_sparse_satellite_of_dominant_region(self):
        heatmap = np.zeros((80, 80), dtype=np.float32)
        heatmap[5:15, 5:15] = 1.0
        heatmap[65:67, 65:68] = 1.0
        zeros = np.zeros_like(heatmap)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=5,
                max_blobs=3,
                merge_radius_cells=0,
                combine_gap_cells=5,
                min_relative_blob_size=0.1,
                box_padding_cells=0,
                bbox_quantile=0.0,
                min_repair_fault_fraction=0.0,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
            )
        ).select(
            heatmap,
            added_faulty_counts=zeros,
            missing_faulty_counts=(heatmap > 0).astype(np.float32),
            moved_faulty_counts=zeros,
        )

        self.assertEqual(len(selection.selected_blobs), 1)
        self.assertEqual(selection.selected_blobs[0].cell_count, 100)
        self.assertFalse(selection.reconstruction_mask[65, 65])

    def test_fault_selector_builds_healthy_halo_around_repair_box(self):
        heatmap = np.zeros((40, 40), dtype=np.float32)
        heatmap[15:25, 15:25] = 1.0
        reliability = 1.0 - heatmap
        faulty_counts = np.zeros_like(heatmap)
        faulty_counts[13:27, 13:27] = 1.0
        zeros = np.zeros_like(heatmap)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=1,
                max_blobs=1,
                merge_radius_cells=0,
                combine_gap_cells=0,
                box_padding_cells=0,
                bbox_quantile=0.0,
                min_repair_fault_fraction=0.75,
                min_halo_healthy_fraction=0.90,
                min_halo_width_cells=1,
            )
        ).select(
            heatmap,
            reliability_map=reliability,
            faulty_counts=faulty_counts,
            added_faulty_counts=zeros,
            missing_faulty_counts=heatmap,
            moved_faulty_counts=zeros,
        )

        blob = selection.selected_blobs[0]
        self.assertEqual(blob.bbox, (15, 15, 25, 25))
        self.assertEqual(blob.halo_bbox, (13, 13, 27, 27))
        self.assertEqual(blob.halo_dilation_cells, 2)
        self.assertTrue(blob.repair_target_met)
        self.assertTrue(blob.halo_target_met)
        self.assertGreaterEqual(blob.repair_fault_fraction, 0.75)
        self.assertGreaterEqual(blob.halo_healthy_fraction, 0.90)
        self.assertEqual(selection.selected_cell_count, 100)
        self.assertEqual(blob.required_healthy_context_cell_count, 64)
        self.assertEqual(selection.halo_cell_count, 96)
        self.assertEqual(selection.healthy_context_cell_count, 96)
        self.assertFalse(
            np.logical_and(selection.reconstruction_mask, selection.halo_mask).any()
        )

    def test_fault_selector_does_not_count_reliable_empty_cells_as_context(self):
        heatmap = np.zeros((20, 20), dtype=np.float32)
        heatmap[8:12, 8:12] = 1.0
        reliability = 1.0 - heatmap
        faulty_counts = np.zeros_like(heatmap)
        faulty_counts[8:12, 8:12] = 1.0
        zeros = np.zeros_like(heatmap)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=1,
                max_blobs=1,
                merge_radius_cells=0,
                combine_gap_cells=0,
                box_padding_cells=0,
                bbox_quantile=0.0,
                min_repair_fault_fraction=0.75,
                min_halo_healthy_fraction=0.90,
                min_halo_width_cells=1,
                max_halo_dilation_cells=4,
            )
        ).select(
            heatmap,
            reliability_map=reliability,
            faulty_counts=faulty_counts,
            added_faulty_counts=zeros,
            missing_faulty_counts=heatmap,
            moved_faulty_counts=zeros,
        )

        blob = selection.selected_blobs[0]
        self.assertFalse(blob.halo_target_met)
        self.assertEqual(blob.healthy_occupied_cell_count, 0)
        self.assertEqual(selection.healthy_context_cell_count, 0)

    def test_fault_selector_trims_repair_box_to_fault_fraction_target(self):
        heatmap = np.zeros((30, 30), dtype=np.float32)
        heatmap[8:22, 8] = 1.0
        heatmap[8:22, 21] = 1.0
        heatmap[8, 8:22] = 1.0
        heatmap[21, 8:22] = 1.0
        reliability = 1.0 - heatmap
        faulty_counts = np.ones_like(heatmap)
        zeros = np.zeros_like(heatmap)

        selection = FaultSelector(
            FaultSelectorConfig(
                min_blob_cells=1,
                max_blobs=1,
                merge_radius_cells=0,
                combine_gap_cells=20,
                box_padding_cells=0,
                bbox_quantile=0.0,
                min_repair_fault_fraction=0.75,
                min_halo_healthy_fraction=0.0,
                min_halo_healthy_cells=0,
                min_halo_context_ratio=0.0,
                min_halo_width_cells=1,
            )
        ).select(
            heatmap,
            reliability_map=reliability,
            faulty_counts=faulty_counts,
            added_faulty_counts=zeros,
            missing_faulty_counts=heatmap,
            moved_faulty_counts=zeros,
        )

        blob = selection.selected_blobs[0]
        self.assertTrue(blob.repair_target_met)
        self.assertGreaterEqual(blob.repair_fault_fraction, 0.75)
        self.assertLess(selection.selected_cell_count, 14 * 14)


if __name__ == "__main__":
    unittest.main()
