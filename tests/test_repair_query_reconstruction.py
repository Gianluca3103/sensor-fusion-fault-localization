import unittest

import torch

from models.reconstruction_head import (
    BEVGridGeometry,
    CoarseReconstructionConfig,
    CoarseReconstructionModel,
    PointPillarsConfig,
    RepairQueryConfig,
    RepairQueryDecoderBlock,
    build_configs,
    load_config,
)


def _geometry():
    return BEVGridGeometry(0.0, 64.0, -32.0, 32.0, 320, 320)


def _small_model():
    pointpillars = PointPillarsConfig(
        enabled=True,
        output_channels=8,
        max_points_per_pillar=10,
        max_pillars=100,
    )
    repair_query = RepairQueryConfig(
        token_dim=8,
        num_heads=2,
        num_decoder_blocks=3,
        mlp_hidden_dim=16,
        region_size_cells=12,
        context_region_radius=1,
    )
    config = CoarseReconstructionConfig(
        backbone="repair_query",
        lidar_channels=8,
        radar_channels=8,
        pointpillars=pointpillars,
        repair_query=repair_query,
    )
    return CoarseReconstructionModel(config, grid_geometry=_geometry())


class RepairQueryReconstructionTests(unittest.TestCase):
    def test_dedicated_config_selects_repair_query_without_changing_defaults(self):
        payload = load_config(
            "configs/coarse_reconstruction_pointpillars_repair_query.json"
        )
        model_config, _, _ = build_configs(payload)
        self.assertEqual(model_config.backbone, "repair_query")
        self.assertEqual(model_config.repair_query.token_dim, 128)
        self.assertEqual(model_config.repair_query.num_decoder_blocks, 3)
        self.assertEqual(model_config.repair_query.num_heads, 8)
        self.assertEqual(model_config.repair_query.region_size_cells, 12)

        baseline_config, _, _ = build_configs(
            load_config("configs/coarse_reconstruction_pointpillars.json")
        )
        self.assertEqual(baseline_config.backbone, "unet")

    def test_local_attention_respects_radius_and_batch_boundaries(self):
        torch.manual_seed(7)
        config = RepairQueryConfig(
            token_dim=8,
            num_heads=2,
            num_decoder_blocks=1,
            mlp_hidden_dim=16,
            region_size_cells=4,
            context_region_radius=1,
        )
        block = RepairQueryDecoderBlock(config, block_index=0).eval()
        queries = torch.randn(2, 8)
        query_coordinates = torch.tensor([[0, 2, 2], [1, 2, 2]])
        context_coordinates = torch.tensor(
            [
                [0, 3, 3],
                [0, 30, 30],
                [1, 3, 3],
            ]
        )
        context = torch.randn(3, 8)
        changed = context.clone()
        changed[1] += 100.0  # Same batch, outside the 3x3 region window.
        changed[2] -= 100.0  # Same XY, different sample.
        with torch.no_grad():
            original, _ = block(
                queries,
                query_coordinates,
                context,
                context_coordinates,
                (40, 40),
            )
            modified, _ = block(
                queries,
                query_coordinates,
                changed,
                context_coordinates,
                (40, 40),
            )
        torch.testing.assert_close(original[0], modified[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[1], modified[1]))

    def test_shifted_query_attention_alternates_by_block(self):
        model = _small_model()
        shifts = [
            block.shift_size_cells
            for block in model.repair_query_decoder.blocks
        ]
        self.assertEqual(shifts, [0, 6, 0])

    def test_attention_supports_mixed_precision_packing(self):
        config = RepairQueryConfig(
            token_dim=8,
            num_heads=2,
            num_decoder_blocks=1,
            mlp_hidden_dim=16,
            region_size_cells=4,
            context_region_radius=1,
        )
        block = RepairQueryDecoderBlock(config, block_index=0).train()
        queries = torch.randn(3, 8, requires_grad=True)
        query_coordinates = torch.tensor(
            [[0, 2, 2], [0, 2, 3], [0, 7, 7]]
        )
        context = torch.randn(3, 8, requires_grad=True)
        context_coordinates = torch.tensor(
            [[0, 1, 1], [0, 5, 5], [0, 15, 15]]
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output, _ = block(
                queries,
                query_coordinates,
                context,
                context_coordinates,
                (20, 20),
            )
            loss = output.square().mean()
        loss.backward()
        self.assertEqual(tuple(output.shape), (3, 8))
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(queries.grad)
        self.assertIsNotNone(context.grad)

    def test_full_model_filters_raw_lidar_and_updates_only_repair_cells(self):
        torch.manual_seed(11)
        model = _small_model().train()
        faulty = torch.rand(1, 3, 320, 320)
        radar_bev = torch.zeros(1, 4, 320, 320)
        reconstruction = torch.zeros(1, 1, 320, 320)
        reconstruction[:, :, 293:297, 159:163] = 1.0
        healthy = torch.zeros_like(reconstruction)
        halo = torch.zeros_like(reconstruction)
        halo[:, :, 287:303, 153:169] = 1.0 - reconstruction[
            :, :, 287:303, 153:169
        ]
        lidar_points = (
            torch.tensor(
                [
                    [5.00, 0.00, 0.0, 0.1],  # Inside repair: must be erased.
                    [5.05, 0.02, 0.1, 0.2],  # Inside repair: must be erased.
                    [6.50, 0.00, 0.2, 0.3],  # Trusted local context.
                    [6.55, 0.02, 0.3, 0.4],
                ]
            ),
        )
        radar_points = (
            torch.tensor(
                [
                    [5.00, 0.00, 0.0, 0.4, -0.2],  # Radar remains in repair.
                    [5.05, 0.02, 0.1, 0.3, 0.1],
                    [6.50, 0.00, 0.2, 0.2, 0.3],
                    [6.55, 0.02, 0.3, 0.1, -0.1],
                ]
            ),
        )
        outputs = model(
            faulty,
            radar_bev,
            reconstruction,
            healthy,
            halo,
            faulty_lidar_points=lidar_points,
            radar_points=radar_points,
            profile_sst=True,
        )

        self.assertEqual(tuple(outputs["coarse_lidar_bev"].shape), (1, 3, 320, 320))
        self.assertEqual(len(outputs["repair_query_coordinates"]), 16)
        expected = torch.nonzero(reconstruction[:, 0] > 0.5, as_tuple=False)
        self.assertTrue(torch.equal(outputs["repair_query_coordinates"], expected))

        trusted = outputs["trusted_lidar_coordinates"]
        self.assertGreater(len(trusted), 0)
        self.assertEqual(
            int(
                reconstruction[
                    trusted[:, 0], 0, trusted[:, 1], trusted[:, 2]
                ].sum()
            ),
            0,
        )
        radar_coordinates = outputs["radar_context_coordinates"]
        self.assertGreater(
            int(
                reconstruction[
                    radar_coordinates[:, 0],
                    0,
                    radar_coordinates[:, 1],
                    radar_coordinates[:, 2],
                ].sum()
            ),
            0,
        )
        self.assertEqual(
            int(
                outputs["repair_query_statistics"][
                    "trusted_lidar_tokens_inside_repair_mask"
                ]
            ),
            0,
        )

        outside = 1.0 - reconstruction
        self.assertTrue(
            torch.equal(outputs["coarse_lidar_bev"] * outside, faulty * outside)
        )
        self.assertEqual(
            int((outputs["replacement_raw"] * outside).count_nonzero()), 0
        )

        loss = outputs["replacement_raw"].sum()
        loss.backward()
        modules = [
            model.lidar_pillar_encoder.feature_net,
            model.radar_pillar_encoder.feature_net,
            model.repair_query_decoder.token_builder.context_projection,
            model.repair_query_decoder.position_encoder,
            model.repair_query_decoder.token_builder.radar_query_projection,
            *model.repair_query_decoder.blocks,
            model.repair_query_decoder.occupancy_head,
            model.repair_query_decoder.density_head,
            model.repair_query_decoder.height_head,
        ]
        for module in modules:
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(
                sum(float(gradient.abs().sum()) for gradient in gradients),
                0.0,
            )
        embedding_gradient = (
            model.repair_query_decoder.token_builder.repair_embedding.grad
        )
        self.assertIsNotNone(embedding_gradient)
        self.assertGreater(float(embedding_gradient.abs().sum()), 0.0)

        required_timings = {
            "pointpillars_ms",
            "token_construction_ms",
            "query_self_attention_ms",
            "cross_attention_ms",
            "output_scatter_ms",
            "complete_forward_ms",
        }
        self.assertEqual(
            set(outputs["repair_query_timing_ms"]), required_timings
        )

    def test_empty_repair_mask_is_an_exact_no_op(self):
        model = _small_model().eval()
        faulty = torch.rand(1, 3, 320, 320)
        radar_bev = torch.zeros(1, 4, 320, 320)
        empty = torch.zeros(1, 1, 320, 320)
        lidar_points = (
            torch.tensor([[6.5, 0.0, 0.2, 0.3], [6.55, 0.02, 0.3, 0.4]]),
        )
        radar_points = (
            torch.tensor(
                [
                    [5.0, 0.0, 0.0, 0.4, -0.2],
                    [5.1, 0.1, 0.1, 0.3, 0.1],
                ]
            ),
        )
        with torch.no_grad():
            outputs = model(
                faulty,
                radar_bev,
                empty,
                empty,
                empty,
                faulty_lidar_points=lidar_points,
                radar_points=radar_points,
            )
        self.assertEqual(len(outputs["repair_query_coordinates"]), 0)
        self.assertTrue(torch.equal(outputs["coarse_lidar_bev"], faulty))


if __name__ == "__main__":
    unittest.main()
