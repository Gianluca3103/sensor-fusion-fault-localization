import unittest
from unittest import mock

import torch

from models.two_stage_reconstruction_head.diffusion_process.local_diffusion import (
    FineDiffusionConfig,
    FineDiffusionRefiner,
    MaskedExactReconstructionLoss,
    MaskedNoDegradationLoss,
    ReconstructionCropExtractor,
    WindowAttention2d,
    fine_diffusion_architecture_metadata,
    validate_fine_diffusion_checkpoint_compatibility,
)
from models.two_stage_reconstruction_head.diffusion_process.diffusion_process import (
    ResidualChannelNormalization,
)
from models.two_stage_reconstruction_head.diffusion_process.residual_statistics import (
    estimate_training_residual_statistics,
)
from models.two_stage_reconstruction_head.diffusion_process.diffusion_pipeline import (
    FrozenCoarseFineDiffusionPipeline,
    load_frozen_coarse_model,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry
from models.two_stage_reconstruction_head.reconstruction_inputs import (
    ReconstructionInputs,
)


def _config(*, global_context=True, sampling_steps=3, bypass_coarse=False):
    return FineDiffusionConfig(
        bypass_coarse_reconstruction=bypass_coarse,
        hidden_dim=16,
        num_heads=4,
        num_transformer_blocks=2,
        window_size=4,
        use_global_faulty_context=global_context,
        global_context_dim=16,
        training_timesteps=12,
        sampling_steps=sampling_steps,
    )


def _inputs(batch=2, size=16):
    torch.manual_seed(4)
    faulty = torch.rand(batch, 3, size, size)
    coarse = faulty.clone()
    clean = faulty.clone()
    radar = torch.rand(batch, 4, size, size)
    repair = torch.zeros(batch, 1, size, size)
    halo = torch.zeros_like(repair)
    repair[0, :, 5:8, 6:10] = 1
    halo[0, :, 3:10, 4:12] = 1
    halo *= 1 - repair
    if batch > 1:
        repair[1, :, 0:2, 0:3] = 1
        halo[1, :, 0:4, 0:5] = 1
        halo *= 1 - repair
    clean = (clean + repair * 0.1).clamp(0, 1)
    coarse = (coarse - repair * 0.1).clamp(0, 1)
    return clean, coarse, faulty, radar, repair, halo


class ReconstructionCropExtractorTests(unittest.TestCase):
    def test_exact_union_bounds_radar_alignment_and_padding(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        extractor = ReconstructionCropExtractor(4)
        crops = extractor.extract(
            {"clean": clean, "coarse": coarse, "radar": radar},
            repair,
            halo,
        )
        self.assertEqual(crops.boxes.tolist(), [[3, 10, 4, 12]])
        self.assertEqual(tuple(crops.valid_mask.shape), (1, 1, 8, 8))
        self.assertTrue(
            torch.equal(crops.tensors["clean"][:, :, :7, :8], clean[:, :, 3:10, 4:12])
        )
        self.assertTrue(
            torch.equal(crops.tensors["radar"][:, :, :7, :8], radar[:, :, 3:10, 4:12])
        )
        self.assertEqual(int(crops.valid_mask[:, :, 7].count_nonzero()), 0)

    def test_repair_only_crop_has_no_margin_and_padding_is_invalid(self):
        repair = torch.zeros(1, 1, 16, 16)
        repair[:, :, 5:7, 6:9] = 1
        halo = torch.zeros_like(repair)
        radar = torch.arange(4 * 16 * 16, dtype=torch.float32).reshape(
            1, 4, 16, 16
        )

        crops = ReconstructionCropExtractor(4).extract(
            {"radar": radar, "repair": repair}, repair, halo
        )

        self.assertEqual(crops.boxes.tolist(), [[5, 7, 6, 9]])
        self.assertEqual(tuple(crops.valid_mask.shape), (1, 1, 4, 4))
        self.assertTrue(
            torch.equal(crops.tensors["radar"][:, :, :2, :3], radar[:, :, 5:7, 6:9])
        )
        self.assertEqual(int(crops.valid_mask.sum()), 6)
        self.assertEqual(int(crops.valid_mask[:, :, 2:, :].count_nonzero()), 0)
        self.assertEqual(int(crops.valid_mask[:, :, :, 3:].count_nonzero()), 0)

    def test_empty_and_every_border_are_safe(self):
        for region in (
            (0, 2, 5, 8),
            (14, 16, 5, 8),
            (5, 8, 0, 2),
            (5, 8, 14, 16),
            (0, 1, 0, 1),
        ):
            tensor = torch.zeros(1, 1, 16, 16)
            tensor[:, :, region[0] : region[1], region[2] : region[3]] = 1
            crops = ReconstructionCropExtractor(4).extract(
                {"value": tensor}, tensor, torch.zeros_like(tensor)
            )
            top, bottom, left, right = crops.boxes[0].tolist()
            self.assertGreaterEqual(top, 0)
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(bottom, 16)
            self.assertLessEqual(right, 16)
        empty = torch.zeros(1, 1, 16, 16)
        crops = ReconstructionCropExtractor(4).extract(
            {"value": empty}, empty, empty
        )
        self.assertFalse(bool(crops.active_samples[0]))


class FineDiffusionRefinerTests(unittest.TestCase):
    def test_default_sampling_is_forward_three_step_residual_flow(self):
        config = FineDiffusionConfig()
        model = FineDiffusionRefiner(config)
        timesteps = model._sampling_timesteps(3, torch.device("cpu"))
        self.assertEqual(config.sampling_steps, 3)
        self.assertEqual(int(timesteps[0]), 0)
        self.assertEqual(int(timesteps[-1]), 667)
        self.assertEqual(len(timesteps), 3)

    def test_sampling_flow_always_starts_from_zero_progress(self):
        model = FineDiffusionRefiner(
            FineDiffusionConfig(
                hidden_dim=16,
                num_heads=4,
                num_transformer_blocks=1,
                window_size=4,
                training_timesteps=1000,
                sampling_steps=10,
                global_context_dim=16,
            )
        )

        timesteps = model._sampling_timesteps(10, torch.device("cpu"))

        self.assertEqual(int(timesteps[0]), 0)
        self.assertEqual(int(timesteps[-1]), 900)
        self.assertEqual(len(timesteps), 10)

    def test_sampling_timesteps_support_full_50_step_schedule(self):
        model = FineDiffusionRefiner(
            FineDiffusionConfig(
                hidden_dim=16,
                num_heads=4,
                num_transformer_blocks=1,
                window_size=4,
                training_timesteps=1000,
                sampling_steps=50,
                global_context_dim=16,
            )
        )

        timesteps = model._sampling_timesteps(50, torch.device("cpu"))

        self.assertEqual(int(timesteps[0]), 0)
        self.assertEqual(int(timesteps[-1]), 980)
        self.assertEqual(len(timesteps), 50)

    def test_initial_sampling_state_is_exactly_coarse(self):
        _clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config(sampling_steps=3)).eval()
        generator = torch.Generator().manual_seed(17)
        output = model.sample(
            coarse,
            faulty,
            radar,
            repair,
            halo,
            sampling_steps=3,
            generator=generator,
            return_debug=True,
        )

        self.assertEqual(int(output["initial_residual"].count_nonzero()), 0)
        self.assertTrue(torch.equal(output["initial_lidar_bev"], coarse))

    def test_zero_initialized_refiner_preserves_coarse_inside_repair_mask(self):
        _clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config(sampling_steps=3)).eval()

        output = model.sample(
            coarse, faulty, radar, repair, halo, sampling_steps=3
        )

        self.assertEqual(int(output["predicted_residual"].count_nonzero()), 0)
        self.assertTrue(
            torch.equal(
                output["final_lidar_bev"] * repair,
                coarse * repair,
            )
        )
        self.assertTrue(
            torch.equal(
                output["final_lidar_bev"] * (1 - repair),
                faulty * (1 - repair),
            )
        )

    def test_training_trajectory_starts_from_coarse_without_teacher_state(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config(sampling_steps=3)).train()

        output = model(
            clean,
            coarse,
            faulty,
            radar,
            repair,
            halo,
            return_debug=True,
        )

        debug = output["debug"]
        self.assertEqual(
            len(debug["training_intermediate_residuals"]), 3
        )
        self.assertTrue(
            all(
                int(state.count_nonzero()) == 0
                for state in debug["training_intermediate_residuals"]
            )
        )
        self.assertTrue(
            torch.equal(output["final_lidar_bev"] * repair, coarse * repair)
        )

    def test_no_degradation_loss_only_penalizes_worse_refinement(self):
        loss = MaskedNoDegradationLoss()
        clean = torch.zeros(1, 3, 2, 2)
        clean[:, 0] = 1.0
        clean[:, 1:] = 0.8
        coarse = clean.clone()
        coarse[:, 0] = 0.6
        coarse[:, 1:] = 0.4
        mask = torch.ones(1, 1, 2, 2)

        self.assertAlmostEqual(float(loss(coarse, coarse, clean, mask)), 0.0)

        better = clean.clone()
        better[:, 0] = 0.9
        better[:, 1:] = 0.7
        self.assertAlmostEqual(float(loss(better, coarse, clean, mask)), 0.0)

        worse = clean.clone()
        worse[:, 0] = 0.2
        worse[:, 1:] = 0.1
        self.assertGreater(float(loss(worse, coarse, clean, mask)), 0.0)

    def test_losses_ignore_non_finite_values_outside_repair_mask(self):
        exact = MaskedExactReconstructionLoss()
        degradation = MaskedNoDegradationLoss()
        clean = torch.zeros(1, 3, 2, 2)
        coarse = torch.full_like(clean, 0.25)
        refined = coarse.clone()
        mask = torch.zeros(1, 1, 2, 2)
        mask[:, :, 0, 0] = 1.0
        refined[:, :, 1, 1] = torch.nan

        self.assertTrue(torch.isfinite(exact(refined, clean, mask)))
        self.assertTrue(
            torch.isfinite(degradation(refined, coarse, clean, mask))
        )

    def test_loss_reports_non_finite_value_inside_repair_mask(self):
        loss = MaskedExactReconstructionLoss()
        clean = torch.zeros(1, 3, 2, 2)
        refined = torch.zeros_like(clean)
        mask = torch.zeros(1, 1, 2, 2)
        mask[:, :, 0, 0] = 1.0
        refined[:, :, 0, 0] = torch.nan

        with self.assertRaisesRegex(
            FloatingPointError, "non-finite inside the reconstruction mask"
        ):
            loss(refined, clean, mask)

    def test_bounded_update_clears_non_finite_values_outside_repair(self):
        model = FineDiffusionRefiner(_config())
        current = torch.zeros(1, 3, 2, 2)
        predicted = torch.zeros_like(current)
        predicted[:, :, 1, 1] = torch.nan
        coarse = torch.full_like(current, 0.25)
        repair = torch.zeros(1, 1, 2, 2)
        repair[:, :, 0, 0] = 1.0

        updated = model._bounded_residual_update(
            current, predicted, coarse, repair
        )

        self.assertTrue(torch.isfinite(updated).all())
        self.assertEqual(int(updated.count_nonzero()), 0)

    def test_residual_regularizer_is_masked_and_magnitude_sensitive(self):
        model = FineDiffusionRefiner(_config())
        mask = torch.zeros(1, 1, 2, 2)
        mask[:, :, 0, 0] = 1.0
        zero = torch.zeros(1, 3, 2, 2)
        small = zero.clone()
        large = zero.clone()
        outside_only = zero.clone()
        small[:, :, 0, 0] = 0.1
        large[:, :, 0, 0] = 0.5
        outside_only[:, :, 1, 1] = 100.0

        self.assertAlmostEqual(
            float(model._residual_regularization_loss(zero, mask)), 0.0
        )
        self.assertGreater(
            float(model._residual_regularization_loss(large, mask)),
            float(model._residual_regularization_loss(small, mask)),
        )
        self.assertAlmostEqual(
            float(model._residual_regularization_loss(outside_only, mask)),
            0.0,
        )

    def test_window_attention_tolerates_amp_output_dtype(self):
        attention = WindowAttention2d(
            hidden_dim=4, num_heads=2, window_size=2, dropout=0.0
        )
        query = torch.rand(1, 4, 4, 4)
        valid = torch.ones(1, 1, 4, 4)

        class HalfAttention(torch.nn.Module):
            def forward(self, query, key, value, **_kwargs):
                return query.half(), None

        attention.attention = HalfAttention()
        output = attention(query, query, valid)

        self.assertEqual(output.dtype, query.dtype)
        self.assertEqual(output.shape, query.shape)

    def test_training_residual_masking_leakage_and_final_preservation(self):
        clean, coarse, faulty, radar, repair, halo = _inputs()
        distinctive = faulty.clone()
        distinctive = distinctive * (1 - repair) + 99 * repair
        model = FineDiffusionRefiner(_config()).train()
        output = model(
            clean,
            coarse,
            distinctive,
            radar,
            repair,
            halo,
            return_debug=True,
        )
        debug = output["debug"]
        crop_batch = debug["crops"]
        self.assertTrue(
            torch.equal(
                debug["residual_gt_physical"],
                crop_batch.tensors["repair"]
                * (
                    crop_batch.tensors["clean"]
                    - crop_batch.tensors["coarse"]
                ),
            )
        )
        self.assertEqual(
            int((crop_batch.tensors["trusted_faulty"] * crop_batch.tensors["repair"]).count_nonzero()),
            0,
        )
        self.assertEqual(
            int((debug["current_residual"] * (1 - crop_batch.tensors["repair"])).count_nonzero()),
            0,
        )
        outside = 1 - repair
        predicted_residual = crop_batch.paste(
            debug["predicted_residual_physical"]
        ) * repair
        self.assertTrue(
            torch.equal(predicted_residual * outside, torch.zeros_like(coarse))
        )
        self.assertTrue(
            torch.equal(output["final_lidar_bev"] * outside, distinctive * outside)
        )
        self.assertTrue(
            torch.equal(output["final_lidar_bev"] * halo, distinctive * halo)
        )
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_residual_normalization_round_trip_and_zero(self):
        normalizer = ResidualChannelNormalization(
            (0.1, 0.2, 0.4), minimum_std=1.0e-4
        )
        physical = torch.tensor(
            [[[[0.0]], [[-0.3]], [[0.8]]]], dtype=torch.float32
        )
        restored = normalizer.denormalize(normalizer.normalize(physical))
        self.assertTrue(torch.allclose(restored, physical, atol=1.0e-7))
        self.assertEqual(float(restored[0, 0, 0, 0]), 0.0)

    def test_sampling_integrates_denormalized_velocity_in_physical_units(self):
        _clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        normalizer = ResidualChannelNormalization((0.1, 0.2, 0.4))
        model = FineDiffusionRefiner(
            _config(sampling_steps=1),
            residual_normalization=normalizer,
        ).eval()

        def fixed_velocity(residual_t, crops, timestep, global_embedding):
            del timestep, global_embedding
            normalized = torch.ones_like(residual_t) * crops.tensors["repair"]
            return normalized, normalized, {}

        with mock.patch.object(
            model, "_predict_velocity", side_effect=fixed_velocity
        ):
            output = model.sample(
                coarse, faulty, radar, repair, halo, sampling_steps=1
            )
        predicted = output["predicted_residual"]
        for channel, expected in enumerate((0.1, 0.2, 0.4)):
            selected = predicted[:, channel : channel + 1][repair > 0.5]
            expected_values = (
                (coarse[:, channel : channel + 1] + expected).clamp(0.0, 1.0)
                - coarse[:, channel : channel + 1]
            )[repair > 0.5]
            self.assertTrue(
                torch.allclose(selected, expected_values)
            )

    def test_residual_statistics_use_only_masked_training_values(self):
        train = {
            "clean_lidar_bev": torch.tensor(
                [[[[1.0, 3.0], [100.0, 100.0]],
                  [[2.0, 4.0], [100.0, 100.0]],
                  [[3.0, 7.0], [100.0, 100.0]]]]
            ),
            "coarse": torch.zeros(1, 3, 2, 2),
            "reconstruction_mask": torch.tensor(
                [[[[1.0, 1.0], [0.0, 0.0]]]]
            ),
        }
        validation = {
            **train,
            "clean_lidar_bev": torch.full((1, 3, 2, 2), 1000.0),
        }
        statistics = estimate_training_residual_statistics(
            [train],
            move_batch=lambda batch: batch,
            coarse_forward=lambda batch: batch["coarse"],
            channels=3,
            minimum_std=1.0e-4,
        )
        self.assertEqual(statistics["split"], "train")
        self.assertEqual(statistics["channels"][0]["sample_count"], 2)
        self.assertAlmostEqual(statistics["raw_channel_stds"][0], 1.0)
        self.assertAlmostEqual(statistics["raw_channel_stds"][1], 1.0)
        self.assertAlmostEqual(statistics["raw_channel_stds"][2], 2.0)
        self.assertNotEqual(
            statistics["channels"][0]["mean"],
            float(validation["clean_lidar_bev"].mean()),
        )

    def test_coarse_is_only_starting_state_and_radar_drives_cross_attention(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config()).eval()
        output = model(
            clean,
            coarse,
            faulty,
            radar,
            repair,
            halo,
            return_debug=True,
        )
        debug = output["debug"]
        self.assertEqual(debug["current_lidar_stem_input"].shape[1], 7)
        self.assertEqual(
            model.transformer.current_lidar_stem.in_channels, 7
        )
        self.assertFalse(hasattr(model.transformer.blocks[0], "cross_attention"))
        self.assertEqual(
            model.transformer.blocks[0].radar_cross_attention.attention.kdim,
            4,
        )
        self.assertNotIn("raw_coarse_cross_attention", debug)
        self.assertNotIn("coarse_cross_attention_valid", debug)
        self.assertTrue(
            torch.equal(
                debug["raw_radar_cross_attention"],
                debug["crops"].tensors["radar"],
            )
        )
        self.assertEqual(
            tuple(debug["raw_radar_cross_attention"].shape[-2:]),
            tuple(debug["crops"].tensors["radar"].shape[-2:]),
        )
        self.assertTrue(
            torch.equal(
                debug["radar_cross_attention_valid"],
                debug["crops"].tensors["repair"]
                * debug["crops"].valid_mask,
            )
        )
        self.assertTrue(
            torch.equal(
                debug["current_lidar_stem_input"][:, :3],
                debug["current_lidar"],
            )
        )
        expected_current_lidar = debug["crops"].tensors["coarse"] + (
            model.residual_normalization.denormalize(
                debug["current_residual"]
            )
        )
        self.assertTrue(
            torch.equal(debug["current_lidar"], expected_current_lidar)
        )
        self.assertTrue(
            torch.equal(
                debug["current_lidar"], debug["crops"].tensors["coarse"]
            )
        )

    def test_each_refinement_step_sees_updated_current_lidar(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config()).train()
        with torch.no_grad():
            model.transformer.output_head[-1].bias.fill_(0.2)
        stem_inputs = []

        def capture_stem_input(_module, arguments):
            stem_inputs.append(arguments[0][:, :3].detach().clone())

        handle = model.transformer.current_lidar_stem.register_forward_pre_hook(
            capture_stem_input
        )
        try:
            output = model(
                clean,
                coarse,
                faulty,
                radar,
                repair,
                halo,
                return_debug=True,
            )
        finally:
            handle.remove()

        debug = output["debug"]
        coarse_crop = debug["crops"].tensors["coarse"]
        self.assertEqual(len(stem_inputs), 3)
        self.assertTrue(torch.equal(stem_inputs[0], coarse_crop))
        self.assertFalse(torch.equal(stem_inputs[1], stem_inputs[0]))
        expected_second = coarse_crop + model.residual_normalization.denormalize(
            debug["training_intermediate_residuals"][0]
        )
        self.assertTrue(torch.equal(stem_inputs[1], expected_second))

    def test_legacy_residual_stem_checkpoint_has_clear_error(self):
        config = _config()
        legacy = {
            "diffusion_state_dict": {
                "transformer.residual_stem.weight": torch.zeros(
                    config.hidden_dim, config.lidar_channels + 4, 3, 3
                )
            }
        }
        with self.assertRaisesRegex(ValueError, "predates explicit current-LiDAR"):
            validate_fine_diffusion_checkpoint_compatibility(legacy, config)

    def test_new_checkpoint_metadata_and_residual_scale_round_trip(self):
        config = _config()
        model = FineDiffusionRefiner(
            config,
            residual_normalization=ResidualChannelNormalization(
                (0.1, 0.2, 0.3)
            ),
        )
        checkpoint = {
            "diffusion_state_dict": model.state_dict(),
            "fine_diffusion_architecture": fine_diffusion_architecture_metadata(
                config
            ),
            "residual_normalization": model.residual_normalization.metadata(),
        }
        validate_fine_diffusion_checkpoint_compatibility(checkpoint, config)
        restored = ResidualChannelNormalization(
            checkpoint["residual_normalization"]["raw_channel_stds"],
            minimum_std=checkpoint["residual_normalization"]["minimum_std"],
        )
        self.assertTrue(
            torch.equal(
                restored.channel_stds,
                model.residual_normalization.channel_stds,
            )
        )

    def test_inference_needs_no_clean_lidar_and_flow_is_masked(self):
        _clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config()).eval()
        output = model.sample(
            coarse,
            faulty,
            radar,
            repair,
            halo,
            sampling_steps=3,
        )
        self.assertEqual(tuple(output["final_lidar_bev"].shape), tuple(coarse.shape))
        self.assertEqual(
            int((output["predicted_residual"] * (1 - repair)).count_nonzero()), 0
        )
        self.assertTrue(
            torch.equal(output["final_lidar_bev"] * (1 - repair), faulty * (1 - repair))
        )
        self.assertTrue(torch.isfinite(output["final_lidar_bev"]).all())

    def test_empty_mask_skips_diffusion(self):
        _clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        repair.zero_()
        halo.zero_()
        output = FineDiffusionRefiner(_config()).eval().sample(
            coarse, faulty, radar, repair, halo, sampling_steps=1
        )
        self.assertTrue(torch.equal(output["final_lidar_bev"], faulty))
        self.assertEqual(int(output["predicted_residual"].count_nonzero()), 0)

    def test_one_cell_corner_mask_is_stable(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        repair.zero_()
        halo.zero_()
        repair[:, :, -1, -1] = 1
        clean[:, :, -1, -1] = 1
        output = FineDiffusionRefiner(_config(sampling_steps=1)).train()(
            clean,
            coarse,
            faulty,
            radar,
            repair,
            halo,
        )
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_backward_reaches_every_requested_branch(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config()).train()
        output = model(
            clean,
            coarse,
            faulty,
            radar,
            repair,
            halo,
        )
        output["loss"].backward()
        names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
        for expected in (
            "transformer.current_lidar_stem",
            "transformer.auxiliary_condition_encoder",
            "self_attention",
            "cross_attention",
            "radar_cross_attention",
            "ffn",
            "modulation",
            "global_encoder",
            "output_head",
        ):
            self.assertTrue(any(expected in name for name in names), expected)

    def test_optional_global_context_and_large_crop_are_finite(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        repair[:, :, 1:15, 1:15] = 1
        halo.zero_()
        for enabled in (False, True):
            model = FineDiffusionRefiner(_config(global_context=enabled)).train()
            output = model(
                clean,
                coarse,
                faulty,
                radar,
                repair,
                halo,
            )
            self.assertTrue(torch.isfinite(output["loss"]))

    def test_frozen_online_coarse_pipeline_blocks_gradients(self):
        class DummyCoarse(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

            def forward(self, faulty, _radar, repair, _healthy, _halo, **_kwargs):
                coarse = faulty * (1 - repair) + 0.5 * self.scale * repair
                return {"coarse_lidar_bev": coarse}

        clean, _coarse, faulty, radar, repair, halo = _inputs(batch=1)
        pipeline = FrozenCoarseFineDiffusionPipeline(
            DummyCoarse(), FineDiffusionRefiner(_config())
        ).train()
        output = pipeline(
            clean,
            faulty,
            radar,
            repair,
            torch.zeros_like(repair),
            halo,
        )
        output["loss"].backward()
        self.assertIsNone(pipeline.coarse_model.scale.grad)
        self.assertTrue(
            any(parameter.grad is not None for parameter in pipeline.diffusion.parameters())
        )

    def test_pipeline_shares_one_input_object_across_both_stages(self):
        captured = {}

        class DummyCoarse(torch.nn.Module):
            def forward(self, faulty, _radar, repair, _healthy, _halo, **kwargs):
                captured["coarse"] = kwargs["shared_inputs"]
                return {
                    "coarse_lidar_bev": faulty * (1 - repair) + 0.5 * repair
                }

        clean, _coarse, faulty, radar, repair, halo = _inputs(batch=1)
        diffusion = FineDiffusionRefiner(_config())
        pipeline = FrozenCoarseFineDiffusionPipeline(DummyCoarse(), diffusion)
        with mock.patch.object(diffusion, "forward", return_value={}) as fine:
            pipeline(
                clean,
                faulty,
                radar,
                repair,
                torch.zeros_like(repair),
                halo,
            )

        shared = fine.call_args.kwargs["shared_inputs"]
        self.assertIsInstance(shared, ReconstructionInputs)
        self.assertIs(shared, captured["coarse"])
        self.assertIs(shared.faulty_lidar_bev, faulty)
        self.assertIs(shared.radar_bev, radar)
        self.assertTrue(
            torch.equal(shared.trusted_faulty, faulty * (1 - repair))
        )
        self.assertTrue(
            torch.equal(shared.effective_halo, halo * (1 - repair))
        )

    def test_direct_ablation_erases_repair_region_without_coarse_model(self):
        _clean, _coarse, faulty, radar, repair, halo = _inputs(batch=1)
        repair[:, :, 3:9, 4:11] = 1
        pipeline = FrozenCoarseFineDiffusionPipeline(
            None,
            FineDiffusionRefiner(_config(bypass_coarse=True)),
        ).eval()

        base, diagnostics = pipeline.coarse_forward(
            faulty,
            radar,
            repair,
            torch.zeros_like(repair),
            halo,
        )

        self.assertTrue(diagnostics["bypassed_coarse_reconstruction"])
        self.assertTrue(torch.equal(base * (1 - repair), faulty * (1 - repair)))
        self.assertEqual(int((base * repair).count_nonzero()), 0)

    def test_pointpillars_checkpoint_restores_grid_geometry(self):
        checkpoint = {
            "model_config": {
                "lidar_channels": 64,
                "radar_channels": 64,
                "target_lidar_channels": 3,
                "pointpillars": {
                    "enabled": True,
                    "output_channels": 64,
                    "max_points_per_pillar": 100,
                    "max_pillars": None,
                },
            },
            "grid_geometry": {
                "x_min": 0.0,
                "x_max": 64.0,
                "y_min": -32.0,
                "y_max": 32.0,
                "height": 320,
                "width": 320,
            },
            "model_state_dict": {},
        }

        class DummyCoarse(torch.nn.Module):
            def load_state_dict(self, *_args, **_kwargs):
                return None

            def to(self, *_args, **_kwargs):
                return self

        with mock.patch("torch.load", return_value=checkpoint):
            with mock.patch(
                "models.two_stage_reconstruction_head.diffusion_process.diffusion_pipeline.CoarseReconstructionModel",
                return_value=DummyCoarse(),
            ) as constructor:
                _model, loaded = load_frozen_coarse_model(
                    "checkpoint.pt", allow_pointpillars=True
                )

        self.assertIs(loaded, checkpoint)
        geometry = constructor.call_args.kwargs["grid_geometry"]
        self.assertIsInstance(geometry, BEVGridGeometry)
        self.assertEqual((geometry.height, geometry.width), (320, 320))
        self.assertEqual((geometry.x_min, geometry.x_max), (0.0, 64.0))


if __name__ == "__main__":
    unittest.main()
