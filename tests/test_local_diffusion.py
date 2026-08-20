import unittest
from unittest import mock

import torch

from models.two_stage_reconstruction_head.diffusion_process.local_diffusion import (
    FineDiffusionConfig,
    FineDiffusionRefiner,
    MaskedNoDegradationLoss,
    ReconstructionCropExtractor,
    WindowAttention2d,
)
from models.two_stage_reconstruction_head.diffusion_process.diffusion_pipeline import (
    FrozenCoarseFineDiffusionPipeline,
    load_frozen_coarse_model,
)
from models.two_stage_reconstruction_head.pointpillars import BEVGridGeometry


def _config(*, global_context=True, sampling_steps=3, bypass_coarse=False):
    return FineDiffusionConfig(
        bypass_coarse_reconstruction=bypass_coarse,
        hidden_dim=16,
        num_heads=4,
        num_transformer_blocks=2,
        window_size=4,
        crop_context_margin_cells=2,
        use_global_faulty_context=global_context,
        global_context_dim=16,
        training_timesteps=12,
        sampling_start_timestep=11,
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
    def test_bounds_margin_halo_alignment_and_padding(self):
        clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        extractor = ReconstructionCropExtractor(2, 4)
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
            crops = ReconstructionCropExtractor(2, 4).extract(
                {"value": tensor}, tensor, torch.zeros_like(tensor)
            )
            top, bottom, left, right = crops.boxes[0].tolist()
            self.assertGreaterEqual(top, 0)
            self.assertGreaterEqual(left, 0)
            self.assertLessEqual(bottom, 16)
            self.assertLessEqual(right, 16)
        empty = torch.zeros(1, 1, 16, 16)
        crops = ReconstructionCropExtractor(2, 4).extract(
            {"value": empty}, empty, empty
        )
        self.assertFalse(bool(crops.active_samples[0]))


class FineDiffusionRefinerTests(unittest.TestCase):
    def test_sampling_timesteps_start_from_configured_low_strength_timestep(self):
        model = FineDiffusionRefiner(
            FineDiffusionConfig(
                hidden_dim=16,
                num_heads=4,
                num_transformer_blocks=1,
                window_size=4,
                training_timesteps=1000,
                sampling_start_timestep=500,
                sampling_steps=10,
                global_context_dim=16,
            )
        )

        timesteps = model._sampling_timesteps(10, torch.device("cpu"))

        self.assertEqual(int(timesteps[0]), 500)
        self.assertEqual(int(timesteps[-1]), 0)
        self.assertEqual(len(timesteps), 10)
        self.assertNotIn(999, timesteps.tolist())

    def test_initial_sampling_residual_uses_zero_residual_prior(self):
        _clean, coarse, faulty, radar, repair, halo = _inputs(batch=1)
        model = FineDiffusionRefiner(_config(sampling_steps=3)).eval()
        generator = torch.Generator().manual_seed(123)

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

        crops = output["crop_batch"]
        expected_generator = torch.Generator().manual_seed(123)
        noise = torch.randn(
            output["initial_residual"].shape,
            dtype=output["initial_residual"].dtype,
            generator=expected_generator,
        )
        start = torch.full((coarse.shape[0],), 11, dtype=torch.long)
        sigma = model.schedule.extract(
            model.schedule.sqrt_one_minus_alpha_bars,
            start,
            noise,
        )
        expected = crops.tensors["repair"] * crops.valid_mask * sigma * noise

        self.assertTrue(torch.allclose(output["initial_residual"], expected))
        self.assertEqual(
            int(
                (
                    output["initial_residual"]
                    * (1 - crops.tensors["repair"] * crops.valid_mask)
                ).count_nonzero()
            ),
            0,
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
            timestep=torch.tensor([2, 7]),
            return_debug=True,
        )
        debug = output["debug"]
        crop_batch = debug["crops"]
        self.assertTrue(
            torch.equal(
                debug["residual_gt"],
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
            int((debug["noisy_residual"] * (1 - crop_batch.tensors["repair"])).count_nonzero()),
            0,
        )
        outside = 1 - repair
        predicted_residual = crop_batch.paste(
            debug["predicted_residual_crop"]
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

    def test_inference_needs_no_clean_lidar_and_ddim_is_masked(self):
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
            timestep=torch.tensor([0]),
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
            timestep=torch.tensor([5]),
        )
        output["loss"].backward()
        names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
        for expected in (
            "transformer.residual_stem",
            "transformer.local_condition_encoder",
            "self_attention",
            "cross_attention",
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
                timestep=torch.tensor([11]),
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
            timestep=torch.tensor([4]),
        )
        output["loss"].backward()
        self.assertIsNone(pipeline.coarse_model.scale.grad)
        self.assertTrue(
            any(parameter.grad is not None for parameter in pipeline.diffusion.parameters())
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
