import unittest

import torch

from models.two_stage_reconstruction_head.diffusion_process.basic_diffusion_unet import (
    BasicDiffusionUNet,
    TimestepResidualBlock,
)
from models.two_stage_reconstruction_head.diffusion_process.local_diffusion import (
    FineDiffusionConfig,
    FineDiffusionRefiner,
    fine_diffusion_architecture_metadata,
)
from models.two_stage_reconstruction_head.reconstruction_crop import (
    ReconstructionCropExtractor,
)


def _small_unet():
    return BasicDiffusionUNet(
        base_channels=4,
        channel_multipliers=(1, 2, 4, 8),
        num_downsamples=3,
        resblocks_per_level=2,
    )


class BasicDiffusionUNetTests(unittest.TestCase):
    def _inputs(self, height, width):
        residual = torch.randn(1, 3, height, width)
        coarse = torch.rand(1, 3, height, width)
        radar = torch.rand(1, 4, height, width)
        repair = torch.ones(1, 1, height, width)
        halo = torch.zeros_like(repair)
        valid = torch.ones_like(repair)
        timestep = torch.tensor([3])
        return residual, coarse, radar, repair, halo, valid, timestep

    def test_input_output_channels_and_zero_initialized_identity_update(self):
        model = _small_unet().eval()
        self.assertEqual(model.input_channels, 13)
        output, debug = model(*self._inputs(80, 80))
        self.assertEqual(tuple(output.shape), (1, 3, 80, 80))
        self.assertEqual(debug["unet_output_shape"], (1, 3, 80, 80))
        self.assertEqual(int(output.count_nonzero()), 0)

    def test_configured_dropout_is_applied_to_residual_blocks(self):
        model = BasicDiffusionUNet(
            base_channels=4,
            channel_multipliers=(1, 2, 4, 8),
            num_downsamples=3,
            resblocks_per_level=2,
            dropout=0.2,
        )
        dropouts = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.Dropout2d)
        ]
        self.assertTrue(dropouts)
        self.assertTrue(all(module.p == 0.2 for module in dropouts))

    def test_pointpillars_conditioning_uses_both_sensor_feature_maps(self):
        model = BasicDiffusionUNet(
            use_pointpillars_conditioning=True,
            lidar_pillar_channels=6,
            radar_pillar_channels=7,
            base_channels=4,
            channel_multipliers=(1, 2, 4, 8),
            num_downsamples=3,
            resblocks_per_level=1,
        ).eval()
        inputs = self._inputs(80, 80)
        lidar_pillars = torch.rand(1, 6, 80, 80)
        radar_pillars = torch.rand(1, 7, 80, 80)

        output, debug = model(
            *inputs,
            lidar_pillars,
            radar_pillars,
        )

        self.assertEqual(model.input_channels, 22)
        self.assertEqual(tuple(output.shape), (1, 3, 80, 80))
        self.assertEqual(debug["unet_sensor_condition"].shape[1], 13)
        self.assertTrue(debug["unet_pointpillars_conditioning"])

    def test_pointpillars_conditioning_requires_both_feature_maps(self):
        model = BasicDiffusionUNet(
            use_pointpillars_conditioning=True,
            base_channels=4,
            channel_multipliers=(1, 2, 4, 8),
            num_downsamples=3,
            resblocks_per_level=1,
        )
        with self.assertRaisesRegex(ValueError, "requires both"):
            model(*self._inputs(80, 80))

    def test_fair_variant_hides_coarse_and_uses_global_context(self):
        model = BasicDiffusionUNet(
            use_pointpillars_conditioning=True,
            lidar_pillar_channels=6,
            radar_pillar_channels=7,
            include_coarse_input=False,
            global_context_dim=16,
            base_channels=4,
            channel_multipliers=(1, 2, 4, 8),
            num_downsamples=3,
            resblocks_per_level=1,
        ).eval()
        inputs = list(self._inputs(80, 80))
        lidar_pillars = torch.rand(1, 6, 80, 80)
        radar_pillars = torch.rand(1, 7, 80, 80)
        global_embedding = torch.rand(1, 16)

        _output, debug = model(
            *inputs,
            lidar_pillars,
            radar_pillars,
            global_embedding,
        )
        first_input = debug["unet_contextual_input"].clone()
        inputs[1] = torch.rand_like(inputs[1])
        _output, changed_debug = model(
            *inputs,
            lidar_pillars,
            radar_pillars,
            global_embedding,
        )

        self.assertEqual(model.input_channels, 19)
        self.assertFalse(debug["unet_coarse_visible"])
        self.assertIs(debug["unet_global_context"], global_embedding)
        self.assertTrue(
            torch.equal(first_input, changed_debug["unet_contextual_input"])
        )

    def test_global_context_is_required_when_enabled(self):
        model = BasicDiffusionUNet(
            global_context_dim=16,
            base_channels=4,
            channel_multipliers=(1, 2, 4, 8),
            num_downsamples=3,
            resblocks_per_level=1,
        )
        with self.assertRaisesRegex(ValueError, "requires a global embedding"):
            model(*self._inputs(80, 80))

    def test_fair_refiner_metadata_records_hidden_coarse_and_global_context(self):
        config = FineDiffusionConfig(
            fine_backbone="unet",
            fine_unet_include_coarse_input=False,
            fine_unet_use_global_faulty_context=True,
            fine_min_context_height=80,
            fine_min_context_width=80,
        )
        model = FineDiffusionRefiner(config)
        metadata = fine_diffusion_architecture_metadata(config)

        self.assertIsNotNone(model.unet_global_encoder)
        self.assertEqual(metadata["version"], 14)
        self.assertEqual(metadata["input_channels"], 10)
        self.assertFalse(metadata["coarse_visible_to_backbone"])
        self.assertTrue(metadata["global_faulty_context"])

    def test_raw_radar_checkpoint_input_shape_remains_unchanged(self):
        model = _small_unet().eval()
        self.assertFalse(model.use_pointpillars_conditioning)
        self.assertEqual(model.input_channels, 13)

    def test_required_shapes_have_expected_stride_eight_bottleneck(self):
        model = _small_unet().eval()
        cases = {
            (80, 80): (10, 10),
            (80, 112): (10, 14),
            (136, 200): (17, 25),
            (249, 320): (32, 40),
            (320, 320): (40, 40),
        }
        with torch.no_grad():
            for shape, deepest in cases.items():
                output, debug = model(*self._inputs(*shape))
                self.assertEqual(tuple(output.shape[-2:]), shape)
                self.assertEqual(debug["unet_bottleneck_shape"][-2:], deepest)
                self.assertGreaterEqual(deepest[0], 10)
                self.assertGreaterEqual(deepest[1], 10)

    def test_every_resblock_receives_timestep_embedding(self):
        model = _small_unet().eval()
        reached = []
        handles = []
        for module in model.modules():
            if isinstance(module, TimestepResidualBlock):
                handles.append(
                    module.time_projection.register_forward_pre_hook(
                        lambda _module, _inputs: reached.append(True)
                    )
                )
        model(*self._inputs(80, 80))
        for handle in handles:
            handle.remove()
        expected = sum(
            isinstance(module, TimestepResidualBlock)
            for module in model.modules()
        )
        self.assertEqual(len(reached), expected)

    def test_no_attention_or_transformer_modules_exist(self):
        names = [module.__class__.__name__.lower() for module in _small_unet().modules()]
        self.assertFalse(any("attention" in name for name in names))
        self.assertFalse(any("transformer" in name for name in names))

    def test_invalid_and_nonrepair_output_is_zero_and_backward_is_finite(self):
        model = _small_unet().train()
        inputs = list(self._inputs(80, 80))
        repair = inputs[3]
        valid = inputs[5]
        repair[:, :, :, 40:] = 0
        valid[:, :, 72:, :] = 0
        with torch.no_grad():
            model.output_conv.weight.normal_(0.0, 0.01)
        output, _debug = model(*inputs)
        active = (repair > 0.5) & (valid > 0.5)
        self.assertEqual(int(output.masked_select(~active.expand_as(output)).count_nonzero()), 0)
        loss = (output - torch.ones_like(output)).square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


class BasicDiffusionUNetCropAndIntegrationTests(unittest.TestCase):
    def test_minimum_context_and_edge_shift_preserve_masks(self):
        repair = torch.zeros(1, 1, 320, 320)
        halo = torch.zeros_like(repair)
        repair[:, :, :4, :5] = 1
        halo[:, :, 4:12, :14] = 1
        original_repair = repair.clone()
        original_halo = halo.clone()
        values = {"repair": repair, "halo": halo, "radar": torch.ones(1, 4, 320, 320)}
        crops = ReconstructionCropExtractor(
            8, minimum_height=80, minimum_width=80
        ).extract(values, repair, halo)
        self.assertEqual(crops.boxes.tolist(), [[0, 80, 0, 80]])
        self.assertEqual(tuple(crops.valid_mask.shape[-2:]), (80, 80))
        self.assertTrue(torch.equal(repair, original_repair))
        self.assertTrue(torch.equal(halo, original_halo))

    def test_249_by_320_real_crop_is_technically_padded_and_invalid(self):
        repair = torch.zeros(1, 1, 320, 320)
        repair[:, :, :249, :] = 1
        halo = torch.zeros_like(repair)
        crops = ReconstructionCropExtractor(
            8, minimum_height=80, minimum_width=80
        ).extract({"repair": repair}, repair, halo)
        self.assertEqual(crops.crop_heights.tolist(), [249])
        self.assertEqual(tuple(crops.valid_mask.shape[-2:]), (256, 320))
        self.assertEqual(int(crops.valid_mask[:, :, 249:].count_nonzero()), 0)

    def test_fine_refiner_keeps_outside_repair_exactly_unchanged(self):
        config = FineDiffusionConfig(
            fine_backbone="unet",
            use_pointpillars_conditioning=False,
            fine_unet_base_channels=4,
            fine_unet_channel_multipliers=(1, 2, 4, 8),
            fine_unet_num_downsamples=3,
            fine_unet_resblocks_per_level=1,
            fine_min_context_height=80,
            fine_min_context_width=80,
            sampling_steps=1,
            training_timesteps=12,
            use_global_faulty_context=False,
            global_context_dim=16,
        )
        model = FineDiffusionRefiner(config).train()
        faulty = torch.rand(1, 3, 96, 96)
        coarse = faulty.clone()
        clean = faulty.clone()
        clean[:, :, 40:48, 40:48] = torch.rand(1, 3, 8, 8)
        repair = torch.zeros(1, 1, 96, 96)
        repair[:, :, 40:48, 40:48] = 1
        halo = torch.zeros_like(repair)
        halo[:, :, 36:52, 36:52] = 1
        halo *= 1 - repair
        output = model(
            clean,
            coarse,
            faulty,
            torch.rand(1, 4, 96, 96),
            repair,
            halo,
            return_debug=True,
        )
        outside = 1 - repair
        self.assertEqual(
            float(
                ((output["final_lidar_bev"] - faulty) * outside)
                .abs()
                .max()
                .detach()
            ),
            0.0,
        )
        crops = output["debug"]["crops"]
        self.assertEqual(tuple(crops.valid_mask.shape[-2:]), (80, 80))
        self.assertTrue(torch.equal(crops.paste(crops.tensors["repair"]), repair))
        self.assertTrue(torch.equal(crops.paste(crops.tensors["halo"]), halo))


if __name__ == "__main__":
    unittest.main()
