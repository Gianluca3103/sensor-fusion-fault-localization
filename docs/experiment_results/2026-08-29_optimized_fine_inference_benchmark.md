# Optimized Fine inference benchmark

Date recorded: 2026-08-29

This experiment measures complete per-sample inference from the faulty input through
the frozen coarse model and Fine reconstruction. Model construction and checkpoint
loading are excluded. Each result uses batch size 1, 25 warm-up batches, 100 measured
validation samples, AMP, a 32-cell inference bucket, and `torch.compile` in
`reduce-overhead` mode. Fine-stage PointPillars conditioning was required for all
three checkpoints.

## Inputs

- Data: `/workspace/reconstruction_vod_radar3_unique_multibox_selector`
- Radar: `/workspace/radar20_pointpillars_cache`
- Frozen coarse checkpoint: `/workspace/coarse_vod_pointpillars_hrnet_b32_radar20_context80_b24_dropout020_bucket32_existing_loss_150epochs/best_model.pt`
- U-Net: `/workspace/fine_diffusion_unet_fair_pointpillars_global_b32_radar20_b12_50epochs/best_sampled_validation_iou.pt`
- Shallow Transformer: `/workspace/fine_diffusion_transformer_best_b32_radar20_b12_crop_bucket32_50epochs/best_sampled_validation_iou.pt`
- Deep Transformer: `/workspace/fine_diffusion_transformer_h128_a192_heads6_blocks6_b32_radar20_b8_50epochs/best_sampled_validation_iou.pt`

The captured terminal output did not print the GPU model, so the device name is not
asserted in this record. The individual `inference_timing.json` files retain the
device name reported by PyTorch.

## End-to-end results

| Backbone | Steps | Mean | P50 | P95 | P99 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 1 | 27.98 ms | 31.27 ms | 46.65 ms | 48.01 ms | 35.74 samples/s |
| Shallow Transformer | 1 | 51.82 ms | 37.01 ms | 179.90 ms | 181.52 ms | 19.30 samples/s |
| Deep Transformer | 1 | 56.49 ms | 39.94 ms | 185.19 ms | 198.34 ms | 17.70 samples/s |
| U-Net | 3 | 32.23 ms | 32.98 ms | 54.78 ms | 61.41 ms | 31.03 samples/s |
| Shallow Transformer | 3 | 57.47 ms | 44.03 ms | 186.24 ms | 196.87 ms | 17.40 samples/s |
| Deep Transformer | 3 | 70.20 ms | 56.79 ms | 199.07 ms | 214.32 ms | 14.24 samples/s |

## Stage breakdown

| Backbone | Steps | Loader | Host-to-device | Coarse | Fine |
|---|---:|---:|---:|---:|---:|
| U-Net | 1 | 0.15 ms | 0.18 ms | 21.78 ms | 5.82 ms |
| Shallow Transformer | 1 | 0.22 ms | 0.17 ms | 23.09 ms | 28.27 ms |
| Deep Transformer | 1 | 0.21 ms | 0.16 ms | 22.29 ms | 33.75 ms |
| U-Net | 3 | 0.16 ms | 0.16 ms | 21.41 ms | 10.45 ms |
| Shallow Transformer | 3 | 0.18 ms | 0.16 ms | 22.01 ms | 35.03 ms |
| Deep Transformer | 3 | 0.19 ms | 0.16 ms | 21.51 ms | 48.25 ms |

## Compiler warning and interpretation

All six runs hit TorchDynamo's recompilation limit of eight graphs. The reported
reason was variation in crop/window dimensions, for example:

- U-Net: `halo_mask` height expected 96 but received 256.
- Transformer: `self_layout.valid_windows` expected 189 windows but received 1280.

Consequently, the very large Transformer P95/P99 tail includes graph fallback or
shape-specialization overhead and should not be interpreted as steady-state latency
for one fixed shape. P50 is the more representative typical-frame number for this
run, while mean and tail latency remain the correct end-to-end results for the
variable crop distribution actually evaluated.

The 32-cell multiple reduces the number of shapes but does not create a small finite
set of truly static shapes. A subsequent optimization should assign crops to an
explicit finite bucket table and precompile every bucket before measurement.

## Saved pod artifacts

The raw timing JSON files were saved below:

`/workspace/fine_inference_optimized_comparison`

with one directory per backbone and refinement-step count.
