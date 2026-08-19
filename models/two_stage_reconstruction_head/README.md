# Reconstruction head

## Deterministic Stage I

The coarse stage uses VoD PointPillars followed by HRNet. LiDAR points use
`[x, y, z, reflectivity]`; Radar points use
`[x, y, z, power, Doppler]`. Each encoder produces a 64-channel 320x320
pseudo-image.

The HRNet input is:

1. masked healthy LiDAR context (64 channels),
2. Radar features inside the reconstruction/halo region (64 channels),
3. reconstruction mask (1 channel),
4. healthy-context mask (1 channel).

With the canonical configuration this is a 130-channel tensor. HRNet preserves
a full-resolution branch while exchanging information with 160x160, 80x80,
and 40x40 branches. Its output head predicts occupancy logits, density, and
height. Only cells inside `reconstruction_mask` replace the faulty LiDAR BEV.

The canonical configuration is `configs/coarse_reconstruction_vod.json`.

## Masks

Fault-selector inputs are generated upstream and cached. The loader reads only
the cached reconstruction, halo, and healthy-context masks during training.
Samples without a selected reconstruction region are excluded from aggregate
training/evaluation reconstruction metrics.

## Stage II

Stage II is a local cropped dense residual-diffusion Transformer. It models
only `reconstruction_mask * (clean - coarse)` and never changes cells outside
the reconstruction mask.

For each sample, the tight repair bounds are expanded by the configured margin
and by the explicit halo extent. Crops retain the original 0.20 m cell size and
are zero-padded to the largest crop in a batch and to a multiple of the
attention window. A valid-crop mask prevents padding from participating in
attention or losses. No geometric resizing is used.

The local condition contains the coarse crop, radar crop, faulty LiDAR with
repair cells erased, reconstruction and halo masks, and continuous local/global
coordinates. An optional small strided encoder summarizes the full faulty
LiDAR after the repair region is erased. Its embedding and the diffusion
timestep modulate alternating normal/shifted window Transformer blocks. The
blocks use local self-attention, local cross-attention to the condition map,
and a depthwise-convolutional FFN.

Training combines masked epsilon MSE with exact cell-aligned reconstruction
loss. Inference uses configurable 1/3/5/10-step DDIM and composes the result as
trusted faulty LiDAR outside the mask and `coarse + residual` inside it.

The canonical configuration is `configs/fine_diffusion.json`; training uses
`python -m models.two_stage_reconstruction_head.diffusion_process.train_fine_diffusion`.
The previous full-BEV residual U-Net remains loadable only for historical
checkpoint compatibility.
