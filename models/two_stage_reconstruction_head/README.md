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

`diffusion_process/` contains the masked residual-diffusion model. It was not
redesigned as part of the HRNet/VoD cleanup.
