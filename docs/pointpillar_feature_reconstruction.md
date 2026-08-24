# Full-grid PointPillars feature reconstruction

This ablation reconstructs the dense LiDAR feature tensor immediately after
PillarScatter. It does not replace or modify the existing physical three-channel
BEV reconstruction model.

## Exact interface

One frozen LiDAR PointPillars encoder produces both input and target features,
and one frozen radar PointPillars encoder produces the radar input:

```text
faulty LiDAR points -> frozen LiDAR PointPillars -> F_faulty [B, C_l, H, W]
clean LiDAR points  -> same frozen LiDAR encoder -> F_clean  [B, C_l, H, W]
stacked radar       -> frozen radar PointPillars -> F_radar  [B, C_r, H, W]

concat(F_faulty, F_radar) -> HRNet -> delta_F
F_reconstructed = F_faulty + delta_F
target = F_clean
```

For the current VoD configuration, both encoders use the aligned 320x320 grid
covering x=[0,64) m and y=[-32,32) m at 0.20 m per cell. The cache manifest
records the actual channel counts and geometry and training validates them.

The model has no fault-selector input, reconstruction region, halo, healthy
context, observability input, or write gate. HRNet may correct any grid cell.
The clean tensor is never an input and is used only to calculate training and
validation losses.

The objective combines full-grid Smooth-L1, extra Smooth-L1 weight on cells
where the target differs from the faulty tensor, and cosine feature alignment
on non-empty clean cells. The changed-cell weighting is derived from the target
only while calculating the loss; it is not visible to HRNet at inference.

## Downstream evaluation

The native lightweight BEV detector can consume the cached post-scatter tensor
directly. Train it once on `F_clean`, freeze it, then evaluate that identical
checkpoint on `F_clean`, `F_faulty`, and `F_reconstructed`. PV-RCNN++ is not
used for this comparison because its standard interface is a raw 3D point
cloud, not the thesis PointPillars post-scatter tensor.
