# Post-PillarScatter Stage-I reconstruction

This experiment is separate from the physical three-channel LiDAR-BEV
reconstructor. It reconstructs the dense output of the thesis PointPillars
`PillarFeatureNet -> PillarScatter` interface.

## Audited interface

- Tensor returned by `PointPillarsEncoder.forward`: `dense_features`.
- Aliases in the existing coarse model: `lidar_sensor_bev` and
  `lidar_pillar_bev`.
- Shape for the selected regular PointPillars configuration:
  `[B, 64, 320, 320]`.
- Metric range: `x=[0,64) m`, `y=[-32,32) m`.
- Resolution: `0.20 m x 0.20 m` per cell.
- It is the direct output of `PillarScatter`; there is no pseudo-point
  conversion or re-pillarization in this experiment.

The current PV-RCNN++ checkpoint cannot consume this tensor: its official
architecture uses a 3D voxel backbone and height compression. The repository's
native CenterNet-style BEV detector can instead be trained with this exact
64-channel tensor as its native input. Once trained on clean cached features,
the same frozen detector is used for clean, faulty, oracle, and reconstructed
conditions.

## Mask alignment

The current feature grid and reconstruction grid have identical metric bounds,
shape, and resolution, so the projected feature masks are exactly equal to the
320x320 masks. The projection utility nevertheless maps destination cell
centres through metric coordinates and supports different grid shapes/extents;
it does not use image interpolation. Cache generation saves alignment figures
under `mask_alignment/`.

## Reconstruction

The feature reconstructor predicts `predicted_delta`. Its final output is:

```text
coarse_features = faulty_features + feature_repair_mask * predicted_delta
```

Consequently, every value outside the repair mask is copied exactly from the
faulty feature tensor. Radar post-scatter features are available over repair
plus halo context. Halo never grants write access.

The loss is:

```text
SmoothL1(coarse_features, clean_features) inside repair
+ lambda_cosine * cosine_feature_loss inside repair
```

Smooth-L1 is also reported separately for actually changed feature cells and
sacrificed healthy cells.

## Required experiment order

1. Cache post-scatter features from one frozen encoder.
2. Train the native BEV detector on clean cached features only.
3. Freeze it and evaluate clean/faulty/oracle features.
4. Continue to reconstruction training only if oracle repair improves detection.
5. Evaluate a trained reconstructor by passing `coarse_features` directly to
   that same frozen detector.

