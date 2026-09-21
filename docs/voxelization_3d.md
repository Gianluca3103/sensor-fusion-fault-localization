# Standalone LiDAR and radar 3D voxelization

This package is a deterministic preprocessing and diagnostic layer. It does
not change PointPillars, the coarse/fine reconstruction models, losses,
training, fusion, or detection.

## Repository contract found during inspection

| Item | Current repository convention |
|---|---|
| LiDAR raw fields | `[x, y, z, reflectivity]` (`float32`) |
| Native VoD radar fields | `[x, y, z, rcs, radial_velocity, compensated_radial_velocity, time_index]` |
| Current aligned model radar cache | `[x, y, z, rcs, compensated_radial_velocity]` (`float32`) |
| Physical axes | `x` forward, `y` lateral, `z` vertical, metres |
| Model BEV support | `x=[0,64)`, `y=[-32,32)`, 0.2 m cells |
| Existing height support | `z=[-3,5)` |
| Existing PointPillars | Hard XY pillars, learned VFE, dense 320x320 scatter; height is not discretized |
| Sample indexing | `{data_root}/{train,val,test}/*.npz`; metadata stores physical `frame_id` and clean LiDAR source path |
| Radar indexing | `{radar_root}/{split}/{frame_id:05d}.npz`; fault variants of one frame share radar |

VoD radar is transformed with the provided KITTI calibration. Accumulated VoD
radar is ego-motion compensated before that transform. HeRCULES V2 uses sensor
extrinsics, interpolated poses, ego-motion compensation, and tracked dynamic
motion compensation. Therefore the maintained radar cache and LiDAR points are
already expressed in the same LiDAR frame.

The current detector-facing radar artifact intentionally contains five reliable
fields. The native VoD-only `radial_velocity` and `time_index` columns are not
present in that cache and are not fabricated by this pipeline. The hard
voxelizer itself preserves every input column, so a future seven-field raw VoD
adapter requires no voxelizer change.

## Canonical representation

- Physical input points: `[x, y, z, attributes...]`.
- Sparse integer coordinates: `[z_index, y_index, x_index]`.
- Half-open bounds: minimum is included and maximum is excluded.
- Default voxel size: `[0.2, 0.2, 0.25]` m in XYZ.
- Default dimensions: `[Nx, Ny, Nz] = [320, 320, 32]`.
- Sparse coordinates are lexicographically sorted in ZYX order.
- Input order is preserved inside each voxel, including under truncation.

For each retained point, `voxel_points` contains all raw features followed by:

1. `x/y/z - full voxel point centroid`; and
2. `x/y/z - geometric voxel center`.

The centroid uses every valid point assigned to the voxel before an optional
capacity limit. The serialized representation is:

- `voxel_coords`: `[N,3]`, `int32`, ZYX;
- `voxel_points`: `[N,K,C_raw+6]`, `float32`;
- `num_points`: retained points per voxel;
- `original_num_points`: points before truncation.

No maximum number of voxels is imposed. `max_points_per_voxel` is `null` in the
base configuration so exploratory statistics are lossless. With `null`, `K` is
the largest occupied-voxel population in that frame. Choose finite modality-
specific capacities before building a large persistent cache.

### Initial VoD capacity audit

A deterministic sample of 100 unique validation frames (seed 42, clean LiDAR,
20-frame aligned radar cache) produced the following initial evidence:

| Statistic | LiDAR | Radar |
|---|---:|---:|
| Mean points / occupied voxel | 9.385 | 1.431 |
| p50 / p90 / p95 | 4 / 20 / 30 | 1 / 2 / 3 |
| p99 / p99.5 / p99.9 / max | 70 / 106 / 258 / 888 | 7 / 8 / 14 / 42 |
| Mean occupied voxels / frame | 9,345.7 | 3,151.4 |
| p95 occupied voxels / frame | 12,181.8 | 4,816.7 |

An initial finite-capacity recommendation is therefore **258 for LiDAR** and
**14 for radar** (the measured p99.9 values), but it is intentionally not
silently enabled. On this sample those caps discard 1.089% and 0.244% of
in-range points, respectively. A LiDAR cap of 64 would truncate only 1.13% of
occupied voxels but 8.18% of all in-range points, demonstrating why voxel
frequency alone is misleading. Run the tool over the intended training corpus
before fixing production capacities.

Points excluded by the common spatial support are reported separately and are
not called truncation. This audit found no non-finite points; it excluded points
outside the requested `[0,64) x [-32,32) x [-3,5)` cuboid as expected.

## Tools

Run all commands from the repository root.

```bash
python -m scripts.analyze_3d_voxels \
  --data-root /path/to/reconstruction_samples \
  --radar-root /path/to/aligned_radar_cache \
  --config configs/voxelization_3d.json \
  --split train --limit-samples 1000 --seed 42 \
  --lidar-source clean \
  --output /path/to/voxel_statistics.json
```

The statistics tool deduplicates physical frames by default, builds exact
streaming histograms, reports occupied voxels per frame and global points per
occupied voxel, and quantifies point/voxel loss for several capacity choices.

```bash
python -m scripts.visualize_3d_voxels \
  --data-root /path/to/reconstruction_samples \
  --radar-root /path/to/aligned_radar_cache \
  --config configs/voxelization_3d.json \
  --split val --sample-index 0 --lidar-source clean \
  --output-root /path/to/voxel_visualizations
```

This saves raw LiDAR/radar 3D views, occupied voxel centers, a joint overlay,
voxel cubes, and XY/XZ/YZ projections. Plot sampling affects display only.

For an interactive, mouse-rotatable viewer:

```bash
python -m scripts.interactive_3d_voxels \
  --data-root /path/to/reconstruction_samples \
  --radar-root /path/to/aligned_radar_cache \
  --config configs/voxelization_3d.json \
  --split val --sample-index 0 --lidar-source clean \
  --output-root /path/to/saved_camera_views
```

Drag to rotate and use the mouse wheel to zoom. Keys `1`, `2`, and `3` select
LiDAR, radar, or both; `R`, `V`, and `X` select raw points, voxel centers, or
their overlay; `C` toggles a bounded number of voxel wireframes; arrow keys
rotate by fixed increments; `0` resets the camera; and `S` saves the current
camera angle.

Raw HeRCULES scenes can be opened directly, without first creating fault
samples or radar caches:

```bash
python -m scripts.interactive_hercules_3d \
  --hercules-root /path/to/HeRCULES \
  --split-manifest configs/hercules_scene_split.json \
  --split val --frame-index 0 \
  --radar-frames 0 \
  --max-history-s 1.0 \
  --max-translation-m 4.0 \
  --max-rotation-deg 5.0 \
  --output-root /path/to/saved_hercules_views
```

`--radar-frames 0` means no numerical cap. Every causal radar scan that still
passes the history and pose-quality gates is used. The viewer prints the exact
accepted scan count, oldest scan age, temporal-filter counts, confirmed dynamic
tracks, and motion-compensated point count. Widening the gates makes the cloud
denser but increases alignment uncertainty; it is therefore explicit rather
than silently accumulating an entire scene.

```bash
python -m scripts.cache_3d_voxels \
  --data-root /path/to/reconstruction_samples \
  --radar-root /path/to/aligned_radar_cache \
  --cache-root /path/to/voxel_cache \
  --config configs/voxelization_3d.json \
  --split train --lidar-source clean --modalities lidar radar
```

LiDAR and radar are cached separately. Every artifact contains source identity,
coordinate conventions, raw/decorated field names, grid configuration, cache
version, SHA-256 configuration fingerprint, and exclusion/truncation counts.
Stale configuration or version metadata is rejected on load.

## Validation

`tests/test_voxelization_3d.py` covers boundary mapping, occupied-voxel counts,
centroid offsets, center round trips, modality separation, empty clouds,
deterministic truncation, cache reproducibility/invalidation, visualization
coordinate sanity, and invalid-point accounting.
