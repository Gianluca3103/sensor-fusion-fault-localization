# PFS-Radar v2: Adaptive Stacking and Doppler Tracking

`PFS_Radar_v2` is a new radar preprocessing path. It leaves the original
`PFS_Radar` implementation unchanged.

## What changes

The cache builder uses the newest causal Continental frame as its pose
reference and walks backward until any accuracy gate is crossed:

- maximum history
- maximum relative translation
- maximum relative rotation
- optional maximum number of frames (disabled by default)

Every accepted frame is ego-motion compensated into the current Aeva LiDAR
frame. Its contribution is softly weighted by age, translation, and rotation.
Density is divided by total effective frame support so slow and fast sequences
have comparable scale.

Raw radial Doppler is compensated with pose-derived radar velocity. The
`auto` sign mode chooses the Continental sign convention that best explains the
static majority in each frame. Large residuals are clustered with DBSCAN,
associated causally across the selected window, and assigned a planar velocity
from both centroid motion and the available radial constraints. Confirmed
tracks are advanced to the current LiDAR timestamp before BEV projection,
reducing trails from moving vehicles.

## Four output channels

The tensor shape remains `[4, H, W]`:

1. static occupancy
2. support-normalized static density
3. tracked dynamic speed
4. dynamic-track occupancy/confidence

These semantics differ from PFS-Radar v1. Use a separate cache root and retrain
the PFS-Radar model from scratch.

## Build the cache

```bash
PYTHON="/path/to/python"
DATASET_ROOT="/path/to/reliability_dataset"
HERCULES_ROOT="/path/to/HeRCULES"
RADAR_V2_ROOT="/path/to/radar_cache_adaptive_doppler_v2"

"$PYTHON" PFS_Radar_v2/prepare_radar_cache.py \
  --dataset-root "$DATASET_ROOT" \
  --hercules-root "$HERCULES_ROOT" \
  --output-root "$RADAR_V2_ROOT" \
  --num-workers 4 \
  --max-delta-ms 30 \
  --max-frames 0 \
  --max-history-s 1.0 \
  --max-translation-m 4.0 \
  --max-rotation-deg 5.0 \
  --doppler-sign auto \
  --dynamic-threshold-mps 1.0 \
  --cluster-eps-m 1.2 \
  --cluster-min-samples 2 \
  --association-distance-m 3.0 \
  --min-track-hits 2
```

`--max-frames 0` removes the frame-count cap. The accepted causal history is
still bounded by `--max-history-s`, `--max-translation-m`, and
`--max-rotation-deg`, and older accepted frames receive smaller soft weights.
Use a positive value only when a strict compute or memory ceiling is required.

For training, use the existing `PFS_Radar/train_pfs_radar.py` with
`--radar-root "$RADAR_V2_ROOT"`. Do not pass the v1 fixed-stack validation
arguments `--radar-frame-count` or `--require-full-radar-stack`; the v2 cache
records and validates its complete adaptive policy itself.

## Sign check

Start with `--doppler-sign auto`. Inspect `alignment_rows[].doppler_sign` in a
few cache metadata records. If the inferred convention is stable for the
dataset, repeat the controlled experiment with explicit `--doppler-sign 1` or
`--doppler-sign -1`. At very low ego speed auto mode deliberately falls back to
`1` because the static scene does not provide enough evidence to infer a sign.

## Recommended ablation

Compare:

1. one radar frame
2. fixed 20-frame PFS-Radar v1
3. adaptive PFS-Radar v2

Report localization IoU, precision, recall, turns, ego-speed bins, dynamic
artifacts, runtime, selected-frame count, and confirmed-track count.
