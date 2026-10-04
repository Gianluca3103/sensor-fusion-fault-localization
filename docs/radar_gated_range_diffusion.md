# Radar-gated range-view diffusion

This is a new trainable reconstruction path. It conditions a range-view U-Net
on the ray–depth cross-attention blueprint and refines **depth residuals** on
missing LiDAR rays with measured local radar support. It does not modify the
old BEV or sparse-voxel diffusion baselines. Only one-sample local smoke
checkpoints exist; no full VoD run or detector evaluation has been completed.

The command below trains the blueprint and diffusion jointly from scratch.
For the recommended blueprint-first schedule, use
[`staged_ray_depth_training.md`](staged_ray_depth_training.md). Its stage-1
checkpoint can initialize this trainer with `--pretrained-blueprint`; an
optional `--freeze-blueprint-epochs` period then trains diffusion before joint
fine-tuning.

## What is enforced

- The outer support set is built from radar tokens actually available to the
  cross-attention branch. Observed LiDAR rays are excluded from additions.
  Clean LiDAR never enters candidate selection or this mask.
- During training, all radar-supported missing rays enter diffusion. A separate
  reliability head learns whether the selected blueprint depth is within the
  configured correctable distance of the clean first return. Clean no-return
  rays and badly displaced proposals are negatives. This prevents the
  reliability threshold from starving training before it has learned.
- During sampling, the reliability threshold narrows the eligible set. Initial
  noise and every reverse diffusion update are multiplied by that set; no
  generated point can appear outside it. A return/no-return head decides which
  eligible rays actually emit a point.
- Each accepted ray has one calibrated XYZ position. The depth correction is
  bounded, and the range is clamped to the calibrated sensor limits. A separate
  head predicts the added point's intensity. Export by appending added points
  to the **original** faulty cloud, which retains its exact XYZ and intensity.

The U-Net reads the full rectangular tile for context while downsampling only
azimuth, retaining every beam row. Tiles have width at least four. The module
currently treats surviving LiDAR returns as trusted; it is an **addition-only**
experiment and does not remove fog ghosts or replace an incorrect observed
first return. Its measured radar support is bounded by the cross-attention
neighborhood; a single ghost can still be proposed. Validation calibration is
needed to reject unreliable proposals. The reliability head is trained on
depth correctness, so its empirical precision is not detector AP.

## Train on the professor machine

After syncing this repository, the verified VoD 20-scan radar files and the
full-scan fault artifacts must exist. In particular, `samples/train` and
`samples/val` must contain compatible NPZ files with `range_view_full_scan`
metadata. The geometry JSON must contain calibrated beam elevations and
azimuth sampling for the same LiDAR. Run from the repository root:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/svefusion-clean-cu118/bin/python"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
CACHE="$BASE/sensor_fusion_outputs/vod_range5_full_cache"
RUN="$BASE/sensor_fusion_outputs/vod_radar_gated_ray_diffusion"

cd "$REPO"
"$PY" -u -m scripts.train_radar_gated_ray_diffusion \
  --samples-root "$CACHE/samples" \
  --vod-root "$VOD" \
  --geometry "$CACHE/angular_geometry.json" \
  --output-root "$RUN" \
  --epochs 80 --batch-size 1 --tile-rows 4 --tile-cols 64 \
  --validate-every 5 --num-workers 4 --device cuda
```

The command saves `last_checkpoint.pt` after each epoch and prints one compact
JSON summary per epoch. To continue to 120 total epochs, set `--epochs 120`
and `--resume "$RUN/last_checkpoint.pt"` with the same geometry and model
dimensions. Use `--train-limit` and `--val-limit` for a small pipeline smoke
run. The default tile has 1280 ray–depth candidates, below the blueprint's
2048-candidate limit. The trainer samples spatial radar regions rather than
individual returns to reduce the dominance of dense road patches.

At validation, the trainer searches for the widest reliability threshold with
at least 90% empirical precision on depth-correctable supported rays and at
least 100 accepted validation proposals. These settings are configurable with
`--target-support-precision` and `--min-calibration-predictions`. If the
criterion cannot be met, the saved threshold is 1.0 and inference abstains.
The threshold is stored in `checkpoint["calibration"]["threshold"]`. It is
estimated from sampled validation tiles; verify its precision on an untouched
split and on the **final generated points** before interpreting it as a safety
or detection claim.

## Export matched reconstructed validation LiDAR

The exporter loads the trained checkpoint and its validation-calibrated gate,
checks that the fault cache contains every official VoD validation ID exactly
once, reuses the sensor encodings across nonoverlapping ray tiles, and writes
one reconstructed `.bin` per frame. It retains the original faulty points and
their intensity exactly before the same forward-FOV filtering used by the
SVEFusion detector preparation. Use a new output root for this architecture:

```bash
EXPORT="$BASE/sensor_fusion_outputs/vod_radar_gated_ray_diffusion_export"
"$PY" -u -m scripts.export_radar_gated_ray_diffusion \
  --samples-root "$CACHE/samples" \
  --vod-root "$VOD" \
  --checkpoint "$RUN/last_checkpoint.pt" \
  --output-root "$EXPORT" \
  --steps 20 --device cuda
```

For a pipeline check, add `--limit 1` and use a separate output root. The
manifest marks that export as incomplete. The full export writes
`$EXPORT/lidar/reconstructed/training/velodyne/<frame>.bin` and
`$EXPORT/lidar/reconstructed/export_manifest.json`. It can resume an interrupted
run only when its recorded checkpoint, settings and frame IDs match exactly.
The local test runs a tiny synthetic validation scan; runtime on the professor
GPU and on 1296 full VoD frames has not been measured. The current attention
search may be slow, so measure a one-frame export before scheduling the full
validation set.

## Inference interface for one rectangular tile

```python
from models.two_stage_reconstruction_head.ray_depth_queries import ray_tile_indices

# Instantiate blueprint_model and diffusion with the checkpoint's settings,
# load their state_dicts, and put both in eval mode on the same device.
rows, cols = ray_tile_indices(geometry, row_start=0, row_stop=4,
                              col_start=0, col_stop=64,
                              batch_size=1, device=device)
with torch.no_grad():
    blueprint = blueprint_model(radar, radar_valid, faulty, faulty_valid,
                                rows, cols)
    result = diffusion.sample(
        blueprint, faulty, faulty_valid, (4, 64), steps=20,
        reliability_threshold=checkpoint["calibration"]["threshold"],
    )
    additions = result.added_points(0)            # [K,4], XYZ + intensity
    original_plus_additions = result.merge_with_observed(faulty, faulty_valid, 0)
```

`sample_full_scan` handles tile stitching and calls the sensor encoders once;
the example above shows the underlying per-tile API. The training command runs
on full-scan VoD samples but optimizes one radar-centred tile per sample and
epoch. Its coverage of all regions should be measured before comparing with
the existing detector baselines.

## Evaluation before detector use

Report radar-supported ray count, correctable clean-ray count, accepted-point
precision and recall, depth error, intensity error, and no-return false
positives. Hold faulty LiDAR fixed and substitute radar from another frame:
the supported additions should move or disappear. Then run exactly matched
clean, faulty, and reconstructed LiDAR + radar detection on the same validation
IDs. The implementation alone establishes none of those outcomes.
