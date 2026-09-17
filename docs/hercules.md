# HeRCULES with the current coarse/diffusion models

No historical models are required. The maintained generator accepts either
`--vod-root` (unchanged) or `--hercules-root`. Use separate output/cache roots
for each dataset and preprocessing experiment.

Expected raw layout: `day/session/LiDAR/Aeva/*.bin`, one Continental directory
per session, `IMU_LiDAR.txt`, `Continental_LiDAR.txt`, `Aeva_gt.txt` and
`Continental_gt.txt`. Ambiguous or missing calibration is an error, not an
identity fallback. Aeva 29-byte records supply measured XYZ and reflectivity;
Continental 29-byte records supply XYZ, RCS and radial velocity.

Frames are sorted chronologically within each session and divided 70/15/15
into train/val/test. Numeric artifact IDs are deterministic for an unchanged
source tree; original path and nanosecond timestamp are retained in metadata.
Do not change the source tree while reusing artifacts. This is a new split
policy, not a claim to reproduce old HeRCULES experiments. Historical radar
context can precede a split boundary, as it would online; no future scans or
clean LiDAR targets are used for input alignment/filtering.

V2 takes causal history bounded by 1 second, 4 m translation and 5 degrees
rotation relative to the radar pose interpolated at the current LiDAR timestamp.
Frame count is uncapped by default; `--hercules-radar-frames 20` adds a cap.
It checks the newest scan is within `--hercules-max-radar-age-ms` (default 30 ms).
For the measured approximately 50 ms radar cadence, use 75 ms for the preview;
this accepts the observed 59.34 ms maximum age without admitting future scans.
This is a configurable freshness gate, not a timestamp offset correction. Larger
gaps still fail. The age and limit are recorded in alignment metadata, and the
limit participates in the cache-policy hash.
It
interpolates poses without extrapolation, and transforms into current Aeva
coordinates. Pose gaps above 200 ms are rejected by default, configurable via
`--hercules-max-pose-gap-ms`. Exact timestamps use recorded poses directly;
velocity uses the shorter adjacent measured interval, also subject to this
limit. No velocity is invented across unsupported gaps. GT pose translations are
treated as sensor positions; extrinsics rotate IMU axes. Doppler sign defaults
to `auto`, selecting the convention best explaining the static majority;
`--hercules-doppler-sign 1` or `-1` fixes it after verification. Low-speed auto
mode falls back to +1. Dynamic returns are DBSCAN-clustered and associated
chronologically. Tracks with at least two hits use centroid/Doppler velocity
to advance their points to the current LiDAR timestamp, with a 30 m/s safety
cap. Unconfirmed/unclustered returns are not advanced. Verify calibration and
sign against your release before full generation. Ground-truth ego poses are dataset
preprocessing inputs; deployment would need estimated ego motion.

The existing VoD filter supplies finite/range/height gates and distinct-scan
XY support (default radius 0.75 m, at least two scans). Latest-scan points are
preserved. Filtering follows ego and confirmed-object motion compensation. The radius
is a starting value, not a tuned optimum; HeRCULES RCS units are not VoD dBsm.

## University machine: generate

Generation now warms the exact Numba observability kernel before discovery and
worker startup and logs the selected backend. Without Numba, it fails with an
installation instruction instead of silently spending minutes in the Python
ray-tracing fallback. `--allow-slow-observability` explicitly permits that
reference backend for diagnostics. Install Numba in the active environment,
not a different Python environment. Initial compilation is a one-time startup
cost; the kernel uses disk caching. The reference path remains available for
array-exact comparison tests. No rays, height bins or confidence rules changed.
Calibration/pose text paths are indexed once per scene per process, preserving
missing/ambiguous-file errors and avoiding stats of every raw scan. Restart
generation after changing files in a scene because the index is cached.

Parallel generation explicitly uses `spawn`, avoiding inherited initialized
Numba/BLAS state on Linux. At most twice the worker count is submitted at once;
completed frames are reported without waiting behind an earlier slow frame.
Seeds, fault assignment and output paths remain fixed before scheduling. Progress
logs show the first completion and then every approximately five seconds when
results arrive, including elapsed time and completed samples/second. This is not
a heartbeat if all workers are blocked. Invalid synchronization still raises;
pending work is canceled but already running tasks may finish during shutdown.

For scene-held-out baseline experiments, create a manifest once using
`python -m tools.create_hercules_scene_split --hercules-root "$RAW" --output "$SPLITS"`.
This selects three scenes reproducibly (seed 42): one full validation scene,
one full test scene and a third divided chronologically at its median frame.
All other scenes are training-only. One second is excluded on either side of
the midpoint, preventing the configured one-second causal radar history from
touching validation frames during testing. Increase the buffer if increasing
radar history. Half-scenes need not have equal frame counts after exclusion.
Validation/test still share the divided scene's environment, so they are not
fully scene-independent. Pass `--hercules-split-manifest "$SPLITS"` to every
generation command and use fresh artifact roots. Frame IDs remain stable;
the manifest contents participate in the radar policy hash. Keep the manifest
with the experiment outputs and never regenerate it for model selection.
The separate dense-reference builder currently uses the original split policy;
do not combine it with this manifest until its split handling is integrated.

Run from the repository root in the existing sensor-fusion environment:

```bash
PYTHON=/home/arrubuntu20/anaconda3/envs/sensor-fusion/bin/python
RAW=/mnt/3D10B36523559581/HeRCULES
DATA=/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/hercules_v2_reconstruction
RADAR=/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/hercules_v2_radar_filtered

for SPLIT in train val test; do
  "$PYTHON" -u -m Fault_Localization_Model.create_vod_reconstruction_dataset \
    --hercules-root "$RAW" --split "$SPLIT" \
    --output-root "$DATA" --radar-cache-root "$RADAR" \
    --hercules-radar-frames 0 --hercules-temporal-radius 0.75 \
    --hercules-max-history-s 1 --hercules-max-translation-m 4 \
    --hercules-max-rotation-deg 5 --hercules-doppler-sign auto \
    --num-workers 1
done
```

Begin with `--num-samples 10` into temporary output roots for a calibration
visual check, then generate fully into fresh roots. Default faults are fog
severities 4/5, FOV loss and total loss, using the same injector, reference BEV,
reliability maps and observability logic as VoD. The existing `--fault-plan`
option can create separate fault datasets. Weather parameters have not been
recalibrated for Aeva; treat them as controlled simulated faults.

Then run your existing coarse or fine training command with these DATA/RADAR
paths. Build selector masks with the same config as training. Keep the current
320x320 grid and feature-channel profile aligned with the model configuration.
Train a new coarse checkpoint on HeRCULES before using it for fine training;
architectural compatibility does not imply VoD weights generalize unchanged.

## PointPillars and training

The current encoders consume Aeva `[x,y,z,reflectivity]` and aligned/tracked
Continental `[x,y,z,rcs,compensated_doppler]`; physical attributes are not
multiplied by temporal weights. V2 exponential age/motion weights are saved as
`radar_point_weights` and applied to the compatibility BEV density (normalized
by effective frame support). The existing PointPillars max pooling is unchanged
and does **not** consume those soft weights. This intentionally differs from
the historical weighted V2 raster, rather than changing sensor channel meanings
or silently modifying the model architecture. Policy, scan transforms, selected
signs, tracking counts and filter counts are stored in artifact metadata.
Changed V2 policies invalidate matching sample/cache entries; use fresh roots
also when changing source calibration or files in place.

These existing configurations explicitly enable PointPillars for both sensors:

```bash
COARSE_CONFIG=configs/coarse_reconstruction_vod_pointpillars_hrnet_b32_context80_halo_existing_loss_unlimited_pillars_dropout020.json
FINE_CONFIG=configs/fine_diffusion_basic_unet_pointpillars_coarse_soft_iou_steps3.json
RUN=/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/hercules_v2_coarse_b32

"$PYTHON" -u -m models.two_stage_reconstruction_head.cache_fault_selector_masks \
  --data-root "$DATA" --config "$COARSE_CONFIG" --num-workers 1
"$PYTHON" -u -m models.two_stage_reconstruction_head.coarse_reconstruction.train_coarse_reconstruction \
  --data-root "$DATA" --radar-root "$RADAR" --output-root "$RUN" \
  --config "$COARSE_CONFIG" --device cuda --epochs 150 --batch-size 8 --num-workers 4

# After coarse training finishes:
"$PYTHON" -u -m models.two_stage_reconstruction_head.diffusion_process.train_fine_diffusion \
  --data-root "$DATA" --radar-root "$RADAR" \
  --coarse-checkpoint "$RUN/best_model.pt" \
  --output-root /mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/hercules_v2_fine_unet \
  --config "$FINE_CONFIG" --selector-config "$COARSE_CONFIG" \
  --device cuda --epochs 50 --batch-size 4 --validation-batch-size 4 --num-workers 4
```

Batch sizes above are conservative starting points, not hardware-tuned values.

## Online geometric augmentation (HeRCULES and VoD)

Both datasets share training-time augmentation. With `augmentation.enabled`
true in the chosen coarse/fine config, each sample's flip, XY translation, yaw
and scale are sampled from a seed derived from the run seed, epoch and relative
sample path. The same sample gets a fresh draw each epoch, reproducibly even
with persistent workers, different worker counts, or changed batch ordering.
An individual draw can coincide with a previous one; disabled/zero-range
operations remain identity. No augmented copies are stored on disk.

One shared transform moves faulty LiDAR/radar raw points, clean/faulty targets,
repair/halo/context masks and observability consistently. PointPillars runs on
the transformed points, not stale encoded features. Validation/test datasets
are constructed without augmentation. Flips, translation up to 0.5 m, yaw up
to 5 degrees and scale 0.95–1.05 are enabled in the configurations above.
This does not regenerate weather/FOV/total-loss faults every epoch: fault
samples remain the controlled generated inputs; geometric augmentation is online.
