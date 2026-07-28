# Sensor Fusion Fault Localization

Thesis pipeline for detecting unreliable regions in degraded LiDAR BEV maps,
optionally conditioned on clean, temporally stacked 4D radar.

The repository contains three maintained stages:

1. Generate degraded LiDAR samples and exact ID-based reliability targets.
2. Train LiDAR-only PFS ablations or the final radar-conditioned PFS model.
3. Calibrate localization thresholds on validation data and evaluate test data.

Raw Hercules data, generated samples, radar caches, checkpoints, and run outputs
are intentionally excluded from Git.

[![Tests](https://github.com/Gianluca3103/sensor-fusion-fault-localization/actions/workflows/tests.yml/badge.svg)](https://github.com/Gianluca3103/sensor-fusion-fault-localization/actions/workflows/tests.yml)

## Repository Layout

```text
Fault_Localization_Model/
  create_grid_reliability_heatmaps.py  Dataset and target generation
  fault_injector.py                    Point-ID-preserving fault routing
  data_injection_utils.py              Weather/fault simulator adapters
  bev_utils.py                         LiDAR BEV projection
  heatmap_metrics.py                   Heatmap and localization metrics
  io_utils.py                          Atomic NPZ/checkpoint/JSON/CSV output
  model_blocks.py                      Shared neural-network building blocks
  sample_utils.py                      Shared sample metadata/filtering
  visualization_utils.py               Shared reliability-map rendering

PFS/
  datasets.py                          Shared LiDAR-only dataset loading
  pfs_model.py                         LiDAR-only PFS and ablation models
  training_utils.py                    Shared loss and artifact helpers
  train_pfs_reliability_map.py         LiDAR-only training
  calibrate_thresholds_eval_test.py    LiDAR-only calibration/evaluation

PFS_Radar/
  radar_data.py                        Radar loading, alignment, stacking, cache
  datasets.py                          Shared radar-conditioned datasets
  pfs_radar_model.py                   Final radar-conditioned model
  prepare_radar_cache.py               Pose-aligned radar cache generation
  train_pfs_radar.py                   Final model training
  calibrate_thresholds_eval_test.py    Final validation/test evaluation
  test_pfs_radar.py                    Side-by-side prediction visualization

Weather_Injector/                      Vendored upstream simulators
scripts/                               Machine pipeline and repair utilities
tests/                                 Unit tests
```

The former standalone v5 baseline is represented by
`PFS/train_pfs_reliability_map.py --model-variant no-pfs`. This avoids
maintaining a second copy of the same encoder-decoder pipeline.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux
# .\.venv\Scripts\Activate.ps1    # PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch CUDA build compatible with the machine's NVIDIA driver.
Verify it before a large run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Generate Reliability Targets

The production dataset uses a temporal `70/15/15` split inside every Hercules
scene, a `320x320` target grid, and a `0.05 m` point-movement tolerance.

```bash
python Fault_Localization_Model/create_grid_reliability_heatmaps.py \
  --data-root "/path/to/HeRCULES" \
  --all-scenes \
  --temporal-split train \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --output-root "/path/to/output/train" \
  --num-samples 70000 \
  --fault-plan fog_sim:5 rain_sim:5 snow_sim:5 old_laser_degradation:0 fov_filter:1 \
  --x-min 0 --x-max 64 \
  --y-min -32 --y-max 32 \
  --resolution 0.20 \
  --grid-size 320 \
  --movement-tolerance-m 0.05 \
  --num-workers 16 \
  --weather-threads 1 \
  --no-previews \
  --seed 42
```

Run the same command for `val` and `test` with 15,000 samples each and separate
output folders. `scripts/run_100k_stack20_machine.sh` automates the full
100,000-sample radar-conditioned pipeline on the configured Linux machine.

Generation is deterministic and resumable. Samples are written atomically.
Every run rewrites a complete `manifest.csv`, including samples reused during
resume, so the manifest remains a faithful inventory after interruption.
Use `scripts/remove_corrupt_npz.py` only to repair data created by older
non-atomic versions.

Generator version 3 derives an independent deterministic injection seed for
every sample. This removes the old behavior where every sample at one severity
shared a stochastic pattern; FOV loss also rotates across samples. Existing
older samples intentionally fail the resume metadata check and are regenerated
rather than mixed with version-3 data. Version 3 also assigns tolerated
sub-5-cm point motion to the point's observed cell, preventing cell-edge
reliability errors.
When multiple generation processes are used, keep `--weather-threads 1` to
avoid nested LISA thread pools oversubscribing the CPU.

## Ground-Truth Definition

Every clean LiDAR point receives a frame-local source ID before injection.
Injectors preserve that relationship explicitly:

```text
faulty points = missing + moved (> movement tolerance) + added
reliability   = correct / (correct + faulty)
fault target  = 1 - reliability
```

Missing points are assigned to their clean location. Moved and synthetic points
are assigned to their degraded location. No nearest-neighbour correspondence or
fine occupancy grid is used to create the target.

## Train the Final Radar Model

Prepare the causal 20-frame, pose-aligned radar cache first:

```bash
python PFS_Radar/prepare_radar_cache.py \
  --dataset-root "/path/to/dataset" \
  --hercules-root "/path/to/HeRCULES" \
  --output-root "/path/to/radar_cache" \
  --radar-frame-count 20 \
  --require-full-stack \
  --max-delta-ms 30 \
  --num-workers 12
```

Cache files are validated against the source timestamp, BEV geometry, velocity
normalization, alignment tolerance, and requested stack length. Missing,
corrupt, one-frame, or otherwise incompatible entries are rebuilt atomically.
Cache format version 3 is causal and session-aware: it never selects a future
radar frame and stores entries under their exact scene/session path. Rebuild
older caches before training with this code.

Then train:

```bash
python PFS_Radar/train_pfs_radar.py \
  --train-root "/path/to/dataset/train" \
  --val-root "/path/to/dataset/val" \
  --radar-root "/path/to/radar_cache" \
  --output-root "/path/to/run" \
  --epochs 150 \
  --batch-size 64 \
  --num-workers 12 \
  --base-channels 16 \
  --dropout 0.15 \
  --learning-rate 7.5e-5 \
  --min-learning-rate 1e-6 \
  --warmup-epochs 10 \
  --weight-decay 2e-3 \
  --stability-weight 0.05 \
  --pfs-reliability-weight 0.10 \
  --localization-loss-weight 0.25 \
  --false-positive-weight 0.65 \
  --localization-tolerance-m 0.20 \
  --metric-threshold 0.15 \
  --metric-grid-size 320 \
  --grid-size 320 \
  --metrics-every 5 \
  --early-stop-patience 20 \
  --min-delta 1e-4 \
  --exclude-faults rain_sim snow_sim \
  --device cuda
```

Radar is never fault-injected. Clean LiDAR is used only by the
feature-stabilization loss during training and is not an inference input.
The clean reference is encoded without gradients or BatchNorm-statistic
updates, then fused with the same detached radar bottleneck as the degraded
branch. This makes the stabilization loss compare like-for-like fused features.
Training and calibration fail fast if one physical source frame appears in
more than one split.

See `PFS_Radar/README.md` for cache, visualization, calibration, refinement,
and overfitting-analysis commands.

## LiDAR-Only Ablations

`PFS/train_pfs_reliability_map.py` supports:

- `pfs`: all three PFS-inspired blocks.
- `pfs-block12`: Blocks 1 and 2 without expert correction.
- `lidar-only`: LiDAR-specific reliability and geometric correction.
- `no-pfs`: plain U-Net-style encoder-decoder baseline.

These variants share one data loader, trainer, evaluator, and visualization
implementation.

## Evaluation Protocol

Choose the prediction threshold on validation data, freeze it, and evaluate the
test split once. Localization metrics use maximum-cardinality one-to-one
matching: one predicted cell and one ideal faulty cell may match only once when
their metric distance is within `--localization-tolerance-m`. This prevents a
broad prediction from receiving multiple true positives from one target.

This matching rule is stricter than the earlier many-to-one implementation.
Recalibrate thresholds and recompute all reported metrics; old localization
scores are not directly comparable.

## Checkpoint Compatibility

Training checkpoints carry a training-semantics version in addition to model,
optimizer, scheduler, RNG, history, and early-stopping state. `--resume` is only
for an exact interrupted run and rejects checkpoints from before the corrected
clean-reference objective. For PFS-Radar, `--init-checkpoint` may load compatible
trusted weights into a fresh run, but final thesis results should be retrained
under the current objective.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers point provenance, reliability targets, one-to-one
localization, atomic writes, resumable manifests, split leakage, radar
projection, causal stacking, session paths, pose transforms, reference-branch
normalization, model shapes, and checkpoint contracts.
