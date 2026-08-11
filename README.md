# Sensor Fusion Fault Localization

Thesis pipeline for detecting unreliable regions in degraded LiDAR BEV maps,
optionally conditioned on clean, paired-frame 4D radar.

The repository contains three maintained stages:

1. Generate degraded LiDAR samples and exact ID-based reliability targets.
2. Train LiDAR-only PFS ablations or the final radar-conditioned PFS model.
3. Calibrate localization thresholds on validation data and evaluate test data.

Raw K-Radar data, generated samples, radar caches, checkpoints, and run outputs
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
  radar_data.py                        Radar cache lookup and loading
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

The production dataset uses the radar-paired K-Radar `os2-64` LiDAR frames,
with a temporal `70/15/15` split inside every sequence, a `320x320` target
grid, and a `0.05 m` point-movement tolerance. Discovery requires the paired
`lidar/<sequence>/info_label` entry and
`radar/pc10p/<sequence>/rpc_<index>.npy`, preventing
samples without radar conditioning from entering the dataset.

Although `os2-64` is a 360-degree LiDAR, generation keeps only points inside
K-Radar's polar support: azimuth approximately `[-53, 53]` degrees and range
up to `118.04 m`, plus its `[-18,18]` degree elevation support, evaluated in
the calibrated radar coordinate frame. The BEV remains `x=[0,64)`,
`y=[-32,32)`, at `0.2 m` resolution.

```bash
python -m Fault_Localization_Model.create_grid_reliability_heatmaps \
  --data-root "/path/to/K-Radar_Data" \
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
  --no-previews \
  --seed 42
```

Run the same command for `val` and `test` with 15,000 samples each and separate
output folders.

Generation is deterministic and resumable. Samples are written atomically.
Every run rewrites a complete `manifest.csv`, including samples reused during
resume, so the manifest remains a faithful inventory after interruption.
Use `scripts/remove_corrupt_npz.py` only to repair data created by older
non-atomic versions.

Generator version 10 uses paired K-Radar `os2-64`/pc10p frames and applies the
calibrated radar-overlap crop to clean and corrupted points. Its three LiDAR
channels are binary occupancy, log point density, and per-cell 90th-percentile
upper height normalized over the same `[-3, 5]` metre interval as radar. It
also traces raw clean-LiDAR rays before fault injection to store a separate
`320x320` observability-confidence target. Temporary 16-bin vertical coverage
and saturating ray support are supervision metadata only; they are not model
inputs and do not change clean or faulty BEV generation. The generator retains
the independent deterministic injection seed and assigns tolerated sub-5-cm
point motion to the observed cell. Older representations intentionally fail
the resume metadata check and are not mixed with version 10 data.
When multiple generation processes are used, keep `--weather-threads 1` to
avoid nested LISA thread pools oversubscribing the CPU.

Render the clean occupancy, ray count, vertical coverage, ray support, and
confidence for one source frame with:

```bash
python scripts/visualize_lidar_observability.py \
  --lidar-pcd "/path/to/lidar/1/os2-64/os2-64_00001.pcd" \
  --calibration "/path/to/lidar/1/info_calib/calib_radar_lidar.txt" \
  --output "/path/to/observability_debug.png"
```

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

Prepare the single-frame K-Radar pc10p cache first:

```bash
python -m PFS_Radar_v2.prepare_radar_cache \
  --kradar-root "/path/to/K-Radar_Data" \
  --radar-point-root "/path/to/K-Radar_Data/radar/pc10p" \
  --odometry-root "/path/to/K-Radar_Data/support/official_k_radar/resources/odometry" \
  --output-root "/path/to/radar_cache" \
  --num-workers 12
```

Cache files use the exact `sequence/radar_index` pairing stored in each
generated LiDAR sample. Cache-format and channel metadata prevent obsolete
radar representations from being reused by the cache builder.

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

## Coarse Reconstruction Backbone Ablation

The PointPillars coarse-reconstruction stage supports two controlled backbones:

- `model.backbone: unet` keeps the original dense compressive U-Net, global
  encoders, and bottleneck cross-attention.
- `model.backbone: sst` keeps the same independent LiDAR and radar PFNs but
  operates on their union of sparse pillar coordinates with six single-stride
  regional Transformer blocks. Coordinates remain on the original `320x320`
  grid throughout; no pooling or stride-2 operation is used.

The strict SST experiment uses 12-cell (2.4 m) regions, a six-cell shift,
128-D tokens, eight attention heads, and no repair-query tokens. Each token is
the projection of `[LiDAR PFN (64), radar PFN (64), reconstruction mask,
healthy-context mask]`. LiDAR pillars inside the reconstruction mask are
removed before this fusion. Missing modalities are represented by zeros.

Use `configs/coarse_reconstruction_pointpillars.json` for the U-Net experiment
and `configs/coarse_reconstruction_pointpillars_sst.json` for SST. Dataset
splits, PointPillars settings, masks, targets, loss, metrics, optimizer, and
training schedule remain identical. `include_repair_tokens` intentionally
defaults to `false`; enabling it is a separate reconstruction-specific
ablation rather than part of the strict backbone comparison.

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
projection, paired-frame lookup, session paths, pose transforms, reference-branch
normalization, model shapes, and checkpoint contracts.
