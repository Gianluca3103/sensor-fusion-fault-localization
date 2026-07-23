# PFS-Radar LiDAR Fault Localization

This folder adapts the Post Fusion Stabilizer (PFS) pipeline to use clean Continental 4D radar in place of the paper's camera modality. The objective remains LiDAR fault localization.

## Data flow

```text
degraded LiDAR BEV -> LiDAR encoder ----+
                                         +-> BEV fusion -> PFS Blocks 1-3 -> decoder -> LiDAR fault heatmap
clean radar BEV    -> radar encoder -----+
```

Radar is never corrupted. Clean LiDAR is available only during training for the feature-stabilization loss and is not required at inference.

The four radar channels are:

1. occupancy
2. log-normalized point density
3. normalized absolute radial velocity
4. normalized radar cross section (RCS)

The radar input is a causal stack containing the frame nearest the current
LiDAR timestamp and the preceding 19 frames. Each historical cloud is
ego-motion compensated into the current Aeva LiDAR frame using the
sensor-specific `Continental_gt.txt` and `Aeva_gt.txt` trajectories plus the
IMU/sensor mounting rotations. The aligned points are concatenated before one
BEV projection. A cache is used so multiple fault variants of the same LiDAR
frame share one radar tensor.

Strict cache preparation rejects sequence-start samples that do not have 20
real historical radar frames or a valid ground-truth pose. It never fills the
window with future frames.

## PFS adaptation

- **Block 1 - shift normalization:** stabilizes global statistics of the fused LiDAR-radar BEV feature.
- **Block 2 - spatial reliability:** combines shifted fused features with raw LiDAR bottleneck features and predicts LiDAR reliability.
- **Block 3 - expert correction:** uses reliability as a hole map; the former camera/semantic expert becomes a radar-supported correction expert.

The final decoder predicts the supervised LiDAR fault heatmap. The internal PFS reliability output is supervised with `1 - fault_heatmap`.

## 1. Prepare radar cache

```bash
PYTHON="/home/arrubuntu20/anaconda3/envs/sensor-fusion/bin/python"
REPO_ROOT="/mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization"
HERCULES_ROOT="/mnt/3D10B36523559581/HeRCULES"
DATASET_ROOT="/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/grid_reliability_20k_grid320_id_v1"
RADAR_ROOT="/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/radar_cache_20k_stack20"

cd "$REPO_ROOT"
"$PYTHON" PFS_Radar/prepare_radar_cache.py \
  --dataset-root "$DATASET_ROOT" \
  --hercules-root "$HERCULES_ROOT" \
  --output-root "$RADAR_ROOT" \
  --num-workers 4 \
  --max-delta-ms 30 \
  --radar-frame-count 20 \
  --require-full-stack
```

Use a new cache root and retrain from scratch when changing from one radar
frame to 20. The tensor still has four channels, but its data distribution is
substantially denser.

## 2. Train

```bash
RUN_ROOT="/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/pfs_radar_20k_grid320"

"$PYTHON" PFS_Radar/train_pfs_radar.py \
  --train-root "$DATASET_ROOT/train" \
  --val-root "$DATASET_ROOT/val" \
  --radar-root "$RADAR_ROOT" \
  --output-root "$RUN_ROOT" \
  --epochs 100 \
  --batch-size 16 \
  --num-workers 8 \
  --base-channels 16 \
  --dropout 0.15 \
  --learning-rate 2e-4 \
  --min-learning-rate 1e-6 \
  --warmup-epochs 10 \
  --weight-decay 2e-3 \
  --stability-weight 0.05 \
  --pfs-reliability-weight 0.10 \
  --localization-loss-weight 0.25 \
  --false-positive-weight 0.70 \
  --localization-radius-cells 1 \
  --early-stop-patience 10 \
  --metric-threshold 0.15 \
  --metric-grid-size 320 \
  --metrics-every 1 \
  --localization-tolerance-m 0.20 \
  --target-fault-threshold 0.0 \
  --grid-size 320 \
  --device cuda
```

Validation prints localization IoU, precision, recall, and F1 each epoch. Early
stopping monitors validation loss with a default patience of 10. Training saves
both `best_model.pt` (validation loss) and `best_localization_iou.pt`
(localization IoU). Increase `--false-positive-weight` cautiously if predictions
remain too broad.

## 3. Visualize test predictions

```bash
"$PYTHON" PFS_Radar/test_pfs_radar.py \
  --test-root "$DATASET_ROOT/test" \
  --radar-root "$RADAR_ROOT" \
  --checkpoint "$RUN_ROOT/checkpoints/best_model.pt" \
  --output-root "$RUN_ROOT/test_predictions" \
  --max-images 30 \
  --visual-grid-size 320 \
  --prediction-threshold 0.045 \
  --localization-tolerance-m 0.20 \
  --device cuda
```

The comparison image contains degraded LiDAR, clean radar, ideal LiDAR reliability, predicted radar-conditioned reliability, and localization matching.

## 4. Calibrate and evaluate metrics

Use validation data to select the prediction threshold, then evaluate the test
set once with that frozen threshold:

```bash
"$PYTHON" PFS_Radar/calibrate_thresholds_eval_test.py \
  --val-root "$DATASET_ROOT/val" \
  --test-root "$DATASET_ROOT/test" \
  --radar-root "$RADAR_ROOT" \
  --checkpoint "$RUN_ROOT/checkpoints/best_model.pt" \
  --output-root "$RUN_ROOT/calibrated_metrics_20cm" \
  --batch-size 32 \
  --num-workers 4 \
  --resize-height 320 \
  --resize-width 320 \
  --grid-size 320 \
  --thresholds 0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.10 0.12 0.15 0.20 0.25 0.30 0.40 0.50 \
  --select-metric localization_iou \
  --localization-tolerance-m 0.20 \
  --target-fault-threshold 0.0 \
  --device cuda
```

The command prints progress and final validation/test metrics. It also saves the
threshold sweep, aggregate test metrics, and per-fault test metrics.

## 5. Locate overfitting

```bash
"$PYTHON" PFS_Radar/analyze_overfitting.py \
  --train-root "$DATASET_ROOT/train" \
  --val-root "$DATASET_ROOT/val" \
  --radar-root "$RADAR_ROOT" \
  --checkpoint "$RUN_ROOT/checkpoints/best_localization_iou.pt" \
  --output-root "$RUN_ROOT/overfitting_diagnosis" \
  --batch-size 16 \
  --num-workers 8 \
  --threshold 0.15 \
  --device cuda
```

This writes train/validation spatial error maps and a per-fault JSON summary.
Positive regions in the validation-minus-train maps identify where validation
error exceeds training error.

## 6. Low-learning-rate refinement

Use `--init-checkpoint` to load model weights while resetting the optimizer,
scheduler, early-stopping counter, and history. This differs from `--resume`,
which restores an interrupted run exactly.

```bash
"$PYTHON" PFS_Radar/train_pfs_radar.py \
  --train-root "$DATASET_ROOT/train" \
  --val-root "$DATASET_ROOT/val" \
  --radar-root "$RADAR_ROOT" \
  --output-root "${RUN_ROOT}_refined" \
  --init-checkpoint "$RUN_ROOT/checkpoints/best_localization_iou.pt" \
  --epochs 50 \
  --learning-rate 5e-5 \
  --min-learning-rate 1e-6 \
  --warmup-epochs 0 \
  --early-stop-patience 15
```
