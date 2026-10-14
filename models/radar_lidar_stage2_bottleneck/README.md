# Stage II bottleneck sparse diffusion

This is a separate Stage II experiment. It requires a trained **radar-seed, bottleneck-support Stage I** checkpoint (`stage1_interface=bottleneck_support_v2`). The current Stage I radar-only student is the only Stage I network used by Stage II. Clean LiDAR is read for supervised targets and validation, never for candidate creation or inference conditioning.

## Actual Stage I interface

`load_encoder` in `scripts/visualize_stage1_confidence_cloud.py` extracts the `radar_only` weights from the Stage I checkpoint. `RadarOnlyBottleneckEncoder` returns `BottleneckSupportOutput`:

| Output | Coordinates | Features | Role |
| --- | --- | --- | --- |
| `bottleneck` | sorted sparse `(batch,z,y,x)` at stride 8 | `[N8, 64]` with the supplied Stage I config | Frozen LiDAR-aware radar conditioning |
| `confidence_region` | radar-derived sparse `(batch,z,y,x)` at stride 2 | `[N2, 1]` score in `[0,1]` | Learned repair-support seeds, not occupancy |

The project grid is normally `(0.2, 0.2, 0.25) m` in XYZ, from `(-128,-128,-32)` to `(128,128,32) m`. Thus Stage I support seeds have `(0.4,0.4,0.5) m` cells and the shared stride-8 bottleneck has `(1.6,1.6,2.0) m` cells. Stage I trains a radar/clean-LiDAR bottleneck teacher with local cross-attention. Its radar-only student distills that representation and predicts scored support seeds. Stage II loads only the student, calls it once per raw frame, then freezes it. Cached training reuses that student's features without rerunning Stage I each epoch.

## Stage II architecture

`make_candidates` uses the existing **edge-aware interpolation** of Stage I support seeds to produce uncapped, sorted unique fine-grid candidates. It does not rasterize spheres, use clean LiDAR, or interpret support confidence as occupied probability. A confidence threshold selects candidate sites. Faulty-LiDAR coverage is then computed from the surviving scan, including observed voxels and measured rays. Stage II can add points only at candidates in the fault region that are not already covered by a surviving measurement. Original faulty points are preserved for the completed cloud.

Each candidate has a four-value diffusion state: `(+1 or -1, 2*dx, 2*dy, 2*dz)`. Positive offsets are normalized within the fine voxel; the occupancy sign denotes an **additional** LiDAR point. Unknown clean space is imputed independently of clean geometry and excluded from the loss. Protected cells are trained as no-addition and clamped as no-addition during sampling. The model predicts four noise values, one auxiliary addition logit, and three bounded within-voxel offsets. `decode_centroids` turns selected offsets into physical XYZ points. One point is predicted per selected occupied fine voxel; multiple points within a voxel are not represented.

With the example config, Stage I fine conditioning is projected to 24 channels, faulty coverage's seven features to 24 channels, then these 48 values plus four noisy-state values are projected to 24. The MinkowskiEngine U-Net has widths `(24,48,96,128)`, three stride-2 downsamples, one timed residual block per level, and symmetric transpose-convolution/skip decoding. A 128-dimensional timestep embedding produces FiLM scale/shift in every residual block.

There is **one** trainable cross-attention block after the stride-8 encoder and before the decoder. It takes `[M8,128]` U-Net queries, finds at most 16 same-frame Stage I source voxels within 6 m using explicit XYZ, and projects frozen `[N8,64]` source features into 128-dimensional keys/values. Four heads use metric relative positions. Queries without a source neighbor pass unchanged. The attention, U-Net, and heads train jointly; Stage I does not. `fusion_mode=concat` replaces only this bottleneck block with spatially matched pooled-source concatenation and a learned projection, preserving the same U-Net and training protocol.

The forward process uses a configurable linear or cosine beta schedule and predicts Gaussian noise. Default sampling is all 50 DDPM steps. `--sampling-method ddim --sampling-steps 10` skips steps for a faster, separately measured inference path. Candidate coordinates and Stage I features stay fixed through every reverse step; sparse coordinate maps and bottleneck neighbor indices are reused during sampling.

## Loss and evaluation boundaries

Training reports each weighted term separately: diffusion noise MSE (positive geometry channels and observable negative occupancy channels), addition BCE on clean positive / measured free sites, positive-voxel smooth-L1 offsets, and a no-addition penalty on protected candidates. Clean-ray free space is negative evidence only before the clean return; unobserved space behind it is **unknown**, with no occupancy or geometry target. The default weights are `1.0, 0.25, 0.25, 0.25`. No Chamfer training loss is used because nearest-neighbor assignments over sparse sampled points would add a costly, unstable objective; Chamfer is a validation metric.

Validation runs full sampling and reports eligible-clean recall, observed voxel precision/F1/IoU, global missing-cloud F1, point distances and symmetric Chamfer, 0.2/0.5 m tolerances, unverified predictions in unknown space, preservation counts, fault-specific scores, inference seconds/frame, and peak allocated GPU memory. Known free-space false positives and **unknown-space predictions are separate**; the latter cannot honestly be called wrong from one LiDAR scan. The model cannot reconstruct outside the learned support, and a one-point-per-voxel output does not reproduce the exact native LiDAR scan pattern.

## Professor machine commands

Run these from a source checkout containing this module. Use the full radar-seed Stage I checkpoint trained on the same radar variant. `FAULTS` must contain exactly one cached faulty scan per requested frame and split.

```bash
set -euo pipefail
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/Sensor-Fusion/.venv_model_v2/bin/python"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
FAULTS="$BASE/sensor_fusion_outputs/vod_staged_cache_20261005/samples"
STAGE1="$BASE/sensor_fusion_outputs/stage1_radar_seeds_0ec4fc9_50ep/best_selected.ckpt"
export PYTHONPATH="$REPO"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
cd "$REPO"
test -s "$STAGE1"

"$PY" -m scripts.smoke_stage2_bottleneck_minkowski

"$PY" -u -m models.radar_lidar_stage2_bottleneck.train \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --stage1-checkpoint "$STAGE1" \
  --output-root "$BASE/sensor_fusion_outputs/stage2_bottleneck_smoke" \
  --config configs/radar_lidar_stage2_bottleneck.json \
  --epochs 1 --batch-size 1 --train-limit 1 --val-limit 1 \
  --validate-every 1 --validation-sampling-method ddim \
  --validation-sampling-steps 10 --num-workers 0 --device cuda --amp bf16
```

For the optional deterministic preprocessing cache, select a volume with sufficient free space. Its files include candidate coordinates, frozen Stage I features, faulty coverage, clean training targets, and validation clouds. A manifest hashes the Stage I checkpoint, code, data roots, radar variant, fault pattern, and geometry config; mismatches fail rather than silently reusing stale data. Check disk space before starting because the full 6,435-frame cache may be large.

```bash
CACHE=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/stage2_bottleneck_cache
df -h "$(dirname "$CACHE")"
"$PY" -u -m scripts.cache_stage2_diffusion \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --stage1-checkpoint "$STAGE1" --output-root "$CACHE" \
  --config configs/radar_lidar_stage2_bottleneck.json \
  --compression fast --device cuda

RUN="$BASE/sensor_fusion_outputs/stage2_bottleneck_25ep"
mkdir -p "$RUN"
"$PY" -u -m models.radar_lidar_stage2_bottleneck.train \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --stage1-checkpoint "$STAGE1" --preprocess-cache-root "$CACHE" \
  --output-root "$RUN" --config configs/radar_lidar_stage2_bottleneck.json \
  --epochs 25 --batch-size 4 --grad-accum-steps 1 \
  --learning-rate 0.0002 --validate-every 5 \
  --validation-sampling-method ddim --validation-sampling-steps 10 \
  --num-workers 2 --device cuda --amp bf16 2>&1 | tee "$RUN/console.log"
```

To resume, pass `--resume "$RUN/last.ckpt"` with the same arguments. For full DDPM validation and a 10-step DDIM comparison:

```bash
CKPT="$RUN/best_global_missing.ckpt"
"$PY" -m models.radar_lidar_stage2_bottleneck.evaluate \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --checkpoint "$CKPT" --preprocess-cache-root "$CACHE" \
  --output "$RUN/val_ddpm50.json" --device cuda
"$PY" -m models.radar_lidar_stage2_bottleneck.evaluate \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --checkpoint "$CKPT" --preprocess-cache-root "$CACHE" \
  --sampling-method ddim --sampling-steps 10 \
  --output "$RUN/val_ddim10.json" --device cuda

"$PY" -m scripts.visualize_stage2_bottleneck \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --checkpoint "$CKPT" --split val --frame-id 08236 04752 00405 \
  --output-root "$RUN/views" --device cuda
```

The visualization writes uncapped PLY clouds and HTML views of additions, Stage I support, and addition error. The completed PLY is the original faulty scan concatenated with additions. To run the simple-fusion ablation, copy the JSON config, set `fusion_mode` to `concat`, and give training a new output root. The existing cache can be shared because its provenance includes the **geometry** config but not the fusion choice.

## Local verification

The CPU suite exercises the schedule, labels, frozen-loader contract, local attention and gradients, no-neighbor behavior, candidate activation, decoding, no-clean-input boundary, reverse sampling, and both fusion modes. The synthetic MinkowskiEngine smoke test is required on a machine with a working ME build. A one-frame real-data train/validation run then verifies Stage I loading, dataset/fault pairing, sparse convolution, AMP, sampling, and checkpoint writing before full training. Passing CPU tests alone does not establish that the professor-machine MinkowskiEngine kernels work.
