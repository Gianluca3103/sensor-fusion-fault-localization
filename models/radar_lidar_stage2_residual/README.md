# Faulty-LiDAR-conditioned residual Stage II

This is a new additive model. Existing radar-only Stage-II checkpoints and
exports remain usable and are not silently loaded as residual checkpoints.
Diffusion remains **off**.

## Inputs and supervision

1. Frozen radar-only Stage I encodes the aligned 20-frame radar stack. A
   learned-region checkpoint supplies predicted LiDAR XYZ, confidence, and
   support extents; the complete patches seed the fine-voxel candidate domain.
   A legacy checkpoint still seeds from occupied radar voxels. The checkpoint type is determined
   by its Stage-I configuration and is recorded in candidate counts.
2. The cached faulty LiDAR is paired by official VoD frame ID. Seven
   faulty-only candidate features include same-voxel occupancy, capped local
   density at 0.4 and 1.0 m, nearest-point distance, nearest intensity, a
   narrow measured-ray mask, and an in-fault-region mask.
3. A sparse U-Net combines radar and faulty features and predicts an
   **addition probability** and point offset at each candidate. Surviving
   faulty points are retained unchanged. A measured voxel or ray, or a site
   outside the fault cache crop, cannot be exported as an addition.
4. Clean LiDAR enters only the training target and validation metrics. A
   positive is a clean occupied candidate that is not already covered by the
   faulty scan. Observed candidates and clean-ray visible free cells are
   negatives. Unmeasured/occluded cells are not labeled empty. Positive,
   observed-negative and free-negative losses are normalized separately so
   the common observed class cannot dominate solely by count. Point offsets
   are supervised only at true additions.

Fog can move or add faulty returns. This additive model retains those returns
and its observed-ray mask can suppress a valid addition behind a fog ghost.
That limitation must be measured by fault type; this model does not repair
false returns. For cropped cache artifacts, `point_filter` defines the region
where the fault was injected; candidates outside it are neither learned nor
exported. For the staged full-scan cache, `range_view_full_scan=true` explicitly
marks the whole LiDAR scan as the affected region.

## Professor-machine smoke test and training

Run from the updated repository checkout. Use the same working MinkowskiEngine
Python and user-space OpenBLAS library as the previous Stage-II run:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
PY="$BASE/Sensor-Fusion/.venv_model_v2/bin/python"
FAULTS="$BASE/sensor_fusion_outputs/vod_staged_cache_20261005/samples"
STAGE1=/absolute/path/to/the/Stage-I/best_selected.ckpt
RUN="$BASE/sensor_fusion_outputs/stage2_residual_20frame_50ep"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8

"$PY" -m scripts.smoke_stage2_residual_minkowski
"$PY" -u -m models.radar_lidar_stage2_residual.train \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --stage1-checkpoint "$STAGE1" --output-root "$RUN" \
  --config configs/radar_lidar_stage2_residual.json \
  --radar-variant radar_20frames_verified_doppler_radial \
  --epochs 50 --batch-size 4 --grad-accum-steps 1 \
  --validate-every 5 --num-workers 2 --device cuda
```

The trainer saves `last.ckpt`, `best_global_missing.ckpt`, and `metrics.jsonl`.
Resume with `--resume "$RUN/last.ckpt"` and the same arguments. Selection uses
global missing-clean-point F1 at 0.2 m. `addition_f1_0.2m` in the metrics is
**candidate-conditioned** and will usually be larger; it is not a global
recovery score. Clean points within 5 cm of faulty points count as surviving
for the validation-only global denominator.

Before detector evaluation, export two frames with `--limit 2` to a separate
smoke directory. The full export has no `--limit`:

```bash
"$PY" -u -m scripts.export_stage2_residual_vod_detector \
  --vod-root "$VOD" --checkpoint "$RUN/best_global_missing.ckpt" \
  --fault-samples-root "$FAULTS" \
  --output-root "$BASE/sensor_fusion_outputs/stage2_residual_detector_export" \
  --device cuda
```

The exporter opens radar and faulty LiDAR only. The generated points have
intensity zero, and the original faulty XYZI rows are preserved exactly before
the detector's forward-FOV filter. Its manifest records the checkpoint and
input provenance. Prepare matched detector inputs with the existing
`prepare_vod_official_faults.py` tool before running the frozen SVEFusion
comparison.

## Cheap ablation before retraining

The old radar-only exporter supports `--exclude-observed-rays` in merged mode.
It applies the same measured-ray/voxel suppression *after* the old model runs.
Compare this export with the previous merged export to estimate how much of
the detector degradation comes from duplicate points alone. Use a fresh
output root; the export manifest will reject a changed mode.
