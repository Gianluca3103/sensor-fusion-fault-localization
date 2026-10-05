# Joint radar–LiDAR relationship and range-view diffusion

This experiment uses the already trained clean-LiDAR teacher as a frozen
geometry target. It jointly trains a radar-only relation encoder, a
training-only local radar/clean-LiDAR cross-attention head, and a range-view
diffusion model. The cross-attention head compares nearby ray-depth features
from the two sensors and provides a feature target plus a clean-return
auxiliary loss. **Its clean-attended output is never fed to diffusion.**
Diffusion sees only radar-derived relation features and surviving faulty
LiDAR. Its target is absolute first-return depth on radar-supported missing
rays, so reconstruction is not limited to a 3 m correction around a selected
radar depth. One generated point per eligible LiDAR ray preserves first-return
geometry; multiple nearby rays may be supported by one radar pattern.

This is a new checkpoint format. Do not pass an old ray-depth blueprint or
range-view reconstruction checkpoint as `--resume`.

## Professor-machine inputs

Use a newly downloaded checkout containing these scripts. The previously
downloaded `sensor-fusion-relationship-0d5d8e1` checkout does not contain this
experiment. The clean teacher from that run remains usable even if the
blueprint student has not finished:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
PY="$BASE/Sensor-Fusion/.venv_model_v2/bin/python"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
CACHE="$BASE/sensor_fusion_outputs/vod_staged_cache_20261005"
TEACHER="$BASE/sensor_fusion_outputs/vod_radar_only_blueprint_0d5d8e1/teacher_best_checkpoint.pt"

test -s "$TEACHER"
test -s "$CACHE/angular_geometry.json"
test -d "$CACHE/samples/train"
test -d "$CACHE/samples/val"
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Set `REPO` to the new checkout before running the commands below.

If the clean teacher checkpoint is absent, train it without launching the old
blueprint student:

```bash
TEACHER_RUN="$BASE/sensor_fusion_outputs/vod_clean_relation_teacher"
cd "$REPO"
export PYTHONPATH="$REPO"
"$PY" -u -m scripts.train_ray_depth_blueprint \
  --samples-root "$CACHE/samples" --vod-root "$VOD" \
  --geometry "$CACHE/angular_geometry.json" \
  --output-root "$TEACHER_RUN" --teacher-only \
  --teacher-epochs 5 --batch-size 1 --grad-accum-steps 4 \
  --tile-rows 4 --tile-cols 64 --width 32 \
  --validate-every 5 --num-workers 2 --device cuda
TEACHER="$TEACHER_RUN/teacher_best_checkpoint.pt"
```

A two-frame smoke run checks the full forward, backward, validation, and
checkpoint path:

```bash
cd "$REPO"
export PYTHONPATH="$REPO"
"$PY" -u -m scripts.train_joint_relation_diffusion \
  --samples-root "$CACHE/samples" --vod-root "$VOD" \
  --geometry "$CACHE/angular_geometry.json" \
  --teacher-checkpoint "$TEACHER" \
  --output-root "$BASE/sensor_fusion_outputs/joint_relation_smoke" \
  --train-limit 2 --val-limit 2 --epochs 1 --validate-every 1 \
  --no-audit-train-at-end --num-workers 0 --device cuda
```

Then train on every VoD training frame. These are starting hyperparameters,
not tuned settings:

```bash
RUN="$BASE/sensor_fusion_outputs/vod_joint_relation_diffusion_80ep"
"$PY" -u -m scripts.train_joint_relation_diffusion \
  --samples-root "$CACHE/samples" --vod-root "$VOD" \
  --geometry "$CACHE/angular_geometry.json" \
  --teacher-checkpoint "$TEACHER" --output-root "$RUN" \
  --radar-variant radar_20frames_verified_doppler_radial \
  --no-radar-height-filter \
  --epochs 80 --batch-size 4 --grad-accum-steps 1 \
  --tile-rows 4 --tile-cols 64 --width 32 --hidden 32 \
  --learning-rate 0.0002 --alignment-weight 0.1 \
  --paired-weight 0.1 --validate-every 5 --num-workers 2 --device cuda
```

`last_checkpoint.pt` is saved every epoch; `best_checkpoint.pt` is selected by
validation loss. Resume with the same model settings and effective batch size,
adding
`--resume "$RUN/last_checkpoint.pt"`. The final epoch also evaluates training
frames with frozen weights and the same tile selection seed as validation.
Compare `train_eval` with `val` in `epoch_metrics.jsonl`; online training loss
is not a fair generalization comparison. The printed return precision/recall
and depth MAE are teacher-forced noisy-step diagnostics. They are not final
sampled-cloud accuracy or object-detection AP.

## Validation export

Use a separate output directory for each checkpoint and return threshold.
First smoke-test two frames; this partial export cannot be scored:

```bash
"$PY" -u -m scripts.export_joint_relation_vod \
  --vod-root "$VOD" --samples-root "$CACHE/samples" \
  --checkpoint "$RUN/best_checkpoint.pt" \
  --output-root "$BASE/sensor_fusion_outputs/joint_relation_export_smoke" \
  --limit 2 --steps 20 --return-threshold 0.5 --device cuda
```

For a matched detector comparison, omit `--limit` and use a fresh output root.
The exporter writes `lidar/reconstructed/training/velodyne/*.bin` and a
manifest for all 1,296 official validation frames. It copies every original
faulty point unchanged, adds only on missing radar-supported rays, and opens
no clean-LiDAR scan. Feed that reconstructed directory to
`prepare_vod_official_faults.py`, then evaluate the frozen detector on the
same clean, faulty, and reconstructed frame IDs.
