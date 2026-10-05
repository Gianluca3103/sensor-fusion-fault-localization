# Staged radar-to-LiDAR ray-depth training

The first command trains a clean-LiDAR geometry teacher, then a radar-only
ray-depth blueprint on **all** VoD training frames. Both phases use the same
radar-derived ray-depth queries. The teacher learns clean first returns and
local features; the deployed student attends to the Doppler-aligned 20-scan
radar stack and matches those features where radar and clean LiDAR overlap.
Faulty LiDAR is excluded from the relationship blueprint, including candidate
selection. It remains an input to the unchanged range-view diffusion stage.
Clean LiDAR never enters the student forward pass or diffusion conditioning.

## Stage 1: blueprint only

These are starting hyperparameters, not a VoD optimum. Keep batch size one:
each scene needs its own tile selected around radar evidence. Four gradient
accumulation steps combine four
independently selected scene tiles in each optimizer update. The `4 × 64` tile
makes 1,280 ray-depth candidates, below the cross-attention limit of 2,048.
The chosen tile never uses the clean or faulty scan to pick a position.

```bash
set -e
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/svefusion-clean-cu118/bin/python"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
CACHE="$BASE/sensor_fusion_outputs/vod_range5_full_cache"
RUN="$BASE/sensor_fusion_outputs/vod_ray_depth_blueprint"

test -s "$CACHE/angular_geometry.json"
test -d "$CACHE/samples/train"
test -d "$CACHE/samples/val"
test -d "$VOD/radar_20frames_verified_doppler_radial/training/velodyne"

cd "$REPO"
"$PY" -u -m scripts.train_ray_depth_blueprint \
  --samples-root "$CACHE/samples" \
  --vod-root "$VOD" \
  --geometry "$CACHE/angular_geometry.json" \
  --output-root "$RUN" \
  --teacher-epochs 5 --epochs 10 --batch-size 1 --grad-accum-steps 4 \
  --tile-rows 4 --tile-cols 64 --width 32 \
  --learning-rate 0.0002 --distill-weight 0.1 --validate-every 5 \
  --num-workers 4 --device cuda
```

The teacher trains for five epochs, then the radar-only student for ten.
Inspect student validation after epochs 5 and 10. Continue the student to 30
or 50 only while validation improves. To continue the same run, use the
same command with `--epochs 30` or `--epochs 50` and
`--resume "$RUN/last_checkpoint.pt"`; keep all other settings the same. This
reloads the frozen teacher and does not repeat teacher training.
If teacher pretraining is interrupted before student epoch one, use the same
command with `--resume-teacher "$RUN/teacher_last_checkpoint.pt"`. This
restores its weights and optimizer, then starts the student stage when the
teacher reaches its requested epoch count.
Gradient accumulation changes update frequency, not the number of scenes
processed per epoch. The clean teacher adds a separate training pass, so
measure several real frames on the training machine before committing to a
long run.

The Doppler-shifted radar variant is selected by default. The optional
per-frame height filter is **off** because it uses faulty LiDAR to modify the
radar input, which would make the relationship blueprint indirectly depend on
faulty LiDAR. Before a long run, add `--train-limit 2 --val-limit 2 --epochs 1
--validate-every 1` and a separate output root for a short data check. This
smoke run does not estimate blueprint quality.

Each validation prints loss, candidate coverage, and depth-aware precision,
recall, and F1 at a 3 m error tolerance. Those scores use only radar-supported
rays missing from observed faulty LiDAR; faulty LiDAR is read for this metric
only. Check `clean_hits` and
`radar_supported_missing_rays` before interpreting them: if either is zero,
the sampled validation tiles did not test actual reconstruction. `best_checkpoint.pt`
is selected by validation depth F1; `last_checkpoint.pt` is saved each epoch.
`teacher_best_checkpoint.pt` and `teacher_metrics.jsonl` record teacher
pretraining. Student metrics also include the detached feature loss and the
number of radar/clean feature pairs actually used. At the final student epoch,
the command also evaluates all training frames with frozen weights
(`train_eval`) and the same deterministic tile-selection rule as validation.
Compare `train_eval` with `val`; online `train` loss was measured while weights
were changing and is not a fair gap estimate.
Candidate coverage measures whether a clean return has a nearby proposal, not
whether the model selected the right return. The generated cloud and detector
AP must be evaluated separately after later stages.

New runs print a short train summary after each epoch and a second validation
line when validation runs. All metrics remain available as JSON records in
`epoch_metrics.jsonl` under each run's output directory. A Python process
already running when this output change is installed keeps its earlier format.

## Later stages

The unchanged range-view diffusion trainer accepts
`--pretrained-blueprint "$RUN/best_checkpoint.pt"`
and `--freeze-blueprint-epochs N`. The first N diffusion epochs update only
the diffusion module; later epochs update both models. Its `--epochs` count is
the total for the diffusion run, so a 20-epoch frozen period followed by 10
joint epochs uses `--freeze-blueprint-epochs 20 --epochs 30`. Do not pass
`--pretrained-blueprint` when resuming that run; use `--resume` with the same
freeze count. The blueprint, radar variant, height filter, and geometry are
checked before initialization.

Stage-1 quality does not prove the model uses the right radar evidence. For a
counterfactual check, replace radar with another frame while holding faulty
LiDAR fixed and inspect changes to supported predictions on the missing rays.
Also compare radar-only blueprint metrics on all training and validation
frames using the same tile selection and geometry before interpreting the
diffusion result. Old stage-1 checkpoints are incompatible with this version:
the student has no observed-LiDAR encoder and uses three radar candidates.
