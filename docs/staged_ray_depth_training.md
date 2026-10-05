# Staged radar-to-LiDAR ray-depth training

Train the ray-depth blueprint before diffusion so its 3D radar/LiDAR
correspondence can be assessed directly. The encoder and cross-attention see
the Doppler-aligned 20-scan radar stack and surviving faulty LiDAR. Clean LiDAR
is used only after the forward pass for first-return/no-return and metric-depth
targets. The optional clean feature teacher is **not** active in this run.

## Stage 1: blueprint only

These are starting hyperparameters, not a VoD optimum. Keep batch size one:
each scene needs its own tile selected around radar evidence on currently
unobserved LiDAR rays. Four gradient accumulation steps combine four
independently selected scene tiles in each optimizer update. The `4 × 64` tile
makes 1,280 ray-depth candidates, below the cross-attention limit of 2,048.
The chosen tile never uses the clean scan to pick a position.

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
  --epochs 10 --batch-size 1 --grad-accum-steps 4 \
  --tile-rows 4 --tile-cols 64 --width 32 \
  --learning-rate 0.0002 --validate-every 5 \
  --num-workers 4 --device cuda
```

Start with 10 epochs and inspect validation after epochs 5 and 10. Continue
to 30 or 50 only while validation improves. To continue the same run, use the
same command with `--epochs 30` or `--epochs 50` and
`--resume "$RUN/last_checkpoint.pt"`; keep all other settings the same.
Gradient accumulation changes update frequency, not the number of scenes
processed per epoch. A single `4 × 64` training tile took about 5.4 seconds
on a local RTX 5060 Laptop GPU in a one-frame smoke test; measure several
real frames on the training machine before committing to a long run.

The selected radar variant and observed-faulty-LiDAR height filter are on by
default. Before a long run, add `--train-limit 2 --val-limit 2 --epochs 1
--validate-every 1` and a separate output root for a short data check. This
smoke run does not estimate blueprint quality.

Each validation prints loss, candidate coverage, and depth-aware precision,
recall, and F1 at a 3 m error tolerance. Those scores use only radar-supported
rays missing from observed faulty LiDAR. Check `clean_hits` and
`radar_supported_missing_rays` before interpreting them: if either is zero,
the sampled validation tiles did not test actual reconstruction. `best_checkpoint.pt`
is selected by validation depth F1; `last_checkpoint.pt` is saved each epoch.
Candidate coverage measures whether a clean return has a nearby proposal, not
whether the model selected the right return. The generated cloud and detector
AP must be evaluated separately after later stages.

New runs print a short train summary after each epoch and a second validation
line when validation runs. All metrics remain available as JSON records in
`epoch_metrics.jsonl` under each run's output directory. A Python process
already running when this output change is installed keeps its earlier format.

## Later stages

The diffusion trainer accepts `--pretrained-blueprint "$RUN/best_checkpoint.pt"`
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
