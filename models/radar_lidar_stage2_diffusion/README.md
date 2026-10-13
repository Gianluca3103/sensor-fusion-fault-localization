# Sparse 3D diffusion Stage II

This is a separate Stage II model. The existing deterministic radar-only and
faulty-conditioned checkpoints remain incompatible by design.

## Inputs and output

At inference, frozen Stage I encodes the aligned 20-frame radar stack and
proposes radar-supported surface locations. Edge-aware interpolation connects
only nearby proposals with compatible learned radar features and scores, then
keeps fine cells above the configured confidence threshold. The old ellipsoid
radii are ignored by new Stage-II runs. Stage II reads those features and the
surviving faulty LiDAR. A fixed candidate grid at the Stage-I fine voxel size
is used throughout diffusion. The state has four values per candidate: one
addition/abstention value and three within-voxel XYZ offsets. One occupied
candidate decodes to one point. The sparse U-Net receives the noisy state,
radar/faulty features, and a timestep embedding. A cosine schedule starts
sampling from Gaussian noise and uses deterministic DDIM reverse steps.

Training clean LiDAR creates positive addition and within-voxel offset targets.
Measured faulty rays and clean-measured free space provide negative addition
targets. Unobserved space is **not** treated as physically free. The loss
balances positive, observed, and visible-free groups, then regresses offsets
only for true additions. Unobserved cells receive random training latents and
no loss so a zero-filled clean visibility mask cannot become a shortcut. There
is no clean-LiDAR input to `prepare` or `sample`.
Observed faulty returns are preserved in the detector export and the model
cannot export duplicate points on measured rays or outside the fault region.

Stage I is frozen. Diffusion is trained from scratch; a deterministic Stage-II
checkpoint cannot be loaded into this architecture. `best_global_missing.ckpt`
is selected by sampled validation F1 at 20 cm on all missing clean points,
including those outside the candidate regions. Validation and export use the
same fixed per-frame sampling seed.

## Performance

Edge-aware candidate sites are built without an ellipsoid or candidate cap.
The diffusion configuration rejects any numerical candidate
cap, and the diffusion viewer displays every generated addition within its
selected spatial crop. Its PLY export contains all additions. CPU nearest-neighbor
queries use up to four threads for large query sets. Set
`STAGE2_KDTREE_WORKERS=1` to compare against serial queries.
During reverse diffusion, all timesteps reuse the same MinkowskiEngine
coordinate manager and sparse output ordering. The conditioner projects each
pooled Stage-I feature once per candidate cell. These changes
preserve the candidate set and network parameter shapes.

An existing 16-timestep checkpoint can also be sampled with three DDIM passes.
The passes use trained timesteps 15, 8, and 0, with a direct reverse update
between them. This changes sampling only; training still draws one timestep per
batch. Compare both settings on identical validation frames before relying on
three-step geometry:

```bash
python -m scripts.evaluate_stage2_diffusion_steps \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --checkpoint "$RUN/last.ckpt" --val-limit 16 \
  --sampling-steps 3 16 --output "$RUN/sampling_3_vs_16.json" \
  --device cuda
```

The detector export accepts `--sampling-steps 3`. Use a distinct output root
for each sampling setting; the export manifest records the choice.

## Frozen-preprocessing cache

`scripts.cache_stage2_diffusion` stores a versioned file per VoD frame for both
train and validation. It includes Stage-I source features, edge-aware fine
candidate coordinates, faulty-LiDAR coverage, four physical neighbor maps,
and clean training targets. The cache builder uses clean LiDAR only for labels.
At training time the conditioning projection, faulty encoder, fusion layers,
and sparse U-Net remain trainable and are recalculated every batch. Detector
inference still uses the uncached radar/faulty-only path.

The cache manifest binds the Stage-I checkpoint hash, radar variant, fault
sample location and pattern, and Stage-II geometry config. An interrupted build
can resume at the same root. After 16 new frames it reports an estimated disk
requirement and stops if available space appears insufficient. Neighbor maps
can make this cache large; compression is enabled by default. Check the pilot
size report before leaving the full build unattended.

Changing from ellipsoids to edge-aware support changes the candidate domain.
Build a new cache root and train Stage II from epoch 1. Previous Stage-II
checkpoints retain their original ellipsoid setting when loaded for comparison.

```bash
python -m scripts.cache_stage2_diffusion \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --stage1-checkpoint "$STAGE1" \
  --output-root "$BASE/sensor_fusion_outputs/stage2_preprocess_cache" \
  --config configs/radar_lidar_stage2_diffusion.json --device cuda

python -m models.radar_lidar_stage2_diffusion.train \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --stage1-checkpoint "$STAGE1" --output-root "$RUN" \
  --preprocess-cache-root "$BASE/sensor_fusion_outputs/stage2_preprocess_cache" \
  --config configs/radar_lidar_stage2_diffusion.json \
  --epochs 50 --batch-size 4 --grad-accum-steps 1 \
  --validate-every 5 --num-workers 2 --device cuda --amp bf16
```

The cached loader can resume an existing Stage-II checkpoint because it does
not change the architecture, trainable parameters, or target definitions.

`--amp bf16` optionally enables bfloat16 autocast for the Stage-II U-Net on
CUDA; Stage I and target construction remain full precision. Run
`python -m scripts.smoke_stage2_diffusion_minkowski` on the target GPU before
using it. The export command automatically uses the precision recorded in a
new checkpoint. To measure the real frame preparation costs, run
`python -m scripts.profile_stage2_cpu_construction --help`.

## Professor-machine run

Run from the updated repository checkout. Use the Stage-I checkpoint from the
new learned-region run, not an older ray-depth blueprint:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
PY="$BASE/Sensor-Fusion/.venv_model_v2/bin/python"
FAULTS="$BASE/sensor_fusion_outputs/vod_staged_cache_20261005/samples"
STAGE1=/absolute/path/to/the/new/stage1/best_selected.ckpt
RUN="$BASE/sensor_fusion_outputs/stage2_sparse_diffusion_20frame"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8

"$PY" -m scripts.smoke_stage2_diffusion_minkowski
"$PY" -u -m models.radar_lidar_stage2_diffusion.train \
  --vod-root "$VOD" --fault-samples-root "$FAULTS" \
  --stage1-checkpoint "$STAGE1" --output-root "$RUN" \
  --config configs/radar_lidar_stage2_diffusion.json \
  --radar-variant radar_20frames_verified_doppler_radial \
  --epochs 50 --batch-size 1 --grad-accum-steps 4 \
  --validate-every 5 --num-workers 2 --device cuda
```

Before committing to the full run, add `--epochs 1 --train-limit 1
--val-limit 1 --validate-every 1 --num-workers 0` with a separate output root.
The full validation set requires one full reverse chain per frame and will be
much slower than deterministic Stage II. Resume with `--resume "$RUN/last.ckpt"`
and identical arguments.

Export with `python -m scripts.export_stage2_diffusion_vod_detector
--vod-root "$VOD" --checkpoint "$RUN/best_global_missing.ckpt"
--fault-samples-root "$FAULTS" --output-root /new/output/root --device cuda`.
Use `--limit 2` and a separate output root for a two-frame export check.
