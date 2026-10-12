# Sparse 3D diffusion Stage II

This is a separate Stage II model. The existing deterministic radar-only and
faulty-conditioned checkpoints remain incompatible by design.

## Inputs and output

At inference, frozen Stage I encodes the aligned 20-frame radar stack and
proposes radar-supported 3D regions. Stage II reads those features and the
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

Learned Stage-I regions are rasterized without a candidate cap. The rasterizer
reduces overlapping cells in NumPy batches; each candidate keeps the strongest
proposal score. CPU nearest-neighbor queries use up to four threads for large
query sets. Set `STAGE2_KDTREE_WORKERS=1` to compare against serial queries.
During reverse diffusion, all timesteps reuse the same MinkowskiEngine
coordinate manager and sparse output ordering. The conditioner projects each
Stage-I source feature before pooling it into candidate cells. These changes
preserve the candidate set and network parameter shapes.

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
