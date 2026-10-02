# Local VoD five-radar range-view test

Run from PowerShell:

```powershell
cd "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"
& .\scripts\run_vod_range5_test.ps1 -TrainSamples 32 -ValSamples 8 -Epochs 3
```

The launcher uses the official VoD train/val split and the existing
`radar5_pointpillars_cache`. It selects frames only when the raw stacked radar
contains all five scan time indices, then injects a repeating mix of fog,
FOV-filter, and total-loss faults into the full, uncropped central LiDAR scan.
The old BEV reconstruction artifacts are not used. Generated samples,
geometry, metrics, visualizations, and checkpoints remain under
`outputs_local/vod_range5_test/`.

The model input is a forward-facing azimuth/elevation range image: 128
elevation bins by 512 azimuth bins, with LiDAR range/validity/reflectivity and
projected five-frame radar features. Its output is converted back to 3D XYZ
points. The elevation rows are uniform angular bins over -25 to +10 degrees,
**not calibrated physical LiDAR beam IDs**; VoD's four-float point files do
not contain ring IDs. The geometry script audits the chosen bounds on training
LiDAR and refuses to proceed if they retain under 99% of forward points.

## Audited virtual-ray replacement for a new training run

The original 128 x 512 geometry is retained for reproducibility. For a new
run, `scripts.fit_vod_lidar_rays` fits 128 nonuniform elevation rows from
**clean training IDs only**, with 2048 front azimuth bins. These are virtual
angular rays, not claimed factory laser IDs: VoD provides motion-compensated
XYZI without a ring ID or per-point firing time. The fitter removes exact
duplicate XYZI rows for geometry statistics, but does not change the training
or detector data. It writes a held-out-train audit of point assignment,
same-cell collisions, and XYZ round-trip error, and refuses to save a geometry
that fails the configured thresholds.

On the local VoD copy, 128 fit and 32 held-out training frames produced 99.94%
assignment, 1.75% collisions among unique points, and 0.040 m 95th-percentile
XYZ round-trip error. The old 128 x 512 grid has about 43% collisions on the
same held-out frames. The local files also contain two exact copies of every
XYZI point tested; the fitter reports this separately instead of counting
duplicates as ray collisions. Check the professor machine's audit before
assuming its dataset has the same duplication.

```bash
python -m scripts.fit_vod_lidar_rays \
  --vod-public /path/to/view_of_delft_PUBLIC \
  --output /path/to/vod_virtual_128x2048_geometry.json
```

Train from scratch with the new geometry and `--use-ray-encoding`. The optional
three unit-vector channels tell the network the direction of each nonuniform
virtual ray. Older checkpoints keep their original ten-channel input and
geometry. This representation change does **not** by itself enforce
free-space or surface continuity, nor can it determine hidden ranges from a
total LiDAR loss; evaluate those separately before claiming physical realism.

### Cache projected training inputs before an 80-epoch run

The full-scan artifacts remain the source of truth. After fitting the new
geometry, build a separate disk cache of the 10 base feature maps and seven
training targets. This avoids loading the clean `.bin`, aligned radar `.npz`,
and recomputing angular projections on every training epoch. The optional
three ray-direction channels are generated from the geometry at load time.
The cache is specific to the geometry, source files, forward-view selection,
radar filter, and intensity setting. The trainer rejects a stale or incomplete
cache. The builder is resumable and stores only `train` samples; final-epoch
validation still reads original samples.

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/svefusion-clean-cu118/bin/python"
CACHE="$BASE/sensor_fusion_outputs/vod_range5_full_cache"
GEOM="$BASE/sensor_fusion_outputs/vod_virtual_128x2048_geometry.json"
INPUT_CACHE="$BASE/sensor_fusion_outputs/vod_virtual_128x2048_train_inputs"
cd "$REPO"
"$PY" -u -m scripts.cache_range_view_inputs \
  --data-root "$CACHE/samples" --radar-root "$CACHE/radar" \
  --geometry "$GEOM" --output-root "$INPUT_CACHE" \
  --require-lidar-intensity --workers 4
```

Resume the same command after an interruption; it reuses valid entries.
Pass `--input-cache-root "$INPUT_CACHE"` to the 80-epoch trainer, with
`--predict-intensity --use-ray-encoding` and the same geometry. Do not enable
`--online-yaw-deg` or external fault-map conditioning with this fixed cache.
Keep enough free disk space for the compressed cache; the builder prints the
completed sample count and writes `manifest.json` only after all entries exist.

The fault-selector policy matches the HeRCULES range-view baseline: **there is
no selector or oracle fault box**. The model receives the whole forward sensor
view, with fault-map conditioning disabled. The merge is conservative by
default: it may append generated points but cannot delete any existing
forward LiDAR points, regardless of its DELETE-head score. Rear points are
outside this experiment's `x >= 0` field of view, as in the HeRCULES setup.
The launcher audits the saved config and validation output for this policy.

After aligning radar into the LiDAR frame and applying the same forward-view
selection, the model drops radar returns below the minimum `z` of the
available **faulty LiDAR input**. The clean target is never used for this
filter. If a total-loss fault leaves no LiDAR points, radar is left unchanged.
This happens when samples are loaded, so existing radar caches need not be
rebuilt. New checkpoints record this preprocessing rule; older checkpoints
continue to evaluate with their original radar preprocessing.

This runs the existing deterministic range-view ADD/range/DELETE model, **not**
the Cartesian sparse-voxel diffusion model. The default batch size is one and
hidden width is eight for a laptop GPU. The checkpoint and `summary.csv` in
each timestamped run directory are the main training outputs.

To inspect a checkpoint without occupying the training GPU, run
`scripts.visualize_range_view_reconstruction` with `--device cpu`, the run's
`last_checkpoint.pt`, and the same sample/radar cache roots used by training.
By default it opens the first three validation samples in a mouse-rotatable
Matplotlib 3D viewer with synchronized camera angles when a GUI backend is
available. It also saves a self-contained `_interactive.html` browser viewer
that works offline without Tk or Qt, a fixed-scale XY/XZ/YZ comparison PNG,
a 3D PNG, full XYZ PLY clouds, and JSON with true point counts. `--no-show`
exports without GUI windows. Browser and PNG views subsample points for
responsiveness; PLY files retain the full clouds. The trainer retains
`last_checkpoint.pt` and a separate `checkpoint_epoch_N.pt` at each validation
epoch, so an earlier model can be re-inferred after training continues.
On phones, the HTML viewer shows one condition at a time with tabs while
keeping the same camera angle and scale. One finger rotates; two fingers pan
and pinch to zoom. The `+` and `−` buttons also zoom, and the radar checkbox
overlays the aligned radar returns used by the model in amber.

For a train-only online geometric augmentation ablation, pass
`--online-yaw-deg 5` to `scripts.train_range_view_reconstruction`. Each training
sample gets a fresh yaw drawn uniformly from -5 to +5 degrees on every load.
The same rotation is applied to faulty LiDAR, clean LiDAR supervision, and
aligned radar before the forward-view crop and range projection. Validation
and test data stay unrotated. No cache regeneration is needed; use a separate
output directory and keep all other training settings identical to the
unaugmented baseline. Fault-map conditioning cannot be combined with this
augmentation unless its predicted map is rotated consistently too.

For an opt-in radar minimum-height ablation, pass
`--radar-floor-band-m 0.1` to `scripts.train_range_view_reconstruction` and
use a fresh output directory. After the forward-FOV crop, each sample uses
its minimum radar `z` as the floor proxy and drops returns with
`z <= min_z + 0.1 m`. LiDAR and supervision are unchanged; the radar cache
is not rewritten. The band is saved in the checkpoint and reused by evaluation
and reconstruction visualization. Validation JSON records removed radar
points per sample. Use 0 to disable only this extra band (the minimum-LiDAR
filter remains active). This is **not** a
slope-aware ground model: one low outlier can cause it to remove almost nothing.

For the Ubuntu full dataset, `scripts/run_vod_range5_full_cache.sh` builds a
separate `radar_5frames_rangeview` raw radar variant with history isolated by
official train/val/test split, then generates full-scan fault samples and lean
LiDAR-aligned radar caches for every available frame. The stack is capped at
five scans; the first frames of a recording may contain fewer. Unlike the
local strict-five smoke test, this preserves those warm-up frames. Generation
is resumable and writes per-split summaries under `samples/`.

For VoD detector experiments, add `--predict-intensity --lambda-intensity 0.1`
to a new `scripts.train_range_view_reconstruction` run. The optional head
predicts LiDAR reflectivity for ADD points. Its Smooth L1 loss compares
`log1p` intensity only on clean, valid ADD rays, keeping the scale manageable
without assuming a particular raw intensity range. Retained original points
keep their measured fourth channel; generated points receive the model's
predicted fourth channel in evaluation, visualization, and detector export.
HeRCULES' fourth LiDAR channel is radial velocity and is rejected as an
intensity training target. Checkpoints without the optional head continue to
load and produce the earlier zero-intensity generated points.
