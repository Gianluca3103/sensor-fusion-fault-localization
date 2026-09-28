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

The fault-selector policy matches the HeRCULES range-view baseline: **there is
no selector or oracle fault box**. The model receives the whole forward sensor
view, with fault-map conditioning disabled. The merge is conservative by
default: it may append generated points but cannot delete any existing
forward LiDAR points, regardless of its DELETE-head score. Rear points are
outside this experiment's `x >= 0` field of view, as in the HeRCULES setup.
The launcher audits the saved config and validation output for this policy.

This runs the existing deterministic range-view ADD/range/DELETE model, **not**
the Cartesian sparse-voxel diffusion model. The default batch size is one and
hidden width is eight for a laptop GPU. The checkpoint and `summary.csv` in
each timestamped run directory are the main training outputs.

For the Ubuntu full dataset, `scripts/run_vod_range5_full_cache.sh` builds a
separate `radar_5frames_rangeview` raw radar variant with history isolated by
official train/val/test split, then generates full-scan fault samples and lean
LiDAR-aligned radar caches for every available frame. The stack is capped at
five scans; the first frames of a recording may contain fewer. Unlike the
local strict-five smoke test, this preserves those warm-up frames. Generation
is resumable and writes per-split summaries under `samples/`.
