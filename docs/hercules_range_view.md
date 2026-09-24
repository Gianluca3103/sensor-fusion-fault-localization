# HeRCULES full-scan range-view dataset

`scripts.create_hercules_range_view_dataset` builds a **new**, separate data
root for the deterministic range-view reconstruction baseline. It does not
create BEV images, a 3D voxel supervision cache, fault-selector boxes, or a
geometric LiDAR stack. The source is the central Aeva scan. It writes:

- `samples/{train,val,test}/*.npz`: full-scan faulty LiDAR, raw-clean row
  provenance IDs, and fault/source metadata;
- `radar/{train,val,test}/{frame_id}.npz`: causally accumulated Continental
  points already aligned into the central LiDAR frame, with fields
  `[x,y,z,RCS,compensated_radial_velocity]`;
- a per-split generation report and exact-count summary.

The range-view loader applies the forward `x >= 0` field of view consistently
to the faulty scan, clean target, and radar. Original full-scan artifacts are
retained so a future full-azimuth experiment can reuse them. Radar is an
adaptive stack **capped at 20 scans** by default: time, translation, rotation,
and synchronization gates can select fewer; the actual alignment rows are
recorded per frame. The default newest-radar age limit is 50 ms.

The default split policy is the repository's existing 70/15/15 chronological
division within each HeRCULES session. The script randomly selects distinct
candidate frames from each split and backfills unsupported synchronization
until it has exactly 7,000 training, 1,500 validation, and 1,500 test
artifacts. If the source split lacks enough synchronized frames, generation
fails rather than silently returning fewer. An existing scene-held-out split
manifest may be supplied with `--split-manifest`; first use `--plan-only` to
verify it has enough frames in every split. No source frame is reused across
the three sets. The first `max-history-s` window of every validation/test
session is excluded so causal radar history cannot reach the preceding split.

HeRCULES Aeva's fourth point field is radial velocity, **not reflectivity**.
The default fault plan therefore uses geometry/dropout faults only:
`fov_filter:1`, `fov_filter:2`, `fov_filter:3`, and `total_loss:1`.
Legacy fog/laser/weather faults use field four as intensity. They are blocked
unless explicitly opted in with `--allow-velocity-as-reflectivity`, and should
not be presented as physically valid HeRCULES optical corruptions.

On the Ubuntu machine, from the repository root:

```bash
PYTHON=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
RAW=/mnt/3D10B36523559581/HeRCULES
OUT=/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs
DATA="$OUT/hercules_range_view_10k"
RADAR="$OUT/hercules_range_view_radar_10k"

"$PYTHON" -m scripts.create_hercules_range_view_dataset \
  --hercules-root "$RAW" --output-root "$DATA" \
  --radar-cache-root "$RADAR" --plan-only

"$PYTHON" -u -m scripts.create_hercules_range_view_dataset \
  --hercules-root "$RAW" --output-root "$DATA" \
  --radar-cache-root "$RADAR" \
  --train-count 7000 --val-count 1500 --test-count 1500 \
  --radar-frames 20 --max-radar-age-ms 50 --workers 2
```

The command is resumable. It validates each existing artifact's source and
generator signature, then skips compatible artifacts. Check
`$DATA/range_view_generation_summary.json` and the file counts after it
finishes. Use **fresh output roots** rather than mixing with legacy BEV
artifacts. A different radar policy or fault plan changes the signature and
will regenerate overlapping filenames; use another root for a comparative
experiment.

For a quick end-to-end smoke test, use separate temporary output roots and
`--train-count 7 --val-count 2 --test-count 1`. The default remains 10,000.

These artifacts feed `scripts.train_range_view_reconstruction` directly with
`--data-root "$DATA" --radar-root "$RADAR"`. Training still requires a
verified HeRCULES LiDAR beam-geometry JSON; the raw 29-byte Aeva records read
by this repository have no ring index or calibrated beam table. Do not use
guessed equally spaced rows for a thesis run. The generation step itself does
not need that geometry.
