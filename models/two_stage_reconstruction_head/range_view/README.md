# Sensor-native range-view reconstruction

This is a separate deterministic baseline. The existing sparse-voxel/BEV
training path remains available and unchanged. The new path has no
reconstruction bounding box or hard fault mask: every calibrated LiDAR ray in
the selected sensor field can receive an ADD proposal. By default only the
forward LiDAR/radar half-space (`x >= 0` in the LiDAR frame) is used, since
the current radar coverage is forward-facing. `--include-rear` restores full
azimuth for an experiment. This front-FOV selection is applied consistently
to faulty LiDAR, clean targets, radar, and generated returns; it is not a
fault-selector mask. Original LiDAR points within the selected FOV are copied into the output by
default. The optional DELETE head can remove an original only above a separate,
high threshold (default `0.999`), and never resolves colliding original points
from one ray-level score. Original and generated points retain separate
provenance; an ADD on an occupied ray does not overwrite its original.

## Required sensor geometry

The repository's four-column VoD/HeRCULES point files do not provide a ring ID,
and the available calibration records do not contain a verified beam elevation
table. **Do not train with guessed or evenly spaced beam angles.** Obtain the
sensor's actual beam/scanline elevations, angular sampling, field of view, and
range bounds, then supply a JSON file:

```json
{
  "beam_elevations_rad": ["replace with measured, ascending numeric elevations"],
  "azimuth_bins": "replace with measured integer column count",
  "azimuth_span_rad": "replace with measured span in radians",
  "azimuth_offset_rad": "replace with calibrated start angle in radians",
  "min_range_m": "replace with physical minimum range in metres",
  "max_range_m": "replace with physical maximum range in metres",
  "max_beam_error_rad": "optional calibrated beam-assignment tolerance"
}
```

The strings above are placeholders, not a usable configuration. For a full
revolution use the measured 2π span. The geometry loader rejects absent beam
elevations. Audit it against full raw scans before training:

```bash
python -m scripts.audit_range_view_geometry \
  --geometry /path/to/measured_sensor_geometry.json \
  --artifact /path/to/full_scan_faults/train/example.npz
```

Review projected coverage and XYZ round-trip error. Discretization means the
back-projected point will generally not equal its input XYZ exactly. For radar,
the existing PointPillars cache is already aligned into the LiDAR frame; do
not transform it again. If starting from radar-frame points, use
`transform_radar_to_lidar` with a calibrated 4×4 transform. Final output stays
in the LiDAR frame; `transform_points` can transform the merged cloud with an
existing LiDAR-to-vehicle/world pose when required.

## Data migration and training

Legacy reconstruction artifacts crop the faulty scan to a BEV region, even
though the raw clean scan is complete. They cannot be used for global
range-view repair. Regenerate faults into a **new** root with the full raw scan:

```bash
python -m scripts.create_range_view_fault_samples \
  --source-root /path/to/legacy_reconstruction_samples \
  --output-root /path/to/full_scan_faults \
  --split train
python -m scripts.create_range_view_fault_samples \
  --source-root /path/to/legacy_reconstruction_samples \
  --output-root /path/to/full_scan_faults \
  --split val
```

The generator reuses the recorded fault type, severity, and injection seed,
and saves full-scan `faulty_lidar_points` plus `faulty_source_ids`. It fails
explicitly if provenance or regeneration metadata is unavailable. Verify
the new artifact count and geometry audit before starting a full run.

For HeRCULES, use `scripts.create_hercules_range_view_dataset` instead of the
legacy BEV generator or the migration script. It creates exactly the requested
train/val/test counts and fresh aligned radar caches; see
[`docs/hercules_range_view.md`](../../../docs/hercules_range_view.md).

```bash
python -m scripts.train_range_view_reconstruction \
  --data-root /path/to/full_scan_faults \
  --radar-root /path/to/aligned_radar_cache \
  --geometry /path/to/measured_sensor_geometry.json \
  --output-root /path/to/range_view_run \
  --epochs 10 --batch-size 4 --device cuda
```

Defaults are append-only, LiDAR+radar, and no fault-map conditioning. For
ablation, add `--allow-original-deletion --delete-threshold 0.999`,
`--no-radar`, or `--use-fault-map-conditioning --fault-map-root ...`.
Fault-map inputs must come from an independent predictor and be saved under
the same split/name as the sample with key
`fault_probability_range_view`. The legacy target-derived heatmap and
reliability map are deliberately excluded to prevent label leakage.

Loss terms are separately logged: ADD classification, ADD-only robust range,
asymmetric DELETE classification, optional same-ray geometry, and clean-ray
free-space/no-return penalty. The false-delete penalty defaults to 30 versus
1 for missed deletion. Training validates the **merged XYZ point cloud**, not
just the range image, and saves fault-wise metrics and range/XYZ visualizations.

## Evaluation

```bash
python -m scripts.evaluate_range_view_reconstruction \
  --checkpoint /path/to/range_view_run/last_checkpoint.pt \
  --data-root /path/to/full_scan_faults \
  --radar-root /path/to/aligned_radar_cache \
  --output-root /path/to/range_view_eval \
  --split val --device cuda \
  --delete-thresholds 0.95 0.99 0.995 0.999
```

This evaluates append-only and conservative-deletion settings, with XYZ
precision/recall/F1/IoU, Chamfer, addition/hallucination, healthy-original
preservation, false deletion, corrupted-point rejection, net improvement, and
per-fault KEEP/ADD/DELETE/REPLACE target counts. `--disable-radar` and
`--disable-fault-map` are inference ablations; training independent models
with/without a modality is needed for a controlled quality comparison.

The first deterministic model is intentionally small: a 2D range-view network
that downsamples only azimuth and uses circular horizontal padding for a full
revolution. Diffusion is not required for this baseline.
