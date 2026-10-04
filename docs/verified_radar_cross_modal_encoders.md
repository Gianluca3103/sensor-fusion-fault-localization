# Verified VoD radar stacking and cross-modal encoders

## Radar audit

The released VoD single-scan radar records contain seven float32 fields:
`x, y, z, RCS, radial velocity, compensated radial velocity, relative scan index`.
The released five-scan stack preserves each raw scan's point order and the
measured RCS/velocity fields. The previous implementation used per-frame
`odomToCamera` poses for longer stacks. On the local VoD copy, its alignment
can differ from the released five-scan coordinates by over 10 cm. Its 5 m
step gate also dropped valid history at frame `01313`.

`--alignment-source official-5` recovers each adjacent rigid transform from
corresponding rows of the released five-scan stack. It checks that the raw
RCS/velocity rows match and that the rigid-fit residual is below 1 mm, then
composes those transforms for longer stacks. Released five-scan scene starts
define history boundaries. If an earlier raw scan is missing (for example
around local frame `04049`), the verified stack starts a new history rather
than inventing a scan.

No filtering is enabled by default, so all finite measured returns are kept.
The old validity and temporal filters remain explicit options for experiments.
Generated verified files go in `radar_20frames_verified`, leaving older stacks
untouched. The generator rejects an existing stack when its manifest reports
different settings or no manifest exists.

Generate all three official splits on the professor machine after updating the
repository:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/svefusion-clean-cu118/bin/python"
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
cd "$REPO"
for split in train val test; do
  "$PY" -u -m scripts.generate_vod_accumulated_radar \
    --vod-root "$VOD" --stack-sizes 20 --split "$split" \
    --alignment-source official-5 --output-suffix verified \
    --isolate-split-history --no-basic-validity-filter \
    --num-workers 4
done
```

The output's seventh field is an ordinal scan age (`-19` through `0` for a
full stack), not elapsed seconds. Ego-motion alignment does not compensate
independently moving objects; the model must use the preserved time and
Doppler fields to reason about them. A stack is evidence from multiple times,
not a single physically instantaneous radar scan.

The ray diffusion training path now uses the **Doppler radial-shifted** verified
stack by default and applies the observed faulty-LiDAR height gate after
calibration. The unshifted verified stack remains available for a controlled
ablation with `--radar-variant radar_20frames_verified`.
`docs/vod_motion_aware_radar.md` describes an optional velocity-age filtering
ablation; its deletion of older moving returns can also remove useful cyclist
evidence.
`docs/vod_doppler_radar_comparison.md` describes the Doppler-shifted stack,
full-validation alignment comparison, and the height-gate audit.

## Encoder interface

`cross_modal_data.py` pairs existing full-scan faulty VoD artifacts with the
verified raw radar files. It aligns radar XYZ to the current LiDAR frame and
keeps all seven radar fields. It pads variable length point sets and returns
boolean masks for a PyTorch `DataLoader`.

`cross_modal_encoders.py` provides:

- A radar encoder that embeds XYZ, RCS, both Doppler values, and scan age.
  Current and historical returns are pooled separately before 3D spatial
  context, reducing the tendency to average a moving object across time.
- Separate encoders for faulty observed LiDAR and clean training LiDAR. Their
  features share a configurable 3D grid, so nearby radar and LiDAR regions
  can be related without claiming that individual returns coincide.
- Fine point embeddings and XYZ alongside coarse region tokens. Later local
  cross-attention can query both rather than relying on a 2 m grid for object
  shape.
- A combined module that accepts clean LiDAR only while training. Inference
  uses radar and faulty LiDAR features alone.

Example training-side input path:

```python
from pathlib import Path
from torch.utils.data import DataLoader
from models.two_stage_reconstruction_head.cross_modal_data import (
    CrossModalVoDDataset, collate_cross_modal,
)
from models.two_stage_reconstruction_head.cross_modal_encoders import RadarLidarEncoders

base = Path("/mnt/3D10B36523559581/Gianluca")
cache = base / "sensor_fusion_outputs" / "vod_range5_full_cache"
vod = Path("/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC")
paths = sorted((cache / "samples" / "train").glob("*.npz"))
dataset = CrossModalVoDDataset(paths, vod, include_clean=True,
                               radar_height_filter=True)
batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_cross_modal)))
encoders = RadarLidarEncoders(history_scans=20).train()
encoded = encoders(
    batch["radar"], batch["radar_valid"],
    batch["observed_lidar"], batch["observed_lidar_valid"],
    clean_lidar=batch["clean_lidar"], clean_valid=batch["clean_lidar_valid"],
)
```

These are new, untrained encoders. The ray-depth cross-attention blueprint and
radar-gated range-view diffusion decoder now consume their features; see
`docs/ray_depth_cross_attention.md` and
`docs/radar_gated_range_diffusion.md`. Existing range-view checkpoints do not
contain weights for this architecture. Clean LiDAR is reserved for training
targets or held-out scoring and never forms inference conditioning features.
