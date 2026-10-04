# Ray–depth radar/LiDAR cross-attention prototype

This is the new reconstruction **blueprint** module. It does not reuse the old
range-view reconstruction network, export virtual points, or run SVEFusion.
Its intended input is the verified, ego-motion-aligned VoD 20-scan radar stack
in the current LiDAR frame (7 fields: XYZ, RCS, radial Doppler, compensated
Doppler, discrete scan ordinal), plus the surviving faulty LiDAR (XYZ,
reflectivity). The scan ordinal is **not** elapsed time in seconds. Moving
targets can remain misaligned after ego-motion compensation; the model sees
scan age and Doppler but does not solve full scene flow.

## Architecture

1. Calibrated ray directions come from `RangeGeometry`. Radar and surviving
   LiDAR propose distinct candidate depths; two uniform depths keep a query
   available under dropout. Uniform depths are not sensor evidence. Clean
   LiDAR never proposes a depth.
2. Each sensor encoder supplies fine point tokens with local 3D grid context
   and occupied coarse voxel tokens. Radar point features include RCS,
   Doppler, and scan ordinal. These form two scales of local pattern evidence.
3. At each ray–depth reference position, two cross-attention branches retrieve
   local radar and observed-LiDAR tokens. Search uses along-ray and lateral
   separation; the radar branch learns bounded, direction-dependent
   uncertainty. Relative XYZ, along-ray offset, perpendicular distance and
   radar attributes also enter attention logits.
4. Per modality, a scale gate mixes fine and coarse evidence. A three-way gate
   then selects radar, observed LiDAR, or an explicit null feature. If a
   candidate has neither source within the local window, its return logit is
   masked; the model must abstain there.
5. Local query self-attention exchanges information across neighboring scan
   rays and candidate depths (within 4 m). It cannot bridge arbitrary depth
   discontinuities. Two fusion blocks are stacked.
6. The output contains a feature and evidence weights per candidate, bounded
   depth corrections, and first-return logits that include a no-return class.
   A later conditional reconstruction head can consume this blueprint.

The prototype defaults to width 32, 4 heads, 16 local neighbors, and 2048
candidates per call. These are starting values, **not** established VoD
hyperparameters. The `first_return_probabilities` property is uncalibrated;
confidence calibration requires a held-out split after training.

## Training and inference interface

```python
from models.two_stage_reconstruction_head.cross_modal_encoders import EncoderGrid
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.ray_depth_attention import RayDepthBlueprintModel
from models.two_stage_reconstruction_head.ray_depth_queries import ray_tile_indices
from models.two_stage_reconstruction_head.ray_depth_training import ray_depth_blueprint_loss

geometry = RangeGeometry.from_json("/path/to/calibrated_geometry.json")
model = RayDepthBlueprintModel(geometry, EncoderGrid(), width=32).cuda()
rows, cols = ray_tile_indices(geometry, row_start=0, row_stop=1,
                              col_start=0, col_stop=256, device="cuda")
# radar [B,N,7], faulty_lidar [B,M,4], masks are boolean; all in LiDAR frame.
model.train()
blueprint = model(radar, radar_valid, faulty_lidar, faulty_valid,
                  rows, cols, clean_lidar=clean, clean_valid=clean_valid)
losses = ray_depth_blueprint_loss(blueprint, geometry, clean, clean_valid)
losses["loss"].backward()

model.eval()
with torch.no_grad():
    blueprint = model(radar, radar_valid, faulty_lidar, faulty_valid,
                      rows, cols)
    depth_m, return_mask = blueprint.predicted_first_return()
```

Use calibrated beam elevations and azimuth sampling from the real LiDAR. With
the default five depth slots, a 256-ray tile uses 1280 candidates. Adjacent
tiles should overlap by at least one beam row and two azimuth columns so local
query exchange has context at boundaries; discard duplicate outputs after
stitching. The prototype currently re-encodes the full sensor inputs for each
tile. A scan-level inference wrapper should cache the encodings before large
scale use. Full-scan throughput and GPU memory have **not** been measured.

`ray_depth_blueprint_loss` derives first-return and negative-ray targets from
the **clean** scan, trains the no-return class, and regresses depth for
evidence-supported candidates within 3 m of a clean return. Its `coverage`
metric reports the fraction of clean positive rays with such a candidate.
Clean returns without support are trained to abstain. This makes coverage an
important diagnostic: a poor proposal set cannot be repaired by attention.
The optional teacher feature term (`teacher_weight > 0`) compares the selected
query feature with a detached local clean-encoder feature. Keep it disabled
until the clean encoder has been pretrained or stabilized with an EMA teacher;
otherwise a random feature target has no geometric meaning.

## Validation before claiming improvement

- Train this module on training frames only. Tune hyperparameters on validation
  frames; preserve a separate held-out evaluation split.
- Compare radar-only, observed-LiDAR-only, and both branches with identical
  candidate and compute budgets. Also compare radar-return queries against
  ray–depth queries if testing the query-design claim.
- Hold faulty LiDAR fixed and replace radar with another frame. The blueprint
  should change near the original radar support and abstain where evidence is
  removed. The unit tests check this on synthetic geometry; real VoD results
  remain to be measured.
- Measure clean-ray target coverage, supported-return precision, depth error,
  and downstream detection on exactly matched frames. Attention maps alone
  do not establish radar-conditioned reconstruction.

The code here supplies a trainable cross-attention blueprint and loss. A new
radar-gated range-view diffusion path is described in
`docs/radar_gated_range_diffusion.md`. Neither stage has a trained checkpoint
or measured detection improvement yet. The decoder's evidence dependence and
clean-depth coverage still need real-VoD validation.
