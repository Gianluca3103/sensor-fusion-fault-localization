# Stage II: deterministic representation gate

Stage II will consume only the frozen, deployed Stage-I radar output. The
attached Stage-II design calls for a MinkowskiEngine sparse 3D U-Net, with
diffusion **disabled** for the first experiments. No dense, BEV, range-view,
faulty-LiDAR, semantic, or object-detection input belongs in this model.

The current repository change implements **Phases A–D only**. It does not
claim that the reconstruction U-Net runs: MinkowskiEngine is absent from the
available local Windows/WSL environments. Run
`python -m scripts.probe_minkowski_stage2 --output me_probe.json` in the
professor-machine training environment before implementing Phase E. The probe
checks sparse construction, stride, transpose, generative transpose, backward,
CUDA and mixed precision. Core CPU and CUDA FP32 checks must pass.

## Audited Stage-I interface

- `Stage1Output.features` contains `s1`–`s4` `SparseSites`. Each has sorted,
  unique integer `[batch,z,y,x]` coordinates, a feature matrix, physical tensor
  stride and a shape. The small Stage-I configuration has 16/32/64/128
  channels, but the actual checkpoint config must be loaded dynamically.
- `Stage1Output.confidence` is defined on exactly the S1 coordinates and
  contains one value in `[0,1]` per S1 radar-derived site. It is supervised by
  local probe quality and is **not a calibrated surface probability**.
- Stage-I grid spacing is configured in physical XYZ, usually
  `(0.2,0.2,0.25)` m. S1–S4 strides are 1/2/4/8. Coarse scale features cannot
  be connected to Stage-II levels merely because their names match.
- Batch is the first coordinate column. Sparse coordinates are in ZYX order;
  sensor points and grid origin/spacing are in XYZ order. The conversion is
  `floor((xyz - grid.minimum_xyz) / grid.size_xyz)`, then reorder to BZYX.
- `forward_radar` reads radar and its validity mask only. The clean teacher,
  correspondence heads and probe are training-only Stage-I components.

## Audited representation

`make_candidates` first selects S1 confidence seeds above a configurable
threshold. It expands them by a bounded number of fine voxels and propagates
the strongest supporting confidence. It enforces a strict site cap *before*
expansion. Candidate coordinates mean permission to reason about geometry,
not occupancy.

`make_targets` uses clean LiDAR only after the candidate domain is fixed. A
candidate is positive when measured clean points occupy that exact fine voxel.
The target offset is their centroid minus voxel center, normalized by voxel
size. `decode_centroids` reverses this to metric XYZ. The audit reports the
fraction of all in-grid clean points covered by candidates and the intrinsic
error of replacing multiple returns in one voxel with its centroid. No
network quality should be compared without these ceilings.

An absent clean return does **not** by itself prove that a candidate is free;
occluded and unobserved space needs a separate visibility mask before applying
negative occupancy BCE. This target module deliberately does not train a loss.

```bash
python -m scripts.audit_stage2_representation \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --checkpoint /path/to/stage1_run/best_selected.ckpt \
  --frame-id 08433 --split val \
  --output-root /path/to/stage2_representation_audit --device cuda
```

The audit exports full radar, clean LiDAR, candidate-site and oracle-centroid
PLY clouds. Diffusion is not present in the active path; no timesteps, noise
or diffusion parameters are created.
