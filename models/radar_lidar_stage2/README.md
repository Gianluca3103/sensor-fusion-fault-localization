# Stage II: deterministic radar-supported LiDAR geometry

Stage II consumes the **frozen, deployed radar-only** Stage-I S1–S4 sparse
features and candidate confidence. A learned-region Stage-I checkpoint supplies
predicted LiDAR surface centers, scored support extents, and contiguous fine
candidate cells. A point-only surface checkpoint seeds small neighborhoods from
predicted positions; a legacy checkpoint seeds from radar S1 positions. Clean
LiDAR is read only by target, loss, metric,
and visualization code. The model forward API accepts Stage-I radar output and
the physical voxel grid; it has no clean/faulty LiDAR or semantic input.

The active baseline has **no diffusion**: no timesteps, noise, denoising loss,
or reverse sampling. `SparseUNet.forward(..., conditioning=None,
timestep=None)` leaves an explicit interface for later study and rejects a
non-null timestep. The reconstruction output is separate from any downstream
fusion with observed LiDAR.

## Representation and ceilings

- Stage-I sites use sorted unique `[batch,z,y,x]` integer coordinates.
  Physical XYZ centers use the Stage-I checkpoint grid, commonly
  0.2 × 0.2 × 0.25 m. S1–S4 strides are 1, 2, 4, 8 and their physical query
  radii are configured in metres. The four features are queried locally,
  aggregated with inverse-distance weights, projected, and fused with
  candidate confidence and normalized position.
- Stage-I candidate confidence above `confidence_threshold` activates an
  entire learned support patch when region extents are present. The candidate
  cap accepts complete patches in score order. Point-only checkpoints use a
  bounded fixed neighborhood; legacy checkpoints use radar S1 voxel centers.
  `max_candidate_sites` caps the **fixed** sparse domain. Candidate means
  *eligible for prediction*, never occupied. The cap and candidate coverage
  must be reported with reconstruction performance.
- Each candidate predicts an occupancy logit and an XYZ centroid offset
  bounded to ±0.5 voxel. An occupied prediction decodes to one XYZ point and
  retains its supporting confidence. This is a V1 representation, not a claim
  that one centroid can reproduce every clean return in a voxel.
- The clean target marks a candidate positive if an observed clean return
  occupies its cell. The offset is the mean of those observed returns relative
  to the cell center. Other cells are **unknown** unless a measured clean ray
  passes through them before its first return. The free-ray label is
  conservative: the candidate center must lie within the configured metric
  tolerance of that ray and at least half a voxel diagonal before the return.
  Behind the return remains unknown and has no negative BCE term.
- Occupancy BCE uses measured positives and visible free sites only; offset
  SmoothL1 uses positives only. Validate the counts of positive, known-free,
  and unknown sites. Sparse negatives are a limitation: a model could still
  predict in unknown cells. Held-out point precision exposes that behavior.
- `scripts.audit_stage2_representation` measures the **candidate ceiling**
  (clean-return coverage) and **representation ceiling** (oracle voxel
  centroids) before learned quality is interpreted.

## MinkowskiEngine and training

The compatibility probe must report `core_ready: true` in the same Python
environment as training. The verified professor-machine probe reported
MinkowskiEngine 0.5.4 with CPU/CUDA forward and backward and CUDA FP16/BF16.
The U-Net uses coordinate-preserving 3³ sparse convolutions, three strided
downsampling levels, and non-generative transposed convolutions. Skip
coordinate-map keys and the final candidate coordinate set are checked at
runtime. Site counts are included in each model output. No generative sparse
transpose is used in the active deterministic path.

Run `python -m scripts.smoke_stage2_minkowski` in that environment first. It
checks Stage-II forward/backward, fixed sparse support, frozen Stage-I feature
boundary, finite gradients, and checkpoint reload without needing VoD data.

Run a **one-frame smoke test first** from the repository root, replacing the
paths with the professor-machine paths:

```bash
python -u -m models.radar_lidar_stage2.train \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --stage1-checkpoint /path/to/stage1_run/best_selected.ckpt \
  --output-root /path/to/stage2_smoke \
  --config configs/radar_lidar_stage2_small.json \
  --epochs 1 --batch-size 1 --train-limit 1 --val-limit 1 \
  --validate-every 1 --num-workers 0 --device cuda
```

Once that succeeds, remove the sample limits and use a new output directory.
The trainer writes `last.ckpt`, `best_geom.ckpt`, and `metrics.jsonl`. Best
selection uses held-out geometry F1 at 0.2 m by default. Training prints a
compact progress bar and an epoch summary. Batch size 1 is a conservative
starting point for 40,000 candidate sites and four sparse U-Net levels; raise
it after checking peak GPU memory.

## Evaluation and inspection

```bash
python -u -m models.radar_lidar_stage2.evaluate \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --checkpoint /path/to/stage2_run/best_geom.ckpt \
  --output-root /path/to/stage2_eval \
  --candidate-thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --ablations real zero shuffle wrong_sample no_confidence s1_only s4_only \
  --visualize-frames 08433 08444 --device cuda
```

The evaluator writes dataset-aggregated point precision/recall/F1 at 0.1,
0.2, and 0.5 m; supervised-visible occupancy metrics; offset errors;
candidate coverage; counts; confidence-bin diagnostics; threshold sweeps;
and evidence ablations. `wrong_sample` needs at least two frames. It replaces
the conditioning features while holding the real candidate domain fixed;
this tests feature reliance, while the fixed domain remains radar-derived.
Selected frames produce physical XYZ PLYs for radar, clean LiDAR, candidates,
oracle occupied centroids, predictions, false predicted points, and missed
clean returns. No object labels enter the model.

For a rotating three-panel comparison without running full validation, use
`python -m scripts.visualize_stage2_reconstruction --vod-root /path/to/vod
--checkpoint /path/to/best_geom.ckpt --frame-id 00000 00001
--fault-samples-root /path/to/fault-cache/samples --output-root /path/to/preview`.
Open each resulting `*_stage2.html` in a browser. It shows radar, clean LiDAR,
and Stage-II predicted points with shared rotate/zoom/pan controls. Faulty
LiDAR can be toggled over the third panel. Full-resolution PLY files are saved
for all four clouds. The preview does not pass faulty LiDAR to the model or
merge it into the predictions.

The active formulation answers whether the radar-derived Stage-I
representation can deterministically support useful LiDAR geometry. Diffusion
should be considered only after candidate coverage, representation loss, and
learned reconstruction are separately measured.

## Frozen detector comparison

`scripts.export_stage2_vod_detector` writes all 1,296 official validation
frames as four-column LiDAR binaries for `prepare_vod_official_faults.py`.
Its default `--mode merged` retains each matched faulty LiDAR XYZI return and
appends Stage-II generated XYZ with **zero intensity**. `--mode generated-only`
exports just the synthetic points as an ablation. Stage I/II inference remains
radar-only in either mode; faulty LiDAR is only merged after inference. Clean
LiDAR is not read by the exporter. The detector must receive its original
five-frame radar branch in the downstream SVEFusion evaluation. Use the same
official detector checkpoint and validation IDs for clean, faulty, and
reconstructed comparisons. Zero synthetic intensity is a compatibility
placeholder and must be reported with detector results.
