# Dense geometric reconstruction experiment

This is an opt-in supervision change, not a new inference architecture. Original configurations without enabled `geometric_reference` and `geometric_loss` retain their existing behavior. Start a fresh experiment; old weights can technically initialize it, but old checkpoints were not trained for this objective.

## Inspected/reused pipeline

- Raw scans, calibration and splits: `Fault_Localization_Model/vod_dataset/vod_io.py` and `hercules_dataset.py`.
- Actual ego poses: VoD `vod_dataset/radar_accumulation.py`; HeRCULES `hercules_dataset.py::sensor_pose` and IMU extrinsics.
- Grid and height encoding: `Fault_Localization_Model/bev_utils.py`, `metric_to_grid`, `HEIGHT_RANGE_M=(-3,5)`, robust upper height (P90).
- Inputs/masks/PointPillars: shared `coarse_dataset.py`; geometric references are a separate batch field excluded from both trainers' model-input dictionaries.
- Existing objectives/metrics: `coarse_reconstruction/coarse_loss.py`, `diffusion_process/local_diffusion.py`, existing coarse/fine metric utilities.
- No maintained object-track adapter or full reconstructed-cloud decoder was available. The new decoder exposes the geometry actually represented by the BEV, without inventing intensity or vertical detail.

## Reference construction and leakage safeguards

`geometric_reference.py` aligns clean source scans into the central LiDAR frame using `inverse(world_from_central) @ world_from_source`. The configured window defaults to five past/five future scans plus the central scan. Only IDs in the same existing split are eligible. Walking stops at missing scans, nonadjacent VoD IDs, different HeRCULES sessions, or excessive pose/time steps. VoD sequence boundaries are inferred from adjacency and pose gates, not explicit recording IDs; verify these boundaries on the real dataset before a thesis run.

VoD reuses the existing odometry-reader convention and the per-frame LiDAR-to-camera calibration. Verify this convention against real measured static landmarks; synthetic tests cannot establish physical calibration correctness.

Noncentral annotated object boxes are excluded before alignment; central object regions are also excluded from neighboring scans. Central measured object geometry is retained. All labeled objects are excluded conservatively, including possibly static cars. There are no invented tracks. Missing annotations cause the default `boxes_or_central_only` strategy to use only the central scan. This is the default HeRCULES behavior because no supported box adapter exists. Explicit `persistence` is available as an experiment, but is not guaranteed to remove slow moving objects and must be visually validated. Explicit `boxes` fails if annotations are missing.

Voxel downsampling defaults to 5 cm; exact central measured points are unioned back afterward. Cache metadata records version, configuration, central/source IDs, split, transforms, retained counts, dynamics strategy and source path/size/mtime fingerprints. Rebuilding checks fingerprints; training checks cache version/config/frame/split but does not re-read raw source fingerprints. Rebuild after modifying raw data. Use distinct cache roots for distinct datasets/fault-generation roots. Offline future scans are supervision only: never model inputs, PointPillars, conditioning or normalization statistics. This changes the reference definition, not what is available at deployment.

## Exact training objective

For cell j, let p_j be soft occupancy and q_j its cell-center XY with predicted height z_j=-3+8*h_j. Let R be dense reference points inside the repair core, with matching candidates permitted in a metric halo (default 0.5 m). Outside-core predicted values are detached, so halo matching cannot optimize healthy cells.

Define d capped at D=0.5 m and rho(d)=d²/(2*delta) for d<delta, otherwise d-delta/2, with delta=0.1 m. Linear distance is also supported. The finite cap is required for the empty-occupancy event.

Accuracy: L_accuracy = sum_j[p_j*rho(min(distance(q_j,R_halo),D))] / max(sum_j p_j, epsilon), over repair-core cells.

Coverage: for each reference r, sort candidate cells by distance. Let a_j=p_j*product_(l<j)(1-p_l). Its expected nearest occupied-cell penalty is sum_j a_j*rho(min(distance(r,q_j),D)) + product_j(1-p_j)*rho(D). L_coverage is the mean over core reference points. Candidates outside D need not be enumerated because their penalty equals the empty-event cap.

Total: L = lambda_coverage*L_coverage + lambda_accuracy*L_accuracy + legacy_weight*L_existing. Defaults are 1.0, 0.5, 0.0. The existing loss is still computed/logged, but has zero objective weight. Coverage penalizes missing support; accuracy penalizes distant predicted support. No hard occupancy threshold is used in training. KD-tree nearest-index selection is detached; gathered distance computations retain local XYZ gradients. Coverage retains occupancy gradients through the Bernoulli expectation. Matching uses bounded candidate neighborhoods/chunks, not a dense all-pairs distance matrix.

Empty reference/prediction cases are explicitly handled. Density has no separate supervision when legacy_weight=0. Height is one upper-height value per cell: this is **2.5D surface supervision, not full 3D point-cloud reconstruction**. Lower surfaces/ground returns in raw references may conflict with upper-height cell geometry; compare clean-BEV versus dense-reference diagnostics to quantify this representation ceiling.

## Evaluation

Exact occupancy IoU/F1 remain against the historical single-frame target. Dense geometric diagnostics report Euclidean unsquared nearest distances in both directions, symmetric mean Chamfer, pooled p95 and geometric precision/recall/F1 at 0.1/0.2/0.5 m. They are frame-macro summaries, not global occupancy IoU. Empty matching distances are infinite (serialized as null); inspect counts, not just means.

Far predictions are not automatically proven hallucinations: unobserved space is unknown. Without an explicit observability map, hallucination rate is null and unmatched support is labeled unknown. Visualizations show clean/faulty/dense/coarse/fine and supported versus far-unknown geometry. No dynamic visual validation or real-data alignment validation has been performed on this machine.

## Commands (Linux/university machine)

Run from the repository root. Set PYTHON, DATA, RADAR, RAW, COARSE, SELECTOR_CONFIG and RUN to real paths. Copy/edit `configs/geometric_reconstruction.json` first: its cache_root is a placeholder. GEO points to that edited overlay. Keep the existing model configuration MODEL unchanged.

```bash
for SPLIT in train val test; do
  "$PYTHON" -u -m tools.build_geometric_references --data-root "$DATA" --raw-root "$RAW" --config "$GEO" --split "$SPLIT" || break
done

# Coarse: the overlay changes only the geometric objective/reference sections.
"$PYTHON" -u -m models.two_stage_reconstruction_head.coarse_reconstruction.train_coarse_reconstruction --data-root "$DATA" --radar-root "$RADAR" --output-root "$RUN" --config "$MODEL" --geometric-config "$GEO" --device cuda --epochs 150 --batch-size 8 --num-workers 8

# Fine: use the current coarse-enabled PointPillars diffusion MODEL.
"$PYTHON" -u -m models.two_stage_reconstruction_head.diffusion_process.train_fine_diffusion --data-root "$DATA" --radar-root "$RADAR" --coarse-checkpoint "$COARSE" --selector-config "$SELECTOR_CONFIG" --output-root "$RUN" --config "$MODEL" --geometric-config "$GEO" --device cuda --epochs 50 --batch-size 8 --validation-batch-size 8 --num-workers 8

"$PYTHON" -u -m tools.evaluate_geometric_reconstruction --data-root "$DATA" --radar-root "$RADAR" --coarse-checkpoint "$COARSE" --fine-checkpoint "$FINE" --selector-config "$SELECTOR_CONFIG" --config "$GEO" --output-root "$OUTPUT" --split test --device cuda --batch-size 1 --visualize-samples 50

"$PYTHON" -u -m tools.benchmark_geometric_reconstruction --data-root "$DATA" --radar-root "$RADAR" --selector-config "$SELECTOR_CONFIG" --config "$GEO" --device cuda --batch-size 1 --iterations 20 --output "$OUTPUT/geometric_loss_timing.json"
```

Omit --fine-checkpoint for coarse-only evaluation. Cache builder saves timing per split including cache hits. Real-batch benchmark times only loss forward/backward with a clean-BEV surrogate prediction, not model inference.

## Verification and limitations

Seven dedicated tests cover identity, translation, density/missing/hallucination cases, finite gradients, split isolation, annotation filtering and model-input exclusion. The synthetic CPU benchmark (320x320, 1600 reference points, batch 1, two measured iterations) recorded approximately 30.7 ms forward and 7.6 ms backward. This is not a real-dataset or GPU performance claim. Real construction timing, peak GPU memory, actual pose alignment and dynamic visualization require running the provided tools on the university machine before drawing experimental conclusions.
