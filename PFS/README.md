# PFS Reliability Map Model

This folder adapts the Post Fusion Stabilizer idea to the Hercules BEV fault-localization problem.

The original PFS paper places a lightweight feature stabilizer between a fused BEV representation and a frozen detector head. In this project there is no frozen 3D detector head, so the adapted setup is:

- input: `faulty_rgb` BEV, the same BEV image a real test sample would provide
- training-only helper: `clean_rgb` BEV, used only for feature stabilization loss
- target: `fault_heatmap`, where `0` means reliable and `1` means unreliable/faulty
- output: learned dense unreliability heatmap

## Model Blocks

`pfs_model.py` implements the three PFS-style blocks:

1. `ShiftNormalization`
   - learns a gated channel-wise correction of BEV feature statistics
   - initialized close to identity

2. `SpatialReliabilityEstimator`
   - predicts an internal reliability gate from BEV features
   - initialized with high reliability so early training does not suppress everything

3. `ExpertCorrection`
   - applies gated residual correction using semantic and geometric expert branches
   - initialized close to identity

The final heatmap decoder is U-Net-like, matching the current fault localization model style.

## Train

From the repo root:

```powershell
cd "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"
& "C:\Users\gianl\miniconda3\python.exe" PFS\train_pfs_reliability_map.py `
  --dataset-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\Fault_Localization_Model\grid_reliability_7500_fog_s3_x64_y32" `
  --output-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\PFS\runs\pfs_7500_fog_s3_x64_y32" `
  --epochs 100 `
  --batch-size 24 `
  --base-channels 16 `
  --dropout 0.15 `
  --learning-rate 3e-4 `
  --weight-decay 1e-3 `
  --loss-mode stable `
  --grad-clip 1.0 `
  --early-stop-patience 12 `
  --resize-height 320 `
  --resize-width 320 `
  --grid-size 100 `
  --stability-weight 0.15 `
  --pfs-reliability-weight 0.10 `
  --max-val-images 24 `
  --num-workers 4 `
  --device cuda
```

For 8 GB VRAM, `--batch-size 24` has worked well with the RTX 5060 laptop GPU. If it runs out of memory, drop to `--batch-size 16`.

## Outputs

The trainer saves:

- `checkpoints/best_model.pt`
- `training_history.csv`
- `plots/training_curve.png`
- `val_predictions/*_comparison.png`

The validation comparison images reuse the current project visualization style: faulty BEV input, ideal heatmap, and model-predicted heatmap side by side.
