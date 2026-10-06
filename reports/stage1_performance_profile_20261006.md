# Stage 1 end-to-end performance profile — 2026-10-06

## Run status and method

The local 50-epoch trainer (PID 7584) was no longer running when the CUDA profile began. It had saved `last.ckpt` and `metrics.jsonl` for epoch 1 at 15:30 local time; validation was not scheduled for epoch 1. The profile did not resume or modify that run.

`scripts/profile_stage1_batch.py` ran one isolated forward, backward, and optimizer step with the same small config, radar variant, batch size 4, and real VoD frames. CUDA timings synchronize before and after each measured component, so they include launch and wait overhead. Initial frames 0–3 have unusually few radar returns; the main measurement uses frames 1000–1003 (about 9,000 radar and 182,000 LiDAR points per frame). A second measurement used frames 2500–2503 (about 4,800 radar and 169,000 LiDAR points per frame). These are single-batch diagnostics, not a full-epoch average.

## Main CUDA measurement, frames 1000–1003

| Stage | Wall time | Calls | Share of 3.48 s step |
| --- | ---: | ---: | ---: |
| Load and collate four frames | 0.017 s | 1 | 0.5% |
| Host-to-device copy | 0.004 s | 1 | 0.1% |
| Forward, all components | 2.006 s | 1 | 57.7% |
| Backward | 1.419 s | 1 | 40.8% |
| Gradient clipping and optimizer | 0.051 s | 1 | 1.5% |

Inclusive forward subcomponent times:

| Component | Time | Calls | Share of forward |
| --- | ---: | ---: | ---: |
| Custom sparse 3D convolution | 1.204 s | 25 | 60.1% |
| Radar and LiDAR point-to-voxel encoders | 0.394 s | 2 | 19.7% |
| CPU KD-tree neighbor search with GPU transfers | 0.148 s | 4 | 7.4% |
| Cross-attention and contrastive loss | 0.099 s | 4 | 4.9% |
| Connectivity diagnostics | 0.054 s | 8 | 2.7% |
| Probe target construction | 0.012 s | 4 | 0.6% |

The second CUDA batch took 2.58 s (1.87 s forward, 0.66 s backward). Its sparse convolutions still took 1.17 s and KD-tree search 0.11 s. A four-frame CPU run also ranked sparse convolutions first: 2.77 s of a 3.34 s forward. These timings identify sparse convolution as the dominant measured forward cost; KD-tree search contributes but is not the main bottleneck.

## Validation path

On validation frame `00100`, full VoD inputs and batch size 1:

| Validation operation | Time |
| --- | ---: |
| Loss forward | 1.086 s |
| Metrics update | 1.069 s |
| Object-instance evaluation | 1.185 s |
| Total compute | 3.340 s |

The validation loop calls the radar and LiDAR encoders separately for loss, aggregate metrics, and object metrics. At the measured frame rate, 1,296 validation frames would require about 72 minutes of compute, plus loading and reporting, every fifth epoch. Frame complexity varies, so this is an illustrative extrapolation, not a measured full-validation duration.

## Why GPU utilization is low

`SparseConv3d.forward` performs 27 Python-level offset iterations per convolution and calls `bool(inbounds.any())` and `bool(present.any())` inside each iteration. With 25 sparse convolutions in a training forward, this can create up to 1,350 GPU-to-CPU decisions per batch, serializing many small GPU operations. The KD-tree path also copies coordinates to CPU and results back to GPU four times. Thus the GPU can have high memory occupancy while doing little sustained work. The first optimization target is replacing or vectorizing this custom sparse convolution; adding DataLoader workers will not address its cost.

## Limits and next steps

The running epoch previously took roughly 116 minutes for 1,285 batches, whereas isolated CUDA batches took 2.58–3.48 seconds. The difference is not fully attributed here; system contention, batch variability, throttling, or training-loop overhead may contribute. The profiler does not claim an exact full-epoch speedup.

1. Replace the Python-loop sparse convolution with a sparse-convolution implementation compatible with Windows/CUDA or a vectorized neighbor map, then benchmark one epoch on the same frames and config. Preserve the 0.2 × 0.2 × 0.25 m fine voxel grid.
2. Reuse radar and LiDAR encoded features in validation rather than recomputing them three times per frame.
3. Only then optimize the CPU KD-tree correspondence path. It is approximately 0.1–0.15 s per measured training forward, much less than sparse convolution plus backward.
4. Re-run the isolated profile and compare median batch time and full-epoch wall time before changing batch size or worker count.

Machine-readable profiles are in `outputs_local/stage1_profile_cuda_train_val.json`, `outputs_local/stage1_profile_cuda_typical4_2500.json`, and `outputs_local/stage1_profile_cpu_full4.json`.

## Implemented kernel-map optimization

`SparseConv3d` now builds a GPU neighbor map in bounded 32,768-output-site chunks and reuses it across convolutions with identical active coordinates. The original method remains as `forward_reference` for parity tests. The model parameters, voxel grid, and checkpoint schema did not change.

Same machine, same frames 1000–1003, isolated batch size 4, synchronized CUDA timings:

| Measurement | Original reference | Kernel map |
| --- | ---: | ---: |
| Sparse convolution forward time, 25 calls | 1.190 s | 0.259 s |
| Whole forward | 1.920 s | 1.008 s |
| Whole forward + backward + optimizer | 3.481 s | 2.489 s |
| Peak CUDA memory allocated | 4,261 MiB | 4,214 MiB |

Frames 2500–2503 showed sparse convolution falling from 1.172 s to 0.216 s and whole-step time from 2.577 s to 1.545 s. These are isolated measurements; a full-epoch throughput measurement is still needed to establish the sustained speedup. `tests/test_radar_lidar_stage1.py` passes 22 tests, including output and gradient parity for normal, growing, and downsampling convolutions on CPU and CUDA, map reuse, a map chunk boundary, empty inputs, and full Stage 1 loss/gradient parity. The saved epoch-1 checkpoint loads strictly into the updated model.
