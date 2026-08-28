# Fine Diffusion Transformer validation threshold sweeps

Recorded: 2026-08-28

Split: validation (1,296 samples; 1,127 used by the selector)

The clean and coarse occupancy thresholds remain fixed at 0.500. Only the Fine Diffusion occupancy threshold is swept.

## Compact comparison

| Model | Best exact IoU | Threshold | Exact gain vs coarse | Best IoU@0.5m | Threshold | IoU@0.5m gain vs coarse |
|---|---:|---:|---:|---:|---:|---:|
| Shallow Transformer, crop bucket 32 | 20.99% | 0.45 | +1.44 pp | 55.41% | 0.30 | +1.07 pp |
| Deep Transformer, h128/a192/6 heads/6 blocks | 22.14% | 0.40 | +2.59 pp | 56.82% | 0.20 | +2.48 pp |

The deeper Transformer is better on both peak exact IoU and peak tolerant IoU. Its best threshold depends on the objective: 0.40 for exact IoU, 0.20 for IoU@0.5m, and 0.30 for a more balanced operating point (21.56% exact IoU and 55.83% IoU@0.5m).

## Shallow Transformer

Run:

`/workspace/fine_diffusion_transformer_best_b32_radar20_b12_crop_bucket32_50epochs`

### Per-fault results at threshold 0.500

| Fault | N | Used | Faulty IoU | Coarse IoU | Fine IoU | Fine-Coarse | Faulty@0.5m | Coarse@0.5m | Fine@0.5m | Fine F1@0.5m | Hallucination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fog_sim_s4 | 648 | 543 | 4.62% | 17.33% | 16.65% | -0.68 pp | 8.44% | 54.48% | 45.65% | 60.02% | 1.81% |
| fog_sim_s5 | 648 | 584 | 6.22% | 20.00% | 21.62% | +1.62 pp | 10.25% | 53.52% | 51.63% | 67.14% | 2.52% |

### Occupancy threshold sweep

| Thr | Exact IoU | Delta | F1 | P | R | IoU@0.5m | F1@0.5m | Delta@0.5m | Benefit+ | Harm+ | Recover | Benefit- | Harm- |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 11.81% | -7.74 pp | 21.13% | 12.54% | 67.00% | 46.07% | 63.08% | -8.27 pp | 114234 | 1653000 | 42.24% | 4417 | 594 |
| 0.15 | 14.23% | -5.33 pp | 24.91% | 15.62% | 61.53% | 50.43% | 67.05% | -3.91 pp | 88780 | 1016860 | 32.83% | 8623 | 1152 |
| 0.20 | 16.24% | -3.32 pp | 27.94% | 18.49% | 57.18% | 53.18% | 69.43% | -1.16 pp | 69153 | 642442 | 25.57% | 16180 | 2173 |
| 0.25 | 17.87% | -1.68 pp | 30.33% | 21.15% | 53.57% | 54.77% | 70.78% | +0.44 pp | 53787 | 405600 | 19.89% | 28512 | 3986 |
| 0.30 | 19.14% | -0.42 pp | 32.12% | 23.60% | 50.27% | 55.41% | 71.31% | +1.07 pp | 40997 | 248260 | 15.16% | 47052 | 6891 |
| 0.35 | 20.08% | +0.52 pp | 33.44% | 25.92% | 47.10% | 55.38% | 71.28% | +1.04 pp | 30432 | 140432 | 11.25% | 72554 | 11357 |
| 0.40 | 20.72% | +1.16 pp | 34.32% | 28.18% | 43.91% | 54.72% | 70.73% | +0.39 pp | 21896 | 67096 | 8.10% | 107136 | 18025 |
| 0.45 | 20.99% | +1.44 pp | 34.70% | 30.38% | 40.45% | 53.49% | 69.69% | -0.85 pp | 15227 | 21428 | 5.63% | 152904 | 27772 |
| 0.50 | 20.87% | +1.31 pp | 34.53% | 32.66% | 36.63% | 51.50% | 67.99% | -2.84 pp | 11023 | 655 | 4.08% | 213573 | 41704 |
| 0.55 | 20.21% | +0.66 pp | 33.63% | 35.00% | 32.36% | 48.53% | 65.35% | -5.80 pp | 9479 | 1 | 3.50% | 286401 | 60477 |
| 0.60 | 19.05% | -0.51 pp | 32.00% | 37.66% | 27.83% | 44.71% | 61.80% | -9.62 pp | 7935 | 0 | 2.93% | 353033 | 80483 |
| 0.70 | 14.71% | -4.84 pp | 25.65% | 44.26% | 18.06% | 34.38% | 51.17% | -19.96 pp | 4841 | 0 | 1.79% | 463870 | 123815 |

At threshold 0.500: beneficial additions 11,023; harmful additions 655; beneficial removals 213,573; harmful removals 41,704.

Saved evaluation:

`/workspace/transformer_threshold_sweeps/fine_diffusion_transformer_best_b32_radar20_b12_crop_bucket32_50epochs`

## Deep Transformer

Run:

`/workspace/fine_diffusion_transformer_h128_a192_heads6_blocks6_b32_radar20_b8_50epochs`

### Per-fault results at threshold 0.500

| Fault | N | Used | Faulty IoU | Coarse IoU | Fine IoU | Fine-Coarse | Faulty@0.5m | Coarse@0.5m | Fine@0.5m | Fine F1@0.5m | Hallucination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fog_sim_s4 | 648 | 543 | 4.62% | 17.33% | 15.35% | -1.97 pp | 8.44% | 54.48% | 39.22% | 53.73% | 0.90% |
| fog_sim_s5 | 648 | 584 | 6.22% | 20.00% | 22.39% | +2.39 pp | 10.25% | 53.52% | 48.68% | 64.58% | 1.49% |

### Occupancy threshold sweep

| Thr | Exact IoU | Delta | F1 | P | R | IoU@0.5m | F1@0.5m | Delta@0.5m | Benefit+ | Harm+ | Recover | Benefit- | Harm- |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 16.44% | -3.11 pp | 28.24% | 18.43% | 60.37% | 55.35% | 71.26% | +1.02 pp | 82196 | 702200 | 30.39% | 4548 | 84 |
| 0.15 | 18.62% | -0.94 pp | 31.39% | 22.04% | 54.54% | 56.60% | 72.29% | +2.27 pp | 54921 | 357379 | 20.31% | 12616 | 525 |
| 0.20 | 19.96% | +0.41 pp | 33.28% | 24.76% | 50.75% | 56.82% | 72.46% | +2.48 pp | 38992 | 192802 | 14.42% | 31915 | 2587 |
| 0.25 | 20.89% | +1.34 pp | 34.57% | 27.17% | 47.49% | 56.56% | 72.25% | +2.22 pp | 27330 | 94764 | 10.11% | 61730 | 6409 |
| 0.30 | 21.56% | +2.00 pp | 35.47% | 29.54% | 44.38% | 55.83% | 71.66% | +1.50 pp | 19282 | 36407 | 7.13% | 105276 | 13169 |
| 0.35 | 21.99% | +2.43 pp | 36.05% | 32.01% | 41.25% | 54.68% | 70.70% | +0.34 pp | 14930 | 9165 | 5.52% | 164612 | 23665 |
| 0.40 | 22.14% | +2.59 pp | 36.26% | 34.65% | 38.02% | 53.09% | 69.36% | -1.24 pp | 13493 | 1266 | 4.99% | 232448 | 37603 |
| 0.45 | 21.99% | +2.44 pp | 36.06% | 37.77% | 34.49% | 51.10% | 67.64% | -3.24 pp | 13221 | 64 | 4.89% | 301922 | 54091 |
| 0.50 | 21.35% | +1.80 pp | 35.19% | 41.51% | 30.54% | 48.23% | 65.07% | -6.11 pp | 13209 | 4 | 4.88% | 367461 | 72854 |
| 0.55 | 20.00% | +0.45 pp | 33.34% | 46.31% | 26.04% | 44.05% | 61.16% | -10.28 pp | 13205 | 0 | 4.88% | 428500 | 94239 |
| 0.60 | 17.58% | -1.98 pp | 29.90% | 52.77% | 20.86% | 38.11% | 55.19% | -16.23 pp | 13198 | 0 | 4.88% | 483215 | 118847 |
| 0.70 | 9.97% | -9.59 pp | 18.13% | 73.75% | 10.33% | 21.32% | 35.15% | -33.02 pp | 13142 | 0 | 4.86% | 554468 | 168816 |

At threshold 0.500: beneficial additions 13,209; harmful additions 4; beneficial removals 367,461; harmful removals 72,854.

