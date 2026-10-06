# Adaptive Sudoku sweep evaluation

5,000-step EMA checkpoints; 48,000 training puzzles; 2,000 validation puzzles; seed 1. Cells show solved percentage / mean model calls. Regular 32/180-step runs use 33/181 calls including final readout; the 4/8-NFE runs use exactly 4/8 calls. EB uses greedy commitment and at most 180 transition steps (plus an endpoint readout when unfinished).

Regular sampling retains each model’s configured update: Gaussian Euler, spherical manifold update, or simplex stochastic bridge. Frozen EB is cumulative; revisable uniform-hybrid EB re-corrupts released continuous commitments at destination time and stops only on simultaneous acceptance. No hidden-feedback resets were applied.

| Run | Regular 4 NFE | Regular 8 NFE | Regular 32 steps | Regular 180 steps | Frozen .1 | Frozen .01 | Revisable .1 | Revisable .01 |
|---|---|---|---|---|---|---|---|---|
| gaussian-hidden | 0.00% / 4.00 | 91.75% / 8.00 | 6.15% / 33.00 | 0.00% / 181.00 | — | — | — | — |
| gaussian-masked-hidden | — | — | 40.90% / 33.00 | 74.05% / 181.00 | 78.40% / 4.47 | 82.05% / 5.50 | — | — |
| gaussian-masked-none | — | — | 21.30% / 33.00 | 24.20% / 181.00 | 68.05% / 4.63 | 77.25% / 8.75 | — | — |
| gaussian-masked-token | — | — | 27.35% / 33.00 | 30.00% / 181.00 | 89.35% / 4.36 | 93.90% / 8.09 | — | — |
| gaussian-token | 36.75% / 4.00 | 65.55% / 8.00 | 63.05% / 33.00 | 57.00% / 181.00 | — | — | — | — |
| gaussian-uniform-hidden | — | — | 95.15% / 33.00 | 96.00% / 181.00 | 79.40% / 7.94 | 77.75% / 16.38 | 79.20% / 36.19 | 78.15% / 100.16 |
| gaussian-uniform-none | — | — | 33.95% / 33.00 | 34.70% / 181.00 | 57.70% / 20.26 | 53.25% / 33.74 | 53.15% / 101.62 | 20.05% / 179.62 |
| gaussian-uniform-token | — | — | 58.45% / 33.00 | 60.60% / 181.00 | 71.10% / 12.99 | 70.80% / 22.79 | 74.60% / 57.19 | 61.15% / 100.42 |
| simplex-hidden | 69.15% / 4.00 | 87.90% / 8.00 | 96.40% / 33.00 | 97.00% / 181.00 | — | — | — | — |
| simplex-masked-hidden | — | — | 56.95% / 33.00 | 87.00% / 181.00 | 83.55% / 3.33 | 91.95% / 3.79 | — | — |
| simplex-masked-none | — | — | 27.65% / 33.00 | 36.55% / 181.00 | 58.30% / 4.02 | 67.30% / 5.99 | — | — |
| simplex-masked-token | — | — | 51.95% / 33.00 | 60.00% / 181.00 | 85.65% / 3.09 | 90.95% / 4.59 | — | — |
| simplex-token | 52.65% / 4.00 | 58.15% / 8.00 | 73.45% / 33.00 | 83.95% / 181.00 | — | — | — | — |
| simplex-uniform-hidden | — | — | 96.25% / 33.00 | 96.45% / 181.00 | 88.85% / 3.27 | 95.15% / 3.55 | 95.25% / 3.52 | 95.30% / 4.80 |
| simplex-uniform-none | — | — | 49.75% / 33.00 | 61.40% / 181.00 | 62.65% / 12.85 | 64.25% / 19.48 | 74.55% / 51.59 | 69.60% / 89.98 |
| simplex-uniform-token | — | — | 72.20% / 33.00 | 78.30% / 181.00 | 80.75% / 6.46 | 82.85% / 11.31 | 87.30% / 25.96 | 84.55% / 39.94 |
| spherical-hidden | 95.05% / 4.00 | 97.45% / 8.00 | 97.75% / 33.00 | 97.75% / 181.00 | — | — | — | — |
| spherical-masked-hidden | — | — | 43.95% / 33.00 | 80.05% / 181.00 | 84.65% / 4.10 | 89.85% / 4.52 | — | — |
| spherical-masked-none | — | — | 20.65% / 33.00 | 23.65% / 181.00 | 16.45% / 6.29 | 11.20% / 12.51 | — | — |
| spherical-masked-token | — | — | 33.35% / 33.00 | 37.90% / 181.00 | 85.55% / 3.77 | 91.10% / 6.45 | — | — |
| spherical-token | 56.15% / 4.00 | 56.35% / 8.00 | 57.75% / 33.00 | 58.80% / 181.00 | — | — | — | — |
| spherical-uniform-hidden | — | — | 96.35% / 33.00 | 95.75% / 181.00 | 93.35% / 4.99 | 95.10% / 7.61 | 95.85% / 7.28 | 96.40% / 69.94 |
| spherical-uniform-none | — | — | 46.60% / 33.00 | 47.95% / 181.00 | 79.30% / 7.72 | 65.60% / 18.63 | 81.70% / 36.87 | 49.15% / 128.25 |
| spherical-uniform-token | — | — | 62.50% / 33.00 | 62.45% / 181.00 | 84.75% / 6.80 | 83.80% / 15.18 | 85.75% / 28.95 | 72.80% / 69.19 |

`hidden` is recurrent hidden self-conditioning; `token` is intermediate token feedback. Continuous-only frozen EB is an added inference policy, not a paper-native sampler. New EB evaluations and cached regular evaluations retain their full per-puzzle results and source paths.

126 evaluation results currently available.

Continuous-only commitment experiments are excluded from the main EB columns; see [experimental results](experimental-continuous-commitment.md).
