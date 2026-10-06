# Sudoku results summary

5,000-step EMA checkpoints, 48,000 Sudoku Hard training puzzles, and 2,000 full-hint validation puzzles. Each cell below shows **success % / mean NFE**. `hidden` is normal self-conditioning; `token` is intermediate token feedback. Detached depth 0/3 means extraction after block 1/4.

[Hint-removal results: all hint levels, valid + unique percentages](sudoku-hard-5k/hint-removal/report.md) · [Full-hint CSV](sudoku-hard-5k/full-hints/summary.csv) · [Setup](README.md)

| Model | 4 NFE | 8 NFE | Regular 32 steps | Regular 180 steps | Frozen EB .1 | Frozen EB .01 | Revisable EB .1 | Revisable EB .01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian-hidden | 0.00% / 4.00 | 91.75% / 8.00 | 6.15% / 33.00 | 0.00% / 181.00 | — | — | — | — |
| gaussian-masked-hidden | — | — | 40.90% / 33.00 | 74.05% / 181.00 | 78.40% / 4.47 | 82.05% / 5.50 | — | — |
| gaussian-masked-none | — | — | 21.30% / 33.00 | 24.20% / 181.00 | 68.05% / 4.63 | 77.25% / 8.75 | — | — |
| gaussian-masked-token | — | — | 27.35% / 33.00 | 30.00% / 181.00 | 89.35% / 4.36 | 93.90% / 8.09 | — | — |
| gaussian-token | 36.75% / 4.00 | 65.55% / 8.00 | 63.05% / 33.00 | 57.00% / 181.00 | — | — | — | — |
| gaussian-uniform-hidden | — | — | 95.15% / 33.00 | 96.00% / 181.00 | 79.40% / 7.94 | 77.75% / 16.38 | 79.20% / 36.19 | 78.15% / 100.16 |
| gaussian-uniform-none | — | — | 33.95% / 33.00 | 34.70% / 181.00 | 57.70% / 20.26 | 53.25% / 33.74 | 53.15% / 101.62 | 20.05% / 179.62 |
| gaussian-uniform-token | — | — | 58.45% / 33.00 | 60.60% / 181.00 | 71.10% / 12.99 | 70.80% / 22.79 | 74.60% / 57.19 | 61.15% / 100.42 |
| masked-hidden | — | — | 38.00% / 33.00 | 69.25% / 181.00 | 73.10% / 4.40 | 80.30% / 4.93 | — | — |
| masked-none | — | — | 10.65% / 33.00 | 11.80% / 181.00 | 78.45% / 5.51 | 85.30% / 10.46 | — | — |
| masked-token | — | — | 31.95% / 33.00 | 34.10% / 181.00 | 82.50% / 3.77 | 90.05% / 7.36 | — | — |
| masked-token-detached-depth0 | — | — | 30.35% / 33.00 | 34.45% / 181.00 | 78.95% / 3.73 | 89.25% / 6.57 | — | — |
| masked-token-detached-depth3 | — | — | 32.45% / 33.00 | 35.35% / 181.00 | 79.65% / 3.77 | 89.50% / 7.08 | — | — |
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
| uniform-ce-hidden | — | — | 96.65% / 33.00 | 96.10% / 181.00 | 93.10% / 3.45 | 95.45% / 4.70 | 96.00% / 3.79 | 96.40% / 52.67 |
| uniform-ce-none | — | — | 47.60% / 33.00 | 48.30% / 181.00 | 63.90% / 19.22 | 65.45% / 29.80 | 69.80% / 71.55 | 53.35% / 117.72 |
| uniform-ce-token | — | — | 79.30% / 33.00 | 78.45% / 181.00 | 84.50% / 7.41 | 86.90% / 13.00 | 90.70% / 22.93 | 86.60% / 39.63 |
| uniform-plugin-hidden | — | — | 92.05% / 33.00 | 92.90% / 181.00 | 81.70% / 10.06 | 87.85% / 42.98 | 91.60% / 106.59 | 91.80% / 180.00 |
| uniform-plugin-none | — | — | 40.90% / 33.00 | 42.35% / 181.00 | 43.15% / 24.77 | 40.40% / 60.25 | 46.70% / 146.47 | 44.25% / 180.00 |
| uniform-plugin-token | — | — | 75.30% / 33.00 | 73.55% / 181.00 | 72.45% / 13.04 | 73.70% / 36.81 | 77.50% / 74.68 | 74.85% / 151.87 |

Regular 32/180-step samplers may include an additional final readout; the table reports measured NFE. Continuous-only entropy commitment is an experimental sampling policy, shown separately below.

| Model | Experimental frozen EB bound | Success % / NFE |
|---|---:|---:|
| gaussian-hidden | 0.01 | 95.35% / 3.93 |
| gaussian-hidden | 0.1 | 92.85% / 3.41 |
| gaussian-token | 0.01 | 28.80% / 11.58 |
| gaussian-token | 0.1 | 40.00% / 4.88 |
| simplex-hidden | 0.01 | 96.70% / 4.83 |
| simplex-hidden | 0.1 | 94.05% / 4.11 |
| simplex-token | 0.01 | 73.85% / 19.83 |
| simplex-token | 0.1 | 68.55% / 13.51 |
| spherical-hidden | 0.01 | 96.60% / 3.68 |
| spherical-hidden | 0.1 | 93.45% / 3.22 |
| spherical-token | 0.01 | 12.15% / 15.63 |
| spherical-token | 0.1 | 24.30% / 6.48 |
