# Sudoku results summary

5,000-step EMA checkpoints, 48,000 Sudoku Hard training puzzles, and 2,000 full-hint validation puzzles. Each cell below shows **success % / mean NFE**. Only normal hidden-representation self-conditioning (`hidden`) runs are shown.

[Hint-removal results: all hint levels, valid + unique percentages](sudoku-hard-5k/hint-removal/report.md) · [Full-hint CSV](sudoku-hard-5k/full-hints/summary.csv) · [Setup](README.md)

| Model | 4 NFE | 8 NFE | Regular 32 steps | Regular 180 steps | Frozen EB .1 | Frozen EB .01 | Revisable EB .1 | Revisable EB .01 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian-hidden | 0.00% / 4.00 | 91.75% / 8.00 | 6.15% / 33.00 | 0.00% / 181.00 | — | — | — | — |
| gaussian-masked-hidden | — | — | 40.90% / 33.00 | 74.05% / 181.00 | 78.40% / 4.47 | 82.05% / 5.50 | — | — |
| gaussian-uniform-hidden | — | — | 95.15% / 33.00 | 96.00% / 181.00 | 79.40% / 7.94 | 77.75% / 16.38 | 79.20% / 36.19 | 78.15% / 100.16 |
| masked-hidden | — | — | 38.00% / 33.00 | 69.25% / 181.00 | 73.10% / 4.40 | 80.30% / 4.93 | — | — |
| simplex-hidden | 69.15% / 4.00 | 87.90% / 8.00 | 96.40% / 33.00 | 97.00% / 181.00 | — | — | — | — |
| simplex-masked-hidden | — | — | 56.95% / 33.00 | 87.00% / 181.00 | 83.55% / 3.33 | 91.95% / 3.79 | — | — |
| simplex-uniform-hidden | — | — | 96.25% / 33.00 | 96.45% / 181.00 | 88.85% / 3.27 | 95.15% / 3.55 | 95.25% / 3.52 | 95.30% / 4.80 |
| spherical-hidden | 95.05% / 4.00 | 97.45% / 8.00 | 97.75% / 33.00 | 97.75% / 181.00 | — | — | — | — |
| spherical-masked-hidden | — | — | 43.95% / 33.00 | 80.05% / 181.00 | 84.65% / 4.10 | 89.85% / 4.52 | — | — |
| spherical-uniform-hidden | — | — | 96.35% / 33.00 | 95.75% / 181.00 | 93.35% / 4.99 | 95.10% / 7.61 | 95.85% / 7.28 | 96.40% / 69.94 |
| uniform-ce-hidden | — | — | 96.65% / 33.00 | 96.10% / 181.00 | 93.10% / 3.45 | 95.45% / 4.70 | 96.00% / 3.79 | 96.40% / 52.67 |
| uniform-plugin-hidden | — | — | 92.05% / 33.00 | 92.90% / 181.00 | 81.70% / 10.06 | 87.85% / 42.98 | 91.60% / 106.59 | 91.80% / 180.00 |

Regular 32/180-step samplers may include an additional final readout; the table reports measured NFE. Continuous-only entropy commitment is an experimental sampling policy, shown separately below.

| Model | Experimental frozen EB bound | Success % / NFE |
|---|---:|---:|
| gaussian-hidden | 0.01 | 95.35% / 3.93 |
| gaussian-hidden | 0.1 | 92.85% / 3.41 |
| simplex-hidden | 0.01 | 96.70% / 4.83 |
| simplex-hidden | 0.1 | 94.05% / 4.11 |
| spherical-hidden | 0.01 | 96.60% / 3.68 |
| spherical-hidden | 0.1 | 93.45% / 3.22 |
