# Experimental continuous-only commitment sampling

These are not native Gaussian/spherical/simplex samplers and contain no discrete hybrid view. An external entropy-selected mask freezes continuous states at clean token representations. This changes the continuous process sampling dynamics. Retained for transparency, separately from the main comparison.

| Model | Bound | Solved (%) | Mean model calls |
|---|---:|---:|---:|
| gaussian-hidden | 0.01 | 95.35 | 3.93 |
| gaussian-hidden | 0.1 | 92.85 | 3.41 |
| gaussian-token | 0.01 | 28.80 | 11.58 |
| gaussian-token | 0.1 | 40.00 | 4.88 |
| simplex-hidden | 0.01 | 96.70 | 4.83 |
| simplex-hidden | 0.1 | 94.05 | 4.11 |
| simplex-token | 0.01 | 73.85 | 19.83 |
| simplex-token | 0.1 | 68.55 | 13.51 |
| spherical-hidden | 0.01 | 96.60 | 3.68 |
| spherical-hidden | 0.1 | 93.45 | 3.22 |
| spherical-token | 0.01 | 12.15 | 15.63 |
| spherical-token | 0.1 | 24.30 | 6.48 |
