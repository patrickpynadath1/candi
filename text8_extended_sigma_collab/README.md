# Text8 extended blending-bandwidth sweep

This bundle contains the corrected positive-kernel Text8 sweep used to extend
the paper figure through bandwidths 3, 4, 6, and 8.

## Contents

- `data/`: aggregate sweep metadata for all nine bandwidths
- `reference/`: original `t8_sticky` results and CANDI/MDLM baselines
- `outputs/`: rendered PNG/PDF figure and the best-metrics CSV
- `scripts/plot_text8_extended_sigma_sweep.py`: standalone plotting script

Every corrected sweep evaluates NFE 32, 64, and 128 at temperatures
`0.1, 0.2, ..., 1.0`. The plotted point is the maximum valid-word fraction
over temperature, independently for word-length thresholds >=5 and >=6.

## Regenerate the figure

From the bundle root:

```bash
python3 scripts/plot_text8_extended_sigma_sweep.py
```

Requirements: Python 3 and Matplotlib.

The script writes:

- `outputs/text8_extended_sigma_sweep.png`
- `outputs/text8_extended_sigma_sweep.pdf`
- `outputs/text8_extended_sigma_best_metrics.csv`
