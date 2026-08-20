# Blurring diffusion on Sudoku-hard

The implementation is split across the existing model, algorithm, and sampler
interfaces:

- `models/blurring_dit.py` provides a bidirectional DiT with a normalized small
  input table and an untied output table.
- `algos/blurring_diffusion.py` applies fixed DCT-II heat-kernel corruption to
  each generated span and trains with per-token cross-entropy.
- `samplers/blurring_ode.py` uses deterministic Gaussian transitions in DCT
  coordinates. Intermediate clean estimates are probability-weighted blends of
  the small input table; only the final state is classified.

For Sudoku, the 91-token puzzle prefix remains clean and the DCT acts on the
89-token answer. The DiT still attends bidirectionally over all 180 positions.

## Initial ablation

The complete initial matrix can be distributed evenly across any number of GPUs:

```bash
new_scripts/run_sudoku_blurring_ablation.sh 4 5 6 7
```

Before the full run, exercise every variant for two optimizer steps on a tiny
dataset (these are real training and conditional-generation runs):

```bash
new_scripts/run_sudoku_blurring_ablation.sh --smoke 4 5
```

Each GPU receives a round-robin queue and runs its assigned jobs sequentially.
Per-run logs are written below
`$DUO_STORAGE_ROOT/new_sudoku_runs/blurring_ablation_{smoke,full}`.

The commands below remain useful for launching individual variants.

The corrected geometric-schedule runs use the `geometric_reblur_` label prefix.
Checkpoints trained with the earlier linear schedule are not sampler-compatible
with the corrected trajectory and must not be used to evaluate this method.

Run live normalized input tables first:

```bash
new_scripts/train_sudoku_blurring_one.sh 0 dct_d2 2 dct live
new_scripts/train_sudoku_blurring_one.sh 1 dct_d4 4 dct live
new_scripts/train_sudoku_blurring_one.sh 2 dct_d8 8 dct live
new_scripts/train_sudoku_blurring_one.sh 3 dct_d16 16 dct live
```

For each input dimension, the load-bearing control changes only the positional
mode rates:

```bash
new_scripts/train_sudoku_blurring_one.sh 0 isotropic_d4 4 isotropic live
```

The fixed-table ablation is selected with `detached`:

```bash
new_scripts/train_sudoku_blurring_one.sh 0 dct_d4_detached 4 dct detached
```

After training, produce the exact-board, valid-grid, and cell-accuracy curve at
1 through 256 NFE:

```bash
new_scripts/eval_sudoku_blurring_curve.sh 0 dct_d4 4 dct
```

Compare DCT and isotropic runs at matched `d_in`, architecture, seed, training
steps, and NFE. Select `d_in` using the fixed validation subset before running
the full 2,000-puzzle curve. MDLM and AR use their existing Sudoku experiments
as the discrete and quality-ceiling baselines.
