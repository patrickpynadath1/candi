# Sudoku

[SudokuTask](../../core/tasks/sudoku.py) learns puzzle-to-solution generation.
The default [data config](../../configs/data/sudoku_hard.yaml) creates 48,000
training and 2,000 validation puzzles at hard difficulty with data seed 42.
The cache directory can point at an existing generated dataset:

```bash
uv run python main.py data.cache_dir=/path/to/data_cache
uv run python main.py mode=evaluate evaluation.checkpoint=/path/to/last.ckpt
```

Generation is implemented locally in [data/sudoku.py](../../core/data/sudoku.py).
Changing difficulty, counts, or data seed changes the dataset cache identity.
Model seed and `data.data_seed` are separate settings. The commonly used cache
name is `sudoku_hard_train48000_valid2000_seed42`.

## Tokens and conditioning

The task requires 12 vocabulary entries and sequence length 180. Tokens encode
digits, empty puzzle cells, row separators, and BOS. A formatted grid occupies
89 positions. Training sequences contain BOS, puzzle, BOS, solution: the first
91 positions form the fixed prompt, and the remaining 89 are the prediction
target. A method's appended mask token is a token-space choice rather than an
additional Sudoku symbol.

Training loss and corruption apply only to the solution. Evaluation replaces
all target positions with placeholders before handing the sequence to a
sampler. Preserving the prompt does not guarantee that the generated solution
agrees with the clues; those are separate checks.

## Metrics

[SudokuRecord and SudokuAggregate](../../core/tasks/sudoku_metrics.py) store:

- `accuracy`: fraction of generated answer sequences exactly matching the
  reference; also `num_correct` and `num_total`.
- `cell_accuracy`: token accuracy over the formatted answer, including row
  separators. `digit_accuracy` measures grid digits only.
- `valid_grid_rate`, and mean valid row/column/box fractions: Sudoku structural
  validity independent of reference equality.
- `clue_consistency_rate`: solutions respecting the given puzzle clues.
- `valid_solution_rate`: a valid grid, clue consistency, and correctly placed
  row separators.

Each record retains decoded generation, ground truth, and all per-puzzle
scores. Means are computed over examples, not over batches.

The imported 20,000-step MDLM checkpoint reproduced all 2,000 original sequences
for both ancestral and entropy-bounded sampling. This is inference parity;
fresh full-budget training reproduction has not been established. See
[measured results](../validation/reproduction-results.md).
