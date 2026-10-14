# N-Queens

The N-Queens task is a small, solver-generated conditional generation benchmark.
It supports 8×8 and 10×10 boards (92 and 724 solutions respectively, counting
rotations/reflections separately). It uses the common task, training, callback,
checkpoint and evaluation interfaces; no download or external solver is needed.

## Representation and correctness

A board is N column indices, one per row, using symbols 1 through N. Zero is an
unknown clue in the prompt, not padding. The sequence is `[N clues][N answer
columns]`; only the answer is corrupted and supervised. This representation
already enforces one queen per row. The checker independently requires all
columns to be distinct and both diagonal families to be non-conflicting.

`valid_solution_rate` is the primary metric: any valid board preserving all clues
counts, even if it differs from the stored reference. The report also includes
`valid_board_rate`, `clue_consistency_rate`, and diagnostic `exact_match_rate`.
Exact match is not the success criterion because prompts can have many solutions.
Saved records include prompt, generated board, reference, and all correctness
flags. Invalid token values and incorrect answer lengths fail validity.

## Data and splits

All solutions are enumerated locally. A deterministic, seed-controlled 80/20
solution split assigns reference boards to training/validation. Partial prompts
are deduplicated both within and across splits. By default, prompts reveal 2–4
queens, with 1,024 training and 128 validation examples. Counts, clue range, board
size, holdout fraction, generator version and data seed are part of cache identity.
An impossible requested count raises an error rather than silently duplicating
prompts. The cache is reused on later runs.

The held-out reference-solution split is stronger than randomly splitting clue
masks of the same reference board, but a validation prompt may admit an alternative
training solution. Correctness alone therefore does not demonstrate discovery of
unseen solutions. At N=8 there are only 92 solutions: this is a deliberately small
constraint/diversity benchmark, not evidence of broad reasoning generalization.

## Train and evaluate

```bash
# Default masked recipe, small model, 10k steps; solve-rate callback every 1k.
python main.py +experiment=nqueens output_dir=core/outputs/nqueens8

# Ten queens uses the correct size, vocabulary and sequence length together.
python main.py +experiment=nqueens10 output_dir=core/outputs/nqueens10

python main.py +experiment=nqueens mode=evaluate \
  evaluation.checkpoint=core/outputs/nqueens8/checkpoints/last.ckpt \
  output_dir=core/outputs/nqueens8-eval
```

The experiment recipe is an initial runnable budget, not a demonstrated convergence
result. It defaults to the framework's masked objective and entropy-bounded
sampler. Components can be overridden normally. Model/checkpoint architecture and
training data settings must match when evaluating. Set the normal `wandb` mapping
to enable the shared logging path; otherwise scalar reports are local CSV/JSON.
Metrics are namespaced `nqueens/`, including `nqueens/valid_solution_rate`.

## Clue-free generation and diversity

```bash
python main.py +experiment=nqueens mode=evaluate \
  evaluation.checkpoint=core/outputs/nqueens8/checkpoints/last.ckpt \
  task.evaluation_mode=unconditional task.samples_per_prompt=4 \
  output_dir=core/outputs/nqueens8-unconditional

# Repeated conditional samples: keep the original clues for each prompt.
python main.py +experiment=nqueens mode=evaluate \
  evaluation.checkpoint=core/outputs/nqueens8/checkpoints/last.ckpt \
  task.samples_per_prompt=16 output_dir=core/outputs/nqueens8-diversity
```

Clue-free evaluation zeros every prompt, with no reference answer exposed to the
sampler. It retains the normal held-out reference only for diagnostic exact-match
reporting. This evaluates a zero-clue condition absent from default training.
Use multiple sampling seeds and stochastic inference when measuring diversity.
The report includes:

- `unique_valid_solutions`: distinct successful boards across the entire run.
- `unique_valid_fraction`: unique successful boards divided by successful samples
  (zero when no sample succeeds).
- `mean_unique_valid_per_prompt`: mean number of different valid completions for
  each distinct clue pattern, including zero-success prompts.
- `solution_coverage`: distinct successful boards divided by all 92/724 labeled
  solutions. This is particularly meaningful for unconditional generation; it is
  not per-prompt conditional coverage and does not quotient board symmetries.

`task.samples_per_prompt` repeats evaluation requests, not training examples.
`evaluation.max_examples` limits total draws, including repeats. Periodic callback
subsampling can select only some repeats; use standalone evaluation with no limit
for controlled diversity comparisons. No claim about novel held-out solutions is
made by these aggregate diversity scores; records enable that additional analysis.

## CPU smoke check

```bash
OMP_NUM_THREADS=1 python main.py +experiment=smoke task=nqueens \
  data.cache_dir=/tmp/nqueens-smoke-cache output_dir=/tmp/nqueens-smoke \
  callback.every_n_steps=1 callback.subset_size=2 \
  sampler=ancestral sampler.steps=2

OMP_NUM_THREADS=1 python main.py +experiment=smoke task=nqueens mode=evaluate \
  data.cache_dir=/tmp/nqueens-smoke-cache output_dir=/tmp/nqueens-smoke-eval \
  evaluation.checkpoint=/tmp/nqueens-smoke/checkpoints/last.ckpt \
  task.evaluation_mode=unconditional sampler=ancestral sampler.steps=2
```

This checks the actual trainer, checkpoint loader and evaluator. It does not test
learning quality. N-Queens metrics accept insertion generations with explicit
lengths. An answer must still contain exactly one column per row to count as valid;
batch padding does not count as generated content.
