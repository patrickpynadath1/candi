# Parallel-view hybrid diffusion

`HybridDiffusion` corrupts one clean token sequence into independent branch
states. Each branch owns its local time map, schedule, representation, input
adapter, process, and reverse transition. `HybridModel` sums branch embeddings
and evaluates one shared transformer and clean-token posterior per joint step.

The `candi_hybrid` preset combines an exact masking branch and a rank-calibrated
one-hot Gaussian branch. Its input weights and frozen-posterior Gaussian update
preserve the legacy CANDI sampling architecture. `synchronize_committed` is an
explicit reference policy: after a mask token commits, its continuous view is
set to the same clean token before the next prediction. Other hybrid presets
leave branch states independent.

Legacy CANDI training redrew a uniform token and inferred whether a position was
revealed from equality with the clean token. That accidentally retained some
selected positions. The new preset intentionally fixes this bug: a selected
position always becomes the mask token, so its keep probability is exactly the
configured schedule. This makes old-checkpoint inference mapping useful for
sampling parity, but it is not an exact reproduction of the old training
distribution.

Evaluate the mapped historical checkpoint through the normal entrypoint:

```bash
python main.py +experiment=candi_hybrid mode=evaluate \
  evaluation.legacy=candi \
  evaluation.checkpoint=/path/to/candi/checkpoints/last.ckpt
```

## CANDI checkpoint evaluation

The 20,000-step original Sudoku-hard checkpoint was evaluated through the new
hybrid implementation using EMA weights, seed 1, 2,000 held-out puzzles, batches
of 64, and entropy-bounded sampling (bound 0.1, greedy commitment, 180-step limit).

| Metric | Original saved evaluation | New hybrid evaluation |
| --- | ---: | ---: |
| Exact solves | 1,560 / 2,000 | 1,566 / 2,000 |
| Accuracy | 78.00% | 78.30% |
| Token accuracy, including separators | 96.6978% | 96.7371% |
| Mean model evaluations | 3.493 | 3.503 |

These are imported-checkpoint inference results, not a fresh training run with
the corrected masking distribution. No temperature tuning was used. The small
inference difference remains; bitwise matching was not pursued further.
Results and resolved configuration are saved in
`core/outputs/hybrid_validation/`.

To evaluate an original checkpoint through the regular entrypoint:

```bash
python main.py +experiment=candi_hybrid mode=evaluate \
  evaluation.legacy=candi evaluation.checkpoint=/path/to/last.ckpt \
  evaluation.max_examples=2000 loader.eval_batch_size=64 \
  data.cache_dir=/path/to/data_cache
```

The importer explicitly maps the shared embedding and all trainable parameters,
checks tensor shapes, and reconstructs EMA order without counting the duplicated
embedding registration or rotary buffer. `evaluation.use_ema=false` selects the
saved final weights instead.

## Fresh standard-masking training check

A separate run trained from random initialization for 2,000 steps on the same
Sudoku-hard training data with standard masking. Full validation using its EMA
weights solved **746/2,000 puzzles (37.3%)**, with **89.74% token accuracy** and
**9.1495 mean NFE**, using the same entropy-bound 0.1 / greedy sampler. This
result is distinct from the imported 20,000-step checkpoint evaluation above.
The original 2,500-step learning-rate warmup was retained. The run and its
artifacts are described in `core/outputs/hybrid_validation/TRAINING_RUN.md`.
