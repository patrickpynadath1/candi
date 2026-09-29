# Hybrid entropy-bounded sampler evaluation

Checkpoint: corrected hybrid run `replaid-hybrid-rb-50k-20260929-163021`,
step 46000, EMA weights. No predicted self-conditioning (`self_condition_rate=0`).
Visible-token embeddings enter the context channel; they are not model predictions.

Evaluation used the same 128 held-out hard-Sudoku puzzles for each sampler,
selected without replacement with NumPy seed 42. Torch sampling seed was 42,
batch size 16, and the maximum clock grid had 180 steps. Full per-puzzle records
are in `analysis_results/replaid_hybrid_eb_comparison/`.

| Sampler | Solved | Solve rate | Mean per-puzzle NFE | Measured seconds |
|---|---:|---:|---:|---:|
| Ancestral | 87/128 | 67.97% | 180.00 | 28.89 |
| EB 0 | 127/128 | 99.22% | 18.12 | 42.68 |
| EB 0.1 | 119/128 | 92.97% | 2.28 | 4.08 |
| EB 1 | 86/128 | 67.19% | 1.96 | 3.19 |

W&B: https://wandb.ai/rz-text-dlp/replaid-sudoku/runs/4my9nzfd

EB chooses the lowest-entropy prefix whose summed entropy minus its largest
member is at most the bound, following the repository's existing EB convention.
Thus bound zero can still accept several numerically deterministic predictions.
The remaining continuous coordinates follow the fixed learned Gaussian clock.
Accepted coordinates become clean embeddings and remain fixed. EB is an inference
heuristic, not the exact reverse kernel used to derive the ELBO.

Per-puzzle NFE counts stop when a puzzle completes, but completed rows remain in
batched model evaluations until that batch finishes. The shared Python prefix
selector also synchronizes with the GPU. Consequently EB 0 reduced logical NFEs
but was slower in this measurement. GPU 5 was shared with training, so these
wall times are observational, not controlled throughput benchmarks.

This is a single-seed 128-puzzle comparison, not the full 10000-puzzle benchmark.
Implementation tests cover runtime registration, prompt preservation, acceptance
completion and public sampling. Reproduce using `scripts/eval_replaid_hybrid_eb.py`
with a fixed checkpoint, the cached Sudoku dataset, and an output directory.
