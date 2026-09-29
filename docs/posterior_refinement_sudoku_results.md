# Official Posterior Refinement on our Sudoku datasets

Official repository: https://github.com/MananAg007/posterior-refinement
Pinned revision: `b1dc5806b5655a60156e24233591151d8c53cac7`. Looped Flows was skipped at the user’s request because official code was unavailable.

Two from-scratch FMLM+ runs: 200,000 training examples each (easy and hard), 50,000 optimizer steps, batch 256, seed 1. The official mini DiT architecture, FMLM+ objective, optimizer and learning-rate settings are retained. Native Sudoku training uses PSD, uniform clean-context sampling, diagonal fraction 0.5 and boundary probability parameter 32.

Adaptations: existing dataset paths and sizes, 50k instead of the upstream script’s 20k steps, a single GPU, two loader workers, local logging, and checkpoints every 5k steps. No upstream Python source modifications. Missing dependencies are isolated in external/posterior-refinement-deps.

Inference exactly follows scripts/eval/sudoku.sh: EMA, posterior refinement, four rounds, one flow step per round, confidence threshold 0.999, uniform-budget fallback, fresh noise. Batch size 200. No EB or custom ancestral sampler and no inference-clock overrides. NFE is measured by a passive backbone hook.

Each trained checkpoint is evaluated on all 10,000 hard validation puzzles with 30, 16, 12, 8, and 0 clues. Thinning uses the same nested subsets as the previous sweep (seed 420000 + validation index). Targets are hidden from generation. Validity includes row/column/box rules, retained-clue consistency, and output format. Zero-clue evaluation also records distinct outputs and distinct valid grids.

[Queue status](/home/patrick/duo/outputs/posterior_refinement_sudoku_50k_20260926/status.json) · [Full plan](/home/patrick/duo/outputs/posterior_refinement_sudoku_50k_20260926/plan.json) · [Results CSV](/home/patrick/duo/outputs/posterior_refinement_sudoku_50k_20260926/summary.csv)

| Job | Status |
|---|---|
| train_easy | complete |
| eval_easy_30 | complete |
| eval_easy_16 | complete |
| eval_easy_12 | complete |
| eval_easy_8 | complete |
| eval_easy_0 | complete |
| train_hard | complete |
| eval_hard_30 | complete |
| eval_hard_16 | complete |
| eval_hard_12 | complete |
| eval_hard_8 | complete |
| eval_hard_0 | complete |

## Results

| Training | Clues | Exact accuracy | Valid solutions | Mean NFE | Distinct valid grids |
|---|---:|---:|---:|---:|---:|
| easy | 0 | 0.00% | 0.00% | 4.00 | 0 |
| easy | 12 | 0.00% | 0.22% | 4.00 | 22 |
| easy | 16 | 0.00% | 0.72% | 4.00 | 72 |
| easy | 30 | 67.45% | 67.45% | 4.00 | 6745 |
| easy | 8 | 0.00% | 0.13% | 4.00 | 13 |
| hard | 0 | 0.00% | 0.38% | 4.00 | 38 |
| hard | 12 | 0.00% | 1.70% | 4.00 | 170 |
| hard | 16 | 0.00% | 5.84% | 4.00 | 584 |
| hard | 30 | 78.04% | 78.04% | 4.00 | 7804 |
| hard | 8 | 0.00% | 0.68% | 4.00 | 68 |
