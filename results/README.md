# Results

**[Markdown results summary](SUMMARY.md)** — side-by-side model/sampler tables for normal hidden-representation self-conditioning, with links to the complete hint-removal tables.

## Sudoku Hard, 5,000 training steps

All runs use 48,000 training puzzles, training seed 1, and EMA checkpoints at step 5,000. The comparisons include standalone discrete methods, continuous methods, their hybrids, and feedback variants. `hidden` denotes normal hidden self-conditioning; `token` denotes intermediate token feedback. Detached token-feedback ablations use depths 3 and 0 (after transformer blocks 4 and 1).

- [Full-hint evaluation](sudoku-hard-5k/full-hints/report.md) · [CSV](sudoku-hard-5k/full-hints/summary.csv): 182 model/sampler evaluations on 2,000 validation puzzles, evaluation seed 1. Reports success rate and mean model calls (NFE).
- [Hint-removal evaluation](sudoku-hard-5k/hint-removal/report.md) · [CSV](sudoku-hard-5k/hint-removal/summary.csv): 910 completed evaluations, using 64 fixed validation puzzles × 32 generations with seed 1729. Retain nested subsets of 30, 23, 15, 8, or 0 clues.

All-hint cells report success / mean NFE. Reduced-hint cells report distinct valid boards divided by total generations, with repetitions counted once within each prompt's 32 samples. At zero hints, all prompts are identical, so uniqueness is pooled over all 2,048 samples. Validity requires Sudoku constraints and agreement with retained clues, not the original answer. Regular 32/180 refers to update steps; exact 4/8-NFE runs are labeled separately. Continuous-only EB is an experimental commitment policy.

Raw generations, logs, datasets, and checkpoints remain in the local output cache and are not committed. Refresh the hint-removal tables with `python examples/report_hint_removal.py` when those cached outputs are available. Sweep definitions are in `examples/`.
