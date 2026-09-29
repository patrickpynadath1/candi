# Training difficulty and fewer-clue Sudoku evaluation

## Existing 30-clue hard validation comparison

EB=0.01, 10,000 identical hard puzzles. CANDI and MDLM checkpoints are 50k steps; easy DUO is pinned at 45k, hard DUO is 50k. Hard-trained wins 13/14 EB comparisons.

| Model | Easy-trained accuracy | Hard-trained accuracy | Change (pp) |
|---|---:|---:|---:|
| candi_onehot_linear_fixed_hybrid | 94.78% | 97.76% | +2.98 |
| candi_onehot_linear_adaptive_hybrid | 85.61% | 84.90% | -0.71 |
| candi_onehot_geometric_fixed_hybrid | 91.80% | 95.30% | +3.50 |
| candi_onehot_geometric_adaptive_hybrid | 81.00% | 84.42% | +3.42 |
| candi_learned_linear_fixed_hybrid | 93.42% | 97.19% | +3.77 |
| candi_learned_linear_adaptive_hybrid | 80.71% | 90.03% | +9.32 |
| candi_learned_geometric_fixed_hybrid | 89.03% | 94.17% | +5.14 |
| candi_learned_geometric_adaptive_hybrid | 79.04% | 87.86% | +8.82 |
| candi_spherical_linear_fixed_hybrid | 94.32% | 97.30% | +2.98 |
| candi_spherical_linear_adaptive_hybrid | 78.15% | 85.19% | +7.04 |
| candi_spherical_geometric_fixed_hybrid | 92.11% | 95.96% | +3.85 |
| candi_spherical_geometric_adaptive_hybrid | 82.88% | 85.83% | +2.95 |
| mdlm_baseline | 95.50% | 96.66% | +1.16 |
| duo_baseline | 71.24% | 89.33% | +18.09 |

## Fewer-clue sweep

Launched September 25: 120 evaluations = 15 models × 2 training difficulties × 4 clue counts (16, 12, 8, 0). Each evaluates all 10,000 original hard validation examples after removing clues. EB=0.01 uses greedy predictions; S-FLM uses its ODE sampler. Sampling budget 180, batch 32, EMA enabled, original inference clocks retained. Easy S-FLM and DUO use the previously pinned 45k checkpoints; all other checkpoints are 50k.

For each example, original clue positions are shuffled using seed `420000 + validation index`. Smaller clue sets are nested subsets of larger ones, shared across every model. Solution targets remain unchanged. Zero-clue prompts are identical, but the evaluator still generates 10,000 samples; these are repeated generations from the empty prompt, not 10,000 distinct puzzles.

**Primary metric: valid_solution_rate** — all rows, columns, and boxes valid, retained clues respected, and correct output format. Exact match to the original solution is secondary because alternative completions are acceptable. Also record valid grid rate, clue consistency, and mean NFE. Each evaluation writes a `clue_audit.json` with prompt hash and its exact clue count.

- 16 clues, hard-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_16/hard/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_16/hard/summary.csv) (created when the queue starts / first result completes).
- 16 clues, easy-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_16/easy/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_16/easy/summary.csv) (created when the queue starts / first result completes).
- 12 clues, hard-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_12/hard/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_12/hard/summary.csv) (created when the queue starts / first result completes).
- 12 clues, easy-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_12/easy/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_12/easy/summary.csv) (created when the queue starts / first result completes).
- 8 clues, hard-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_8/hard/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_8/hard/summary.csv) (created when the queue starts / first result completes).
- 8 clues, easy-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_8/easy/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_8/easy/summary.csv) (created when the queue starts / first result completes).
- 0 clues, hard-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_0/hard/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_0/hard/summary.csv) (created when the queue starts / first result completes).
- 0 clues, easy-trained: [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_0/easy/status.json), [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/fewer_clues_10k/clues_0/easy/summary.csv) (created when the queue starts / first result completes).

## Completed fewer-clue results

All 120 evaluations completed. Percentages below are **valid completions respecting retained clues**, not exact matches to the saved solution. Each cell covers 10,000 generations. EB bound 0.01; S-FLM uses ODE.

### Hard-trained models

| Model | 16 clues | 12 clues | 8 clues | 0 clues | Mean NFE, 16 → 0 clues |
|---|---:|---:|---:|---:|---:|
| CANDI / onehot / linear / fixed | 99.95% | 99.99% | 99.97% | 99.98% | 37.10 → 53.73 |
| CANDI / onehot / linear / adaptive | 99.27% | 99.23% | 98.71% | 73.70% | 31.91 → 53.52 |
| CANDI / onehot / geometric / fixed | 99.00% | 99.04% | 99.00% | 96.91% | 35.04 → 50.31 |
| CANDI / onehot / geometric / adaptive | 98.71% | 98.20% | 97.09% | 86.88% | 29.20 → 53.73 |
| CANDI / learned / linear / fixed | 99.92% | 99.88% | 97.01% | 0.00% | 37.00 → 8.92 |
| CANDI / learned / linear / adaptive | 99.61% | 98.71% | 94.71% | 94.59% | 34.75 → 55.38 |
| CANDI / learned / geometric / fixed | 97.99% | 97.88% | 98.10% | 96.86% | 34.14 → 50.00 |
| CANDI / learned / geometric / adaptive | 98.14% | 75.20% | 58.25% | 47.95% | 34.86 → 55.28 |
| CANDI / spherical / linear / fixed | 99.87% | 99.90% | 99.84% | 99.84% | 36.31 → 52.86 |
| CANDI / spherical / linear / adaptive | 99.71% | 99.35% | 98.18% | 96.07% | 31.19 → 52.66 |
| CANDI / spherical / geometric / fixed | 98.99% | 99.19% | 99.15% | 98.28% | 33.66 → 49.46 |
| CANDI / spherical / geometric / adaptive | 99.51% | 99.30% | 98.12% | 92.89% | 31.43 → 53.99 |
| mdlm | 99.85% | 99.94% | 99.91% | 100.00% | 37.34 → 54.00 |
| sfm | 47.98% | 34.15% | 19.06% | 0.19% | 180.00 → 180.00 |
| duo | 43.98% | 30.09% | 19.43% | 0.65% | 47.92 → 58.15 |

### Easy-trained models

| Model | 16 clues | 12 clues | 8 clues | 0 clues | Mean NFE, 16 → 0 clues |
|---|---:|---:|---:|---:|---:|
| CANDI / onehot / linear / fixed | 99.75% | 99.74% | 99.39% | 98.25% | 38.72 → 57.67 |
| CANDI / onehot / linear / adaptive | 98.41% | 97.41% | 97.05% | 86.28% | 35.18 → 55.03 |
| CANDI / onehot / geometric / fixed | 95.57% | 95.40% | 94.30% | 88.26% | 36.93 → 54.66 |
| CANDI / onehot / geometric / adaptive | 98.93% | 98.56% | 98.01% | 94.06% | 36.55 → 58.47 |
| CANDI / learned / linear / fixed | 99.08% | 98.69% | 98.35% | 84.19% | 38.55 → 55.08 |
| CANDI / learned / linear / adaptive | 95.25% | 92.91% | 88.02% | 60.97% | 35.91 → 58.81 |
| CANDI / learned / geometric / fixed | 87.06% | 29.87% | 0.01% | 0.00% | 35.87 → 50.37 |
| CANDI / learned / geometric / adaptive | 96.82% | 95.19% | 91.62% | 0.00% | 34.91 → 4.57 |
| CANDI / spherical / linear / fixed | 98.39% | 97.93% | 98.19% | 97.01% | 36.48 → 52.14 |
| CANDI / spherical / linear / adaptive | 94.70% | 85.69% | 70.61% | 6.00% | 32.70 → 55.64 |
| CANDI / spherical / geometric / fixed | 96.43% | 95.75% | 94.64% | 74.18% | 33.56 → 49.50 |
| CANDI / spherical / geometric / adaptive | 98.43% | 97.06% | 94.36% | 72.58% | 34.46 → 56.47 |
| mdlm | 99.92% | 99.92% | 99.95% | 100.00% | 45.47 → 60.00 |
| sfm | 12.31% | 4.35% | 1.02% | 0.00% | 180.00 → 180.00 |
| duo | 34.61% | 15.64% | 4.76% | 0.08% | 78.36 → 88.48 |

Hard-trained one-hot / linear / fixed CANDI retains 99.98% validity without clues, at mean NFE 53.7325. Hard-trained MDLM reaches 100% at mean NFE 53.9984; output diversity must be considered separately. Some variants fail sharply without clues: hard-trained learned / linear / fixed CANDI falls from 99.92% at 16 clues to 0% at zero clues. Its saved outputs include malformed sequences of separator tokens.

Every run records zero exact matches to the original solution; the reported validity metric accepts other completions. Removing clues changes the task from recovering a constrained answer to finding any consistent completion. Zero-clue inputs are identical across all 10,000 examples.

## Independent verification and zero-clue diversity

Independently checked all 1,200,000 decoded completions against row, column, box, format, and retained-clue constraints. Every result agrees with its saved validity flag. [Audit and diversity counts](/home/patrick/duo/docs/sudoku_fewer_clues_audit.json). Distinct means distinct grid strings, not equivalence classes under Sudoku symmetries.

| Model | Training | Valid zero-clue samples | Distinct valid grids | Most frequent output count |
|---|---|---:|---:|---:|
| candi_learned_geometric_adaptive_hybrid | easy | 0 | 0 | 10,000 |
| candi_learned_geometric_fixed_hybrid | easy | 0 | 0 | 175 |
| candi_learned_linear_adaptive_hybrid | easy | 6,097 | 6,097 | 1 |
| candi_learned_linear_fixed_hybrid | easy | 8,419 | 8,419 | 1 |
| candi_onehot_geometric_adaptive_hybrid | easy | 9,406 | 9,406 | 1 |
| candi_onehot_geometric_fixed_hybrid | easy | 8,826 | 8,826 | 1 |
| candi_onehot_linear_adaptive_hybrid | easy | 8,628 | 8,628 | 1 |
| candi_onehot_linear_fixed_hybrid | easy | 9,825 | 9,825 | 1 |
| candi_spherical_geometric_adaptive_hybrid | easy | 7,258 | 7,258 | 1 |
| candi_spherical_geometric_fixed_hybrid | easy | 7,418 | 7,418 | 1 |
| candi_spherical_linear_adaptive_hybrid | easy | 600 | 600 | 1 |
| candi_spherical_linear_fixed_hybrid | easy | 9,701 | 9,701 | 1 |
| duo_baseline | easy | 8 | 8 | 1 |
| mdlm_baseline | easy | 10,000 | 1 | 10,000 |
| sfm_baseline | easy | 0 | 0 | 1 |
| candi_learned_geometric_adaptive_hybrid | hard | 4,795 | 4,795 | 1 |
| candi_learned_geometric_fixed_hybrid | hard | 9,686 | 9,686 | 1 |
| candi_learned_linear_adaptive_hybrid | hard | 9,459 | 9,458 | 2 |
| candi_learned_linear_fixed_hybrid | hard | 0 | 0 | 9,177 |
| candi_onehot_geometric_adaptive_hybrid | hard | 8,688 | 8,686 | 2 |
| candi_onehot_geometric_fixed_hybrid | hard | 9,691 | 9,691 | 1 |
| candi_onehot_linear_adaptive_hybrid | hard | 7,370 | 7,152 | 8 |
| candi_onehot_linear_fixed_hybrid | hard | 9,998 | 9,998 | 1 |
| candi_spherical_geometric_adaptive_hybrid | hard | 9,289 | 9,289 | 1 |
| candi_spherical_geometric_fixed_hybrid | hard | 9,828 | 9,828 | 1 |
| candi_spherical_linear_adaptive_hybrid | hard | 9,607 | 9,602 | 2 |
| candi_spherical_linear_fixed_hybrid | hard | 9,984 | 9,984 | 1 |
| duo_baseline | hard | 65 | 65 | 1 |
| mdlm_baseline | hard | 10,000 | 2 | 9,984 |
| sfm_baseline | hard | 19 | 19 | 1 |

At zero clues, easy-trained MDLM repeats one valid grid 10,000 times; hard-trained MDLM returns two grids (9,984 and 16 occurrences). Hard-trained one-hot / linear / fixed CANDI returns 10,000 distinct outputs, of which 9,998 are valid, at similar mean NFE (53.73 versus MDLM 54.00). Greedy sampling and identical empty prompts matter for interpreting this diversity comparison.
