# Sudoku easy → hard generalization results

Recorded September 23, 2026. Campaign: `sudoku_candi_baselines_200k_50k_20260922`.

All 23 evaluations are complete: 15 primary methods, seven inference-schedule comparisons, and one additional DUO sampler comparison. Models trained on generated **easy** Sudoku and were evaluated on the same **10,000 hard validation puzzles**.

MDLM achieved **90.64%** exact puzzle accuracy. The best CANDI variant was one-hot / linear / fixed at **88.64%**. Switching adaptive-trained CANDI models to a fixed linear inference time schedule changed accuracy by at most **0.07 percentage points**. The primary DUO result uses the requested **ancestral sampler**.

## Data and evaluation protocol

| Setting | Value |
|---|---|
| Training data | 200,000 easy puzzle–solution pairs |
| Training validation data | 10,000 easy puzzles |
| Evaluation data | All 10,000 hard validation puzzles |
| Generator difficulty | Easy: 40 clues; hard: 30 clues |
| Dataset generation seed | 42 |
| Training and evaluation seed | 1 |
| Weights | EMA |
| Evaluation batch size | 32 |
| Sampling budget | 180 steps; EB samplers may finish earlier |
| CANDI / MDLM checkpoint step | 50,000 |
| S-FLM / DUO checkpoint step | 45,000, pinned copies of available checkpoints |
| Accuracy | Exact complete-solution match |

“Hard” refers to this repository’s generated 30-clue dataset, not Sudoku-Extreme or an externally rated difficulty benchmark. The hard training split was not used by these easy-trained checkpoints.

Every saved ground-truth answer in all 23 evaluations was verified against the hard validation cache, in the same order. The audit found **zero identical solution grids** shared between easy training and hard validation. This check does not establish non-equivalence under Sudoku symmetries.

### Samplers

| Models | Sampler | Settings |
|---|---|---|
| One-hot CANDI | `candi_entropy_bounded` | EB = 0.1; greedy clean-token predictions; greedy final denoising |
| Learned / spherical CANDI | `candi_embedding_entropy_bounded` | EB = 0.1; greedy clean-token predictions; greedy final denoising |
| MDLM | `entropy_bounded` | EB = 0.1; greedy clean-token predictions; greedy final denoising |
| S-FLM | `sfm` | ODE sampler; 180 steps; greedy final denoising |
| DUO, primary | `ancestral` | 180 steps; ancestral final denoising |
| DUO, additional comparison | `uniform_entropy_bounded` | EB = 0.1; greedy clean-token predictions |

DUO uses the curriculum-free `duo_base` objective. All CANDI variants are hybrid models. The learned-embedding linear path is `linear_rank`; the spherical linear path is `truncated_linear`. The geometric embedding paths use `geometric_rank`.

## Primary results: 15 methods

Mean NFE is the measured number of model evaluations per puzzle. These runs share a maximum sampling-step budget, but do **not** consume equal inference compute.

| Method (raw result) | Checkpoint step | Correct / 10,000 | Accuracy | Mean NFE |
|---|---:|---:|---:|---:|
| [MDLM](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/mdlm_baseline/results.json) | 50,000 | 9,064 | 90.64% | 3.7770 |
| [CANDI one-hot / linear / fixed](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_onehot_linear_fixed_hybrid/results.json) | 50,000 | 8,864 | 88.64% | 3.2908 |
| [CANDI spherical embeddings / linear / fixed](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_spherical_linear_fixed_hybrid/results.json) | 50,000 | 8,781 | 87.81% | 3.7973 |
| [CANDI learned embeddings / linear / fixed](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_learned_linear_fixed_hybrid/results.json) | 50,000 | 8,757 | 87.57% | 3.7257 |
| [CANDI one-hot / geometric / fixed](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_onehot_geometric_fixed_hybrid/results.json) | 50,000 | 8,637 | 86.37% | 4.1814 |
| [CANDI spherical embeddings / geometric / fixed](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_spherical_geometric_fixed_hybrid/results.json) | 50,000 | 8,532 | 85.32% | 3.7244 |
| [CANDI learned embeddings / geometric / fixed](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_learned_geometric_fixed_hybrid/results.json) | 50,000 | 8,178 | 81.78% | 4.0374 |
| [CANDI one-hot / linear / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_onehot_linear_adaptive_hybrid/results.json) | 50,000 | 6,827 | 68.27% | 2.4604 |
| [CANDI spherical embeddings / geometric / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_spherical_geometric_adaptive_hybrid/results.json) | 50,000 | 6,353 | 63.53% | 2.5256 |
| [CANDI learned embeddings / linear / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_learned_linear_adaptive_hybrid/results.json) | 50,000 | 6,167 | 61.67% | 2.6021 |
| [CANDI one-hot / geometric / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_onehot_geometric_adaptive_hybrid/results.json) | 50,000 | 5,929 | 59.29% | 2.7582 |
| [CANDI learned embeddings / geometric / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_learned_geometric_adaptive_hybrid/results.json) | 50,000 | 5,708 | 57.08% | 2.6397 |
| [CANDI spherical embeddings / linear / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/candi_spherical_linear_adaptive_hybrid/results.json) | 50,000 | 5,697 | 56.97% | 2.5810 |
| [DUO](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_duo_ancestral_10k/duo_baseline/results.json) | 45,000 | 5,638 | 56.38% | 180.0000 |
| [S-FLM](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/sfm_baseline/results.json) | 45,000 | 5,315 | 53.15% | 180.0000 |

## Inference-schedule comparison: seven adaptive-trained models

These comparisons use the **same checkpoint, EMA weights, hard puzzles, seed, batch size, sampler, and step budget** as the corresponding primary evaluation. After loading the checkpoint, the learned adaptive time remapping is bypassed and its underlying fixed linear schedule is used instead. For CANDI, this gives `alpha(t) = 1 - t`; S-FLM retains its base schedule’s truncation endpoints.

Each model’s linear/geometric corruption path stays unchanged. Thus “fixed linear inference” here changes the time remapping; it does not turn a geometric corruption path into a linear path. Saved `inference_schedule.json` files record both schedules.

| Adaptive-trained model (fixed-inference result) | Checkpoint step | Learned-clock accuracy | Fixed-linear-clock accuracy | Change (percentage points) | Correct with fixed clock | Mean NFE, learned → fixed |
|---|---:|---:|---:|---:|---:|---:|
| [CANDI learned embeddings / geometric / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/candi_learned_geometric_adaptive_hybrid/results.json) | 50,000 | 57.08% | 57.13% | +0.05 | 5,713 | 2.6397 → 2.6365 |
| [CANDI learned embeddings / linear / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/candi_learned_linear_adaptive_hybrid/results.json) | 50,000 | 61.67% | 61.74% | +0.07 | 6,174 | 2.6021 → 2.6014 |
| [CANDI one-hot / geometric / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/candi_onehot_geometric_adaptive_hybrid/results.json) | 50,000 | 59.29% | 59.28% | -0.01 | 5,928 | 2.7582 → 2.7639 |
| [CANDI one-hot / linear / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/candi_onehot_linear_adaptive_hybrid/results.json) | 50,000 | 68.27% | 68.20% | -0.07 | 6,820 | 2.4604 → 2.4629 |
| [CANDI spherical embeddings / geometric / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/candi_spherical_geometric_adaptive_hybrid/results.json) | 50,000 | 63.53% | 63.55% | +0.02 | 6,355 | 2.5256 → 2.5222 |
| [CANDI spherical embeddings / linear / adaptive](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/candi_spherical_linear_adaptive_hybrid/results.json) | 50,000 | 56.97% | 56.94% | -0.03 | 5,694 | 2.5810 → 2.5809 |
| [S-FLM](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/sfm_baseline/results.json) | 45,000 | 53.15% | 53.45% | +0.30 | 5,345 | 180.0000 → 180.0000 |

For all six CANDI models, the inference-only change left accuracy effectively unchanged (within ±0.07 percentage points). It did not recover the accuracy gap to the separately trained fixed-schedule models. S-FLM improved from 53.15% to 53.45% (+0.30 percentage points).

## Additional DUO sampler comparison

Both evaluations use the same pinned 45,000-step checkpoint and all 10,000 hard validation puzzles. The primary result is ancestral, as requested. This comparison changes both the sampler and its final-denoising policy, and the measured inference costs differ substantially.

| Sampler (raw result) | Correct / 10,000 | Accuracy | Mean NFE |
|---|---:|---:|---:|
| [ancestral](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_duo_ancestral_10k/duo_baseline/results.json) | 5,638 | 56.38% | 180.0000 |
| [uniform_entropy_bounded](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/duo_baseline/results.json) | 6,863 | 68.63% | 20.8733 |

## EB-bound sweep (launched September 23)

An additional 28 evaluations are queued: the 12 CANDI variants, MDLM, and DUO at each of **EB = 0.01 and 0.05**, compared with their existing EB = 0.1 results. S-FLM has no EB sampler. These runs retain the original inference clocks, EMA weights, greedy EB settings, 180-step limit, batch size 32, seed, and the same 10,000 **hard validation** puzzles. CANDI and MDLM use their 50,000-step easy-trained checkpoints; DUO uses the same pinned 45,000-step checkpoint as its previous EB evaluation. DUO's primary ancestral result remains in the main table.

At EB = 0.1, MDLM scores 90.64% at mean NFE 3.7770, while CANDI one-hot / linear / fixed scores 88.64% at mean NFE 3.2908. The difference is **0.4862 NFE** and 2.00 percentage points. The sweep measures accuracy versus actual mean NFE; using the same EB bound does not guarantee equal inference compute.

- [EB = 0.01 status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/eb_sweep/eb_0p01/status.json) · [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/eb_sweep/eb_0p01/summary.csv)
- [EB = 0.05 status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/eb_sweep/eb_0p05/status.json) · [results CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/eb_sweep/eb_0p05/summary.csv)

The CSVs update after each completed evaluation. Artifacts live under `/home/patrick/.cache/duo/outputs`, accessible through the existing repository `outputs` symlink.

## Interpretation and scope

These are single-seed results. CANDI and MDLM were evaluated at 50,000 training steps; S-FLM and DUO were evaluated at 45,000 steps, so their training durations are not matched. Fixed-trained CANDI variants outperform their adaptive-trained counterparts in all six embedding/path pairs under the tested greedy EB settings. The inference-clock ablation alone does not explain that difference or establish its training-time cause.

## Artifacts and reproduction

- [Primary results CSV (15 methods)](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/primary_results.csv)
- [Inference-clock comparisons CSV (seven pairs)](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/inference_clock_comparison.csv)
- [All evaluations CSV (23 results)](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/all_evaluations.csv)
- [Completed-results data audit](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/completed_results_data_audit.json)
- [Train/evaluation solution-overlap audit](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/data_overlap_audit.json)
- [Pinned baseline checkpoint manifest](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/available_baseline_checkpoints/manifest.json)
- [Training campaign](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/campaign.json)
- [Original evaluation plan](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_10k/plan.json)
- [Fixed-linear inference plan](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_fixed_linear_clock_10k/plan.json)
- [DUO ancestral evaluation plan](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/generalization/easy_to_hard_duo_ancestral_10k/plan.json)
- [Evaluation queue script](/home/patrick/duo/new_scripts/eval_sudoku_generalization.py)
- [Inference-clock override script](/home/patrick/duo/new_scripts/eval_sudoku_fixed_clock.py)

Each evaluation directory contains `results.json`, `evaluation.json` with the exact command and checkpoint step, the saved Hydra configuration, and `eval.log`. Evaluation uses the frozen training source at `outputs/sudoku_candi_baselines_200k_50k_20260922/source`. Absolute artifact links refer to the shared workspace.
