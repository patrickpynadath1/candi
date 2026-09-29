# Hard-trained Sudoku models evaluated on hard validation

Launched September 24, 2026. All 15 models use their final **50,000-step hard-trained checkpoints**, with EMA weights. Each evaluation uses the same **10,000 hard validation puzzles**, dataset seed 42, model/evaluation seed 1, batch size 32, and 180-step sampling budget. Adaptive models retain their learned inference schedules.

There are **57 evaluations**: 14 EB-capable models at each bound 0.01, 0.05, and 0.1 (42 runs), plus standard sampling for all 15 models. Accuracy and measured mean NFE are recorded in each queue's summary CSV as evaluations finish. Bounds do not imply equal NFE between models.

| Models | EB predictor | Standard predictor |
|---|---|---|
| CANDI one-hot, four path/schedule variants | `candi_entropy_bounded` | `candi` |
| CANDI learned/spherical embeddings, eight variants | `candi_embedding_entropy_bounded` | `candi_embedding` |
| MDLM | `entropy_bounded` | `ancestral` |
| DUO | `uniform_entropy_bounded` | `ancestral` |
| S-FLM | Unsupported | `sfm` (ODE) |

EB uses greedy clean estimates and greedy final denoising. Standard CANDI uses its native stochastic reverse sampler; the generic `ancestral` predictor is not compatible with CANDI. MDLM and DUO use the base ancestral predictor with ancestral final denoising. S-FLM uses its supported ODE sampler with greedy final decoding.

| Queue | Status | Completed results (created after first completion) |
|---|---|---|
| EB 0.01 | [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/eb_0p01/status.json) | [CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/eb_0p01/summary.csv) |
| EB 0.05 | [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/eb_0p05/status.json) | [CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/eb_0p05/summary.csv) |
| EB 0.1 | [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/eb_0p1/status.json) | [CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/eb_0p1/summary.csv) |
| Standard / ancestral / ODE | [status](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/standard/status.json) | [CSV](/home/patrick/duo/outputs/sudoku_candi_baselines_200k_50k_20260922/hard_to_hard_10k/standard/summary.csv) |

Each queue has a `plan.json`; each method has `evaluation.json` recording its checkpoint, exact command, GPU, and checkpoint step, plus `eval.log`, Hydra configuration, and eventually `results.json`. Runs use the campaign's frozen training source. Outputs are stored in `/home/patrick/.cache/duo/outputs`, accessible through `/home/patrick/duo/outputs`.

Checkpoint preflight verified global step 50,000 for every model. The resumed one-hot / linear / fixed CANDI run uses `last-v1.ckpt`; its older `last.ckpt` is step 15,000 and is excluded.

## Completed results

All **57/57 evaluations completed**, each covering 10,000 hard validation puzzles. Saved evaluation metadata confirms hard-trained checkpoints at step 50,000 and hard evaluation data for every run. Cells show **accuracy (%) / mean NFE**.

| Model | EB 0.01 | EB 0.05 | EB 0.1 | Standard |
|---|---:|---:|---:|---:|
| CANDI onehot / linear / fixed | 97.76 / 3.3896 | 96.91 / 2.8501 | 95.69 / 2.6575 | 65.89 / 180.0000 |
| CANDI onehot / linear / adaptive | 84.90 / 2.4601 | 76.81 / 2.2357 | 72.09 / 2.1321 | 75.95 / 180.0000 |
| CANDI onehot / geometric / fixed | 95.30 / 4.3217 | 93.74 / 3.4224 | 92.57 / 3.1248 | 58.02 / 180.0000 |
| CANDI onehot / geometric / adaptive | 84.42 / 2.6860 | 75.98 / 2.3803 | 71.42 / 2.2459 | 76.20 / 180.0000 |
| CANDI learned / linear / fixed | 97.19 / 3.4600 | 95.90 / 2.9090 | 94.31 / 2.7167 | 64.74 / 180.0000 |
| CANDI learned / linear / adaptive | 90.03 / 2.3836 | 82.88 / 2.1904 | 78.58 / 2.0990 | 79.40 / 180.0000 |
| CANDI learned / geometric / fixed | 94.17 / 4.8515 | 92.95 / 3.7897 | 91.69 / 3.4367 | 51.12 / 180.0000 |
| CANDI learned / geometric / adaptive | 87.86 / 3.0889 | 79.63 / 2.6889 | 74.29 / 2.5514 | 69.32 / 180.0000 |
| CANDI spherical / linear / fixed | 97.30 / 3.7592 | 96.13 / 3.1111 | 94.68 / 2.8712 | 63.13 / 180.0000 |
| CANDI spherical / linear / adaptive | 85.19 / 2.5780 | 76.63 / 2.3237 | 71.78 / 2.2178 | 75.11 / 180.0000 |
| CANDI spherical / geometric / fixed | 95.96 / 4.2628 | 94.50 / 3.3963 | 92.81 / 3.1078 | 56.20 / 180.0000 |
| CANDI spherical / geometric / adaptive | 85.83 / 2.6430 | 77.52 / 2.3668 | 72.40 / 2.2539 | 75.41 / 180.0000 |
| MDLM | 96.66 / 3.9468 | 94.95 / 3.5227 | 93.31 / 3.3376 | 34.11 / 180.0000 |
| S-FLM | — | — | — | 75.15 / 180.0000 |
| DUO | 89.33 / 17.2642 | 89.72 / 11.4524 | 88.32 / 9.5993 | 78.90 / 180.0000 |

Best result: **CANDI one-hot / linear / fixed, EB 0.01: 97.76% at 3.3896 mean NFE**. MDLM at the same bound scores 96.66% at 3.9468 mean NFE. This CANDI variant exceeds MDLM accuracy while using fewer NFE at each tested bound on hard→hard evaluation. This does not by itself explain the earlier easy→hard generalization gap.

All six fixed-schedule CANDI variants outperform their adaptive-trained counterparts under EB at each tested bound. Under native standard sampling, the adaptive variants outperform the corresponding fixed variants in all six pairs. EB versus standard changes both trajectory selection and clean-token sampling policy (greedy versus stochastic), so the accuracy difference cannot be attributed solely to EB step allocation.
