# Recorded reproduction and pilot results

These are historical measurements, not newly rerun benchmarks. Unless noted,
Sudoku evaluations use the 48,000/2,000 hard split (data seed 42), run/evaluation
seed 1, and all 2,000 validation puzzles. Cell accuracy includes answer-format
separators; digit accuracy does not. Different budgets, weight selection and
sampling policies make rows unsuitable as a controlled method leaderboard.

Generated artifacts and weights are ignored by Git and may live on a separate
cache disk through `core/outputs`. Artifact directory names below identify the
original evidence; they are not requirements for using the package. Historical
`/dev/shm` run paths were relocated to durable cache storage.

## MDLM imported-checkpoint reproduction

The 20,000-step reference checkpoint, EMA weights, batch 64, temperature 1 and
no filtering produced exact sequence agreement:

| Sampler | Reference and core solved | Cell accuracy | Mean NFE | Identical sequences |
| --- | ---: | ---: | ---: | ---: |
| Ancestral, 180 steps plus greedy final prediction | 910/2,000 (45.50%) | 90.4449% | 181 | 2,000/2,000 |
| EB, entropy bound .1, greedy commitments | 1,809/2,000 (90.45%) | 98.4888% | 3.1525 | 2,000/2,000 |

EB NFE ranged from 1 to 11. This comparison was repeated after sampler ownership
and Task extraction; all per-puzzle records matched. Controlled training loss
was 8.334506034851074 in both implementations, with maximum parameter-gradient
difference zero. BF16 raw-logit comparison also matched exactly. Embedding/rotary
preparation outside autocast and explicit inference logsumexp normalization
were necessary precision boundaries.

Fresh full-size training was separately exercised for 100 steps (final train
loss 2.383145, validation objective 2.335216). Neither that smoke nor imported
checkpoint parity establishes fresh 20,000-step training convergence. Compact
machine-readable historical evidence is in
[historical-summary.json](historical-summary.json); its older test
counts and method names describe those stages, not current totals.

Artifact directories: `baseline_eb_verified`, `baseline_ancestral_verified`,
`sampler_orchestration_parity`, and `task_sudoku_parity` under `core/outputs`.

## Gaussian diffusion pilot and solvers

A 29M-parameter one-hot ContinuousDiT (width 512, eight blocks/heads) trained
2,000 steps at batch 256, LR 3e−4, warmup 250, BF16 and EMA .999. Final logged
training loss was .2182. Rank lookup sigma bounds were .2–4, 1,000 points,
percentiles .01–.4; the effective Sudoku sigma range was approximately .3083–4.
All evaluations used temperature 1 without filtering or solver tuning.

| Weights / solver | Solved | Digit accuracy | Cell accuracy | NFE |
| --- | ---: | ---: | ---: | ---: |
| EMA / Euler, 180 steps | 394 (19.70%) | 84.38% | 85.79% | 181 |
| Final / Euler, 180 steps | 345 (17.25%) | 84.13% | 85.56% | 181 |
| EMA / Heun, 90 steps | 385 (19.25%) | 84.15% | — | 181 |
| EMA / Dormand–Prince, rtol 1e−3, atol 1e−4 | 356 (17.80%) | 83.69% | — | 75 |

One-hot representation extraction subsequently preserved every one of the 2,000
Euler evaluation records. Sphere-rank embedding schedule checks matched the
original over 12 dimension/coupling/clamp configurations and 1,001 times each.
A separate trainable 16-dimensional embedding model completed three CUDA steps
and four-example evaluations with all three solvers: execution, not quality.

These solvers use posterior-derived velocity. The new time-Euler equations,
prompt restoration and final decoding intentionally replace legacy continuous
sampling behavior. Adaptive step selection is batch-shared and counts rejected
stages; NFE and results can depend on batch size. Recorded times on shared A6000
hardware were 117.7s Euler, 117.7s Heun, and 52.3s ODE, not controlled speed
benchmarks. Artifacts: `gaussian_sudoku_pilot`, `gaussian_heun_final`,
`gaussian_ode_final`, `representation_onehot_parity` and
`representation_embedding_smoke`.

## MDLM hidden self-conditioning pilots

All runs used 2,000 steps, width 512/eight blocks, batch 256, BF16, original
2,500-step warmup, EMA evaluation and unchanged sampling settings. Ancestral
used 181 NFE; EB used entropy bound .1 and greedy commitments.

| Feedback implementation | Bootstrap probability | Final train loss | Ancestral solved | EB solved | EB mean NFE |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | — | .3159 | 0 (0%) | 394 (19.70%) | 22.762 |
| Historical unnormalized projected hidden | .5 | .6509 | 0 (0%) | 0 (0%) | 68.6995 |
| Historical L2-normalized projected hidden | .5 | .3392 | 8 (.40%) | 57 (2.85%) | 14.416 |
| Current zero-initialized affine LayerNorm, direct hidden addition | .9 | .2718 | 304 (15.20%) | 469 (23.45%) | 8.4925 |

The final row has ancestral cell accuracy 84.3826% and EB cell accuracy
88.1331%. These are single-seed early runs with multiple design differences,
not an isolated estimate of any one change's benefit. Earlier projected feedback
could grow dramatically under repeated reuse; its results must not be presented
as measurements of the current design.

The historical normalized-model 128-puzzle diagnostic solved 2 with reused
feedback, 7 with feedback disabled, and 10 with a fresh bootstrap per prediction
(versus 29 for the separate no-conditioning control). That experiment motivated
further work; it did not establish a sole cause or change the production sampler.
Artifacts: `mdlm_no_self_conditioning_sudoku_hard_control`,
`mdlm_hidden_self_conditioning_sudoku_hard`,
`mdlm_normalized_hidden_self_conditioning_sudoku_hard`, and
`mdlm_layernorm_hidden_self_conditioning_sudoku_hard`. The latter contains
`comparison.json` and durable model/EMA weights without optimizer state.

## Spherical flow matching and permuted autoregressive

The spherical Sudoku preset used trainable unit embeddings of dimension 512,
width 512/eight blocks, batch 256, LR .0003, warmup 2,500 and EMA .9999. It
retained physical-coordinate adaptation in this named preset, completed 19
refits by step 1,950, and trained for only 2,000 of the preset's 20,000 steps.
With 180 model calls it solved **200/2,000 (10.0%)**, cell accuracy 84.338%, digit
accuracy 82.791%, clue consistency 99.35%. The fixed 128-example check solved
20 (15.625%); do not substitute that subset rate for full validation.
Artifacts: `spherical_sudoku_hard_validation`.

Permuted AR was checked on Text8, not Sudoku: 3.2M parameters, width 256/four
blocks/eight heads, length 256, batch 128, LR .0003, 200-step warmup, EMA .999.
Training resumed from 3,000 to 10,000 steps. All 255 target positions were
generated with KV caching, random permutations, temperature 1 and no filters.
Over 128 EMA samples:

| Word-validity rate | Generated | Held-out |
| --- | ---: | ---: |
| All lengths | 68.20% | 99.01% |
| Length ≥3 | 60.03% | 98.80% |
| Length ≥5 | 29.90% | 97.97% |
| Length ≥7 | 9.07% | 96.92% |

Generated counts: 5,500 words, 3,751 valid, 2,995 distinct, 1,258 distinct valid.
The 3,000-step valid rate was 61.69%. Samples were accepted as an initial port
check, not a quality benchmark. Artifacts: `permuted_autoregressive_text8`.

## RePlaid and simplex

An earlier, superseded softplus-schedule RePlaid comparison solved zero Sudoku
puzzles after 2,000 steps in both original and core implementations. EMA cell
accuracy was 32.0051% original and 33.7635% core; final-weight accuracy was
29.7567% and 32.4410%. This did not reproduce the later successful reference.

After bringing in the corrected tanh learned schedule, scaling and gradient
handling, a fresh 2,000-step **final/raw-weight** evaluation solved
**31/2,000 (1.55%)**, with 77.1483% cell accuracy and 74.8914% digit accuracy.
Sampling used 180 Gaussian DDPM transitions, fresh self-conditioning bootstrap,
final-transition noise and 362 NFE. A held-out objective diagnostic totaled
.0908674 (prior .00998166, reconstruction 0, diffusion .0808857); it is not the
last training loss. Artifacts: `replaid_comparison/corrected_2k`.
The successful reference trained 50,000 steps on 200k/10k data, so this short
48k/2k run does not establish equivalent quality.

Simplex mathematical parity checks gave identical float64 log-Gamma, log-Beta,
log-Dirichlet and nine concentration/churn bridge outputs (`atol=rtol=0`).
Thirty-four focused tests passed with CUDA, including four forward/backward
and generation cases. Evidence: `simplex_port_validation/source_parity.json`.
Standalone trained Sudoku quality and the source's paper protocol were not
reproduced; the latter used different data size, serialization and EMA policy.

## Hybrid results

| Run | Budget / weights | Solved | Cell accuracy | Mean NFE |
| --- | --- | ---: | ---: | ---: |
| Historical CANDI recorded evaluation | 20k / EMA | 1,560 (78.00%) | 96.6978% | 3.493 |
| Same checkpoint imported into hybrid | 20k / EMA | 1,566 (78.30%) | 96.7371% | 3.503 |
| Fresh CANDI with standard masking | 2k / EMA | 746 (37.30%) | 89.7354% | 9.1495 |
| Fresh uniform + simplex | 2k / EMA | 237 (11.85%) | 82.8674% | 181 |

CANDI uses entropy bound .1 and greedy commitments, with no temperature tuning.
The imported result is close rather than bitwise parity. Fresh standard masking
intentionally fixes legacy accidental retention and changes training corruption.
Both fresh runs retained 2,500-step warmup. Uniform/simplex digit accuracy was
81.1753%; its independently tracked branches used 180 joint reverse steps plus
one final prediction. Artifacts: `hybrid_validation` and
`uniform_simplex_hybrid_2k`.

## OpenWebText pretrained point

The cached original `mdlm_orig_1024` checkpoint at 65,000 steps was evaluated
with EMA, width 768/12 layers/AdaLN, GPT-2-large scoring, seed 1, temperature .8,
and 32 samples of length 1,024. Ancestral sampling used 32 steps plus final
greedy denoising: **33 NFE**. Perplexity was **30.7164**, mean empirical token
entropy **4.92239 nats**, over **23,944 scored tokens**. The reference scorer
used the corrected initial BOS handling. Runtime was approximately 34 seconds
on an A6000. This demonstrates a real pretrained evaluation through the
frontier runner, not a multi-point or statistically stable frontier.

Artifacts: `frontiers/openwebtext_mdlm_pretrained`, point
`97030580c7d928f6`, including CSV/JSON reports, plot, config, logs and samples.
See [coverage](coverage.md) for distributed-training and resume checks.
