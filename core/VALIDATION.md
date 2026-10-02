# MDLM migration validation

Validation performed on October 1, 2026, using PyTorch 2.5.1+cu121 and NVIDIA
RTX A6000 GPUs. The new implementation reproduces the recorded ancestral and
greedy entropy-bounded evaluations from the saved MDLM checkpoint. Fresh
training has been exercised for 100 optimizer steps with the full architecture;
a new 20,000-step training run has not been completed.

## Full validation set comparison

The imported checkpoint was trained for 20,000 steps on Sudoku hard. Both
evaluations use its EMA weights, seed 1, batch size 64, and all 2,000 validation
puzzles from `sudoku_hard_train48000_valid2000_seed42`.

| Sampler | Recorded solved | Core solved | Core cell accuracy | Mean NFE | Identical generated sequences |
| --- | --- | --- | --- | --- | --- |
| Ancestral, 180 steps plus final greedy | 910 / 2,000 (45.5%) | 910 / 2,000 (45.5%) | 90.4449% | 181 | 2,000 / 2,000 |
| EB, bound 0.1, greedy commitments | 1,809 / 2,000 (90.45%) | 1,809 / 2,000 (90.45%) | 98.4888% | 3.1525 | 2,000 / 2,000 |

The EB NFE range is 1–11. Cell accuracy includes the row separators, matching
the recorded evaluation. This verifies model import and inference equivalence,
not convergence of a freshly trained 20,000-step core model.

The original artifacts used for comparison were:

```text
/home/patrick/.cache/duo/new_sudoku_runs/hard_eb_0p1/mdlm/checkpoints/last.ckpt
/home/patrick/.cache/duo/new_eval_results/sudoku_hard_eb_0p1_greedy/mdlm/results.json
/home/patrick/.cache/duo/outputs/sudoku_hard_paired_samplers_20260919/mdlm/standard/results.json
```

These are evidence and optional import inputs, not package dependencies. New
full result records are in `outputs/baseline_eb_verified/results.json` and
`outputs/baseline_ancestral_verified/results.json` in the working checkout.
Generated run directories are ignored by Git. The compact measured summary is
retained in [validation_summary.json](validation_summary.json).

## Precision and training equivalence

A direct comparison of the copied and legacy DiT with identical weights and
inputs produced zero maximum raw-logit difference in BF16 inference. A controlled
training batch with identical corruption, dropout RNG, and weights produced:

```text
Legacy weighted loss:                 8.334506034851074
Core weighted loss:                   8.334506034851074
Maximum parameter gradient difference: 0.0
```

Two precision boundaries were required to recover this behavior:

- Prepare embeddings and rotary frequencies outside autocast. Computing the
  rotary einsum under BF16 autocast caches rounded angles and changes predictions.
- Normalize the inference posterior outside the network autocast context, using
  the legacy explicit logsumexp operation. Promote its result before entropy or
  reverse-transition arithmetic.

The training path retains its mixed-precision context. Sampling temperature,
top-k, top-p, and greedy projection are explicitly excluded from training and
validation losses. All parity comparisons use temperature 1.

## Executed training and infrastructure checks

- Full 25.3M-parameter model, hard-Sudoku cache, effective batch 256, BF16, and
  baseline optimizer settings: 100 fresh optimizer steps on one GPU, with
  validation, checkpoint save, EMA, and periodic EB generation. The final logged
  training loss was 2.383145 and validation NLL was 2.335216. These early losses
  are execution checks, not evidence of final Sudoku accuracy.
- Tiny-model two-GPU DDP training: completed with gradient accumulation,
  antithetic time partitioning, collective validation, rank-zero generation,
  and checkpoint saving.
- Native CPU checkpoint resume: optimizer steps and EMA updates advanced from
  2 to 4; the learning-rate scheduler resumed at step 4. Both evaluators loaded
  the resumed checkpoint and generated outputs.
- Hydra multirun: two seed jobs trained and saved separate outputs.
- Self-containment: copied only `core/` into a temporary parent directory,
  cleared `PYTHONPATH`, and successfully ran training sweeps and an imported
  baseline evaluation without the legacy repository modules available.

The baseline pickle contains an old `SudokuTokenizer` object. The local importer
maps that object to the core tokenizer rather than triggering an implicit import
of `tasks.sudoku`. Legacy training-loop and optimizer import is not supported.

## Posterior and sampler ownership refactor

The sampler now owns generation orchestration and calls the process, schedule,
and predictor available through Algo. Algo only connects model outputs to
parameterized predictions and manages evaluation mode during generation.

Both full validation evaluations were repeated after this change. All 2,000
per-puzzle records for each sampler exactly match the earlier core results,
including generated sequences and evaluation counts. Accuracy remains 90.45%
for EB and 45.5% for ancestral. The repeated artifacts are
`outputs/sampler_orchestration_parity/0/results.json` (EB) and
`outputs/sampler_orchestration_parity/1/results.json` (ancestral).

Additional regression tests cover filtering helpers, posterior-owned commitment
policies, sampler-owned process calls and final denoising, and reuse of the same
prediction for entropy and commitment without extra network evaluations.

## Config-driven construction

After removing the handwritten assembly module, the CUDA-enabled suite passed
all 28 tests. Coverage includes recursive algorithm construction, component and
vocabulary overrides, reconstruction from resolved checkpoint configuration,
selection of an alternate target, training resume, and both evaluators. The
full 2,000-puzzle comparisons above were run before this construction-only change.

## Task-independent entrypoint

The CUDA-enabled suite passed 30 tests after moving task validation into the
selected data module and making data modules, evaluators, and callbacks
config-driven. A non-Sudoku fixture trained and evaluated with vocabulary size 5
and sequence length 8 through the same entrypoint. Focused integration checks
also exercised the configured Sudoku generation callback, training resume, and
both standalone evaluators. Full baseline sampling was not repeated for this
entrypoint-only change.

## Sudoku Task abstraction

The Sudoku Task owns dataset preparation, tokenization, explicit training masks,
sanitized generation inputs, and typed metric accumulation (`SudokuRecord` and
`SudokuAggregate`). Data loading, generation evaluation, and periodic evaluation
use shared infrastructure and one Task instance.

Both full 2,000-puzzle evaluations were repeated with the original checkpoint:
EB solved 1,809 puzzles (90.45%, mean NFE 3.1525), and ancestral solved 910
(45.5%, mean NFE 181). Every generated sequence and reference metric matches the
original artifacts. All per-puzzle records also match the earlier core results.
Artifacts are `outputs/task_sudoku_parity/0/results.json` and
`outputs/task_sudoku_parity/1/results.json`.

The subsequent dataclass change was checked by rescoring all 4,000 generated
sequences with the typed accumulator. All record fields and aggregate metrics
matched exactly; the model was not rerun for this serialization-only change.

The final CUDA-enabled suite passed 34 tests, including exact loss and gradient
agreement before/after Task collation, prevention of answer leakage, metric
aggregation across unequal batch sizes, shared Task identity, and a non-Sudoku
Task using the same training/evaluation/callback infrastructure. This remains
checkpoint reproduction, not a newly completed 20,000-step training run.

## Text8 Task

The character Text8 port was checked against the original tokenizer and packing
helper: all 35 vocabulary IDs match, including mask ID 4, and 25 packed
128-token sequences from the real corpus match the original `_group_texts`
function exactly with the same groups of eight source chunks. The new cache
uses deterministic single-shard packing; legacy multi-worker sharding and
cache-dependent source chunking are not claimed to match.

The existing 90M/5M/5M raw corpus was prepared into source-hashed core caches.
A 19.2K-parameter smoke model completed two optimizer steps on an A6000 using
BF16, validation, checkpoint saving, and periodic generated/held-out word
metrics. Both ancestral and EB standalone evaluators then loaded that checkpoint
and produced four paired generated/reference records each. The training
vocabulary contains 239,973 distinct words under the documented boundary rule.
This is execution validation, not pretrained model parity or generation quality.

Artifacts are `outputs/text8_task_smoke/`, `outputs/text8_task_eval/0/results.json`
(EB), and `outputs/text8_task_eval/1/results.json` (ancestral). Metric tests cover
repeated valid words, corpus unions versus per-sample uniqueness, inclusive
length filters, missing denominators, boundary fragments, reserved tokens,
training-only vocabulary, and identical generated/held-out scoring rules.

## OpenWebText and TinyGSM Tasks

OpenWebText's six tests cover token-weighted corpus perplexity, empirical token
entropy, EOS/context boundaries, local data packing, and a two-step training,
checkpoint, callback, and standalone evaluation workflow. A controlled comparison
against the original metric methods matched total negative log likelihood
15.615314483642578 over eight scored tokens. The reference scorer was also
exercised with a locally cached GPT-2 model. This verifies metric mathematics and
execution, not pretrained diffusion model quality or full-corpus training.

TinyGSM's eight tests cover prompt preservation, grouping by prompt length,
completion parsing, numeric accuracy and bootstrap statistics, cache separation,
and sandbox execution. Finite training token IDs and attention masks matched the
legacy implementation on controlled train/validation fixtures. A locally cached
SmolLM tokenizer was used for a two-step CPU training/checkpoint/evaluation smoke
run. Standalone GSM8K evaluation succeeded without access to the training data.

Real Bubblewrap execution checks include arithmetic, timeout handling, rejected
imports, function return values, and stdout answer extraction. Code evaluation
fails closed when the sandbox is unavailable. TinyGSM streaming is not ported;
finite data and bounded subsets are supported. No full TinyGSM training run or
pretrained accuracy reproduction is claimed.

## Gaussian one-hot diffusion on Sudoku Hard

A 29M-parameter ContinuousDiT (hidden 512, eight blocks, eight heads) trained
for 2,000 optimizer steps on the existing 48,000/2,000 Sudoku Hard split
(data seed 42, run seed 1). Global batch size was 256, learning rate 3e-4,
warmup 250 steps, BF16, and EMA decay setting 0.999 with the existing warm-start
rule. Training took about 607 seconds on one RTX A6000. The final logged training
loss was 0.2182. This is a short pilot, not a converged-training claim.

The schedule lookup spans sigma 0.2 to 4.0 with 1,000 points and normalized
rank percentiles 0.01 to 0.4. For the Sudoku vocabulary, the effective sigma
range is approximately 0.3083 to 4.0. The algorithm uses an unshaped posterior
and unweighted CE. Evaluation used temperature 1, no top-k/top-p filtering,
and no tuning.

Final standalone evaluation used the generalized time-Euler sampler with 180
steps from time 0.999 to 1e-5, plus a final posterior evaluation (181 NFE):

| Weights | Exact puzzles | Digit accuracy | Cell accuracy | Valid grids | Clue consistency |
|---|---:|---:|---:|---:|---:|
| EMA | 394/2,000 (19.70%) | 84.38% | 85.79% | 20.60% | 96.90% |
| Raw | 345/2,000 (17.25%) | 84.13% | 85.56% | 17.75% | 98.20% |

Each full evaluation took approximately 118 seconds. Per-puzzle records are in
`outputs/gaussian_sudoku_pilot/evaluate_ema/results.json` and
`outputs/gaussian_sudoku_pilot/evaluate_raw/results.json`; the checkpoint is
`outputs/gaussian_sudoku_pilot/checkpoints/last.ckpt`. Commands, settings, and
scalar results are saved in `outputs/gaussian_sudoku_pilot/port_validation.json`.

All nine legacy algorithm utility functions matched on controlled fixtures.
Gaussian tests cover corruption, schedule derivatives and clamping, posterior
shaping isolation, single-example backward, fixed-RNG VE training equivalence,
and linear-path velocity and integration. The linear interpolation path was
smoke-tested for loss/backward/sampling, not trained for Sudoku quality.

Training began before the generalized schedule interface was added. Constant
alpha=1 preserves its forward corruption, model preconditioning, and loss;
fixed-RNG tests verify this. In-training callback results used the earlier
sigma-grid Euler implementation and are not the final results reported above.
The final time-Euler path intentionally replaces the legacy swapped sigma
arguments and sigma-squared divisor, preserves prompts at every step, and adds
an explicit final prediction. No legacy continuous-model trajectory parity is
claimed.

## Continuous solver comparison

The same 2,000-step checkpoint, EMA weights, seed 1, and full 2,000-puzzle
validation split were evaluated with one prespecified setting per solver.
Temperature remained 1, top-k 0, and top-p 1; there was no shaping or solver
hyperparameter search.

| Solver | Settings | Exact puzzles | Digit accuracy | Mean NFE | Seconds |
|---|---|---:|---:|---:|---:|
| Euler | 180 steps | 394/2,000 (19.70%) | 84.38% | 181 | 117.7 |
| Heun | 90 steps | 385/2,000 (19.25%) | 84.15% | 181 | 117.7 |
| Dormand–Prince | rtol=1e-3, atol=1e-4 | 356/2,000 (17.80%) | 83.69% | 75 | 52.3 |

All solvers consume velocity derived from the posterior. Directly predicted
velocity is deliberately deferred. The shared sampler checks the predictor type,
keeps prompts fixed at internal stages, and counts field calls plus final
posterior decoding. Adaptive budgets include rejected stages; reaching a budget
fails explicitly. Adaptive batches share step selection, so batch size can affect
NFE and numerical results. Observed runtime is from shared hardware, not a
controlled performance benchmark.

The Euler lifecycle refactor matched every intermediate state exactly in a
seeded reference-loop test. Additional tests cover analytic solver accuracy,
prompt preservation, posterior-only enforcement, NFE and integration-step
budgets, and nonfinite predictions. Full solver outputs are
`outputs/gaussian_heun_final/results.json` and
`outputs/gaussian_ode_final/results.json`. The combined settings and commands are
in `outputs/gaussian_sudoku_pilot/sampler_validation.json`.

## Representation extraction and embedding rank schedule

`GaussianDiffusion` and `GaussianProcess` now operate independently of one-hot
versus embedding geometry. The representation owns encoding, posterior means,
and decoding. Schedule compatibility is checked during algo construction:
one-hot rank rejects embeddings, sphere rank rejects one-hot/non-unit/mismatched
dimension embeddings, and linear interpolation accepts either representation.
Model state and vocabulary dimensions are also checked.

Before editing, a small one-hot model's state, fixed-randomness loss, all parameter
gradients, and sampled tokens were captured. After extraction, its original
config and state loaded strictly and all captured values matched exactly.
The existing 2,000-step Sudoku checkpoint was then evaluated on all 2,000 puzzles:
all per-puzzle records match exactly, retaining 394/2,000 (19.70%) and 181 NFE.
The new artifact is `outputs/representation_onehot_parity/results.json`.

The embedding sphere schedule was compared directly against the original
`SphereRankSchedule` across dimensions 2, 16, and 128; linear and geometric rank
couplings; and endpoint widths 0 and 0.01. All 12 configurations matched exactly
on 1,001 times each (maximum absolute difference 0). Derivative tests separately
compare the returned derivative to autograd; a random-codebook diagnostic checks
the sphere approximation against actual pairwise table ranks.

A 144K-parameter model with trainable 16-dimensional embeddings completed three
CUDA/BF16 training steps on the existing Sudoku Hard data, validation, a
representation-decoding callback, and checkpoint saving. Its checkpoint loaded
and evaluated four examples with each of Euler, Heun, and adaptive ODE, using
representation decoding. Artifacts are under
`outputs/representation_embedding_smoke/`. This verifies execution and checkpoint
handling, not embedding generation quality or convergence.

New tests cover codebook lookup/mean/decoding and gradients, construction-time
compatibility rejection, both representations with all three solvers, both final
decoding policies, checkpoint state round trips, unit geometry, and schedule
endpoints/derivatives. Learned contextual or sequence-compressing encoders are
not implemented; the representation interface currently supports tokenwise
continuous states.

## Regression coverage

The final CUDA-enabled suite passed all 97 tests. Ruff lint and formatting
checks also passed.

The tests under `tests/` exercise posterior support and shaping isolation,
weighted CE and gradients, empirical corruption and reverse probabilities,
schedule derivatives, antithetic partitions, prompt preservation, EB entropy
before greedy projection, evaluation budgets, checkpoint and EMA round trips,
training resume, both evaluators, and absence of legacy imports. A CUDA test
checks that rotary frequency caches remain FP32 under autocast.

Run the suite from the directory containing `core/`:

```bash
python -m pytest core/tests -q
```

Exact mid-epoch data-stream replay is not guaranteed by the current shuffled
DataLoader. Optimizer, scheduler, EMA, and RNG state resume are supported; a
prefetch-aware data cursor would be additional work if bitwise interrupted-run
continuation becomes a requirement.
