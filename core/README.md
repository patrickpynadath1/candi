# Modular diffusion tasks

This package trains and evaluates MDLM through independent mathematical
components coordinated by Algo. It contains its own DiT, task data pipelines,
Lightning training wrapper, checkpoint and EMA handling, samplers, and Hydra
configuration. No legacy repository imports or configs are required.

The first slice supports ancestral sampling and entropy-bounded sampling (EB).
See [ARCHITECTURE.md](ARCHITECTURE.md) for component ownership and
[VALIDATION.md](VALIDATION.md) for the measured migration checks.

## Environment and entrypoint

Use Python 3.10 or later and the dependencies in `requirements.txt`. The existing
project environment can be used directly. FlashAttention is not required: the
ported noncausal DiT uses PyTorch SDPA and native rotary tensor operations.

Run commands from the directory containing `core/`:

```bash
python main.py --cfg job
```

Hydra configs live in `configs/`. There is one construction and
execution path; `mode=train`, `mode=evaluate`, and `mode=sample` select the
workflow. Sampling writes generated outputs and task metrics using the shared
evaluator. Hydra multirun supplies sweeps.

The root `main.py` and `frontier.py` are the command-line entrypoints. To move
the implementation, copy those runners, `core/`, `configs/`, `pyproject.toml`, and
`uv.lock`. Provide external datasets or checkpoints through their explicit
configuration fields. The former legacy implementation is available in Git history.

## Config-driven construction

Hydra constructs the component graph recursively. The training wrapper calls
`instantiate(config.algorithm)` and evaluation calls `instantiate(config.sampler)`.

`algorithm=mdlm` selects `configs/algorithm/mdlm.yaml`, which declares the
model, predictor, process, schedule, time sampler, and objective through config
references. Component groups remain independently selectable and overridable:

```bash
python main.py algorithm=mdlm model=dit process=absorbing \
  parameterization=posterior sampler=ancestral parameterization.top_p=0.95
```

To add an implementation, add its class and a YAML config in the corresponding
group with `_target_` and constructor arguments. Hydra imports only selected
targets. Constructors receive explicit dependencies and local settings, not
the entire config.

The `token_space` config declares model vocabulary size and absorbing-mask ID.
Sudoku appends a mask to its 12 data tokens; Text8 preserves its existing
35-token vocabulary and reserved mask ID 4.
## Task and data selection

`task=sudoku` selects `core.tasks.sudoku.SudokuTask` and its default Sudoku-hard
data config. Existing `data.cache_dir`, `data.difficulty`, and dataset-size
overrides still work. The entrypoint constructs the Task once and shares that
instance with the generic `TaskDataModule`, evaluator, and `TaskEvaluation`
callback. Their configs are `datamodule=task`, `evaluator=generation`, and
`callback=task`; they do not need to change when selecting another Task.

Task owns its tokenizer, data preparation, training/validation datasets, named
evaluation datasets, training collation, generation inputs, and metric
accumulator. The data vocabulary excludes process-specific corruption states.
Sudoku validates its 12-token, 180-position representation and identifies
`accuracy` on `validation` as its primary metric, maximized. This metric is exact
solution accuracy, preserving the historical baseline; validity, clue
consistency, and cell accuracy remain additional metrics.

The training wrapper consumes `input_ids`, `loss_mask`, and an optional
`corruption_mask`. Sudoku also supplies `prompt_mask` and `padding_mask`; empty
puzzle cells are not padding. Existing cached datasets retain their historical
`attention_mask` column, which Task translates at the batch boundary.

The shared generation evaluator asks Task for sanitized prompt tokens and
fixed-position masks, calls Algo and the selected sampler, and accumulates Task
metrics. Reference answers stay with the scorer. Task metrics aggregate across
examples, independent of batch boundaries. `SudokuRecord` and `SudokuAggregate`
in `core/tasks/sudoku_metrics.py` are typed dataclasses; the accumulator returns
`SudokuAggregate` containing `list[SudokuRecord]`. Dictionary conversion happens
only at the shared evaluator's reporting boundary. The evaluator adds NFE and timing;
it does not interpret Sudoku layouts or compute puzzle correctness.

Periodic evaluation logs the Task's metrics and writes results under
`task_callback/`. Set `callback.every_n_steps=0` to disable periodic evaluation.
Standalone results also record the primary metric's name, direction, and data
source. The existing `best.ckpt` remains selected by validation NLL, retaining
the baseline checkpoint policy; primary Task metrics are tracked separately.

To add another task, implement the methods in `core/tasks/base.py` and add its
Hydra task option. Sudoku, Text8, OpenWebText, and TinyGSM are available. Maze and Zebra remain
to be ported. HubPaths is excluded from this migration for now. The
current shared evaluator supports one generated sequence per example and
indexable evaluation datasets; additional evaluation modes will be added with
the tasks that require them.

## Text8

Select `task=text8` to use the character task with the current model settings,
or `+experiment=text8` for the legacy tiny architecture (128 tokens, width 256,
8 blocks, 8 heads, AdaLN enabled). The task alone defaults to 256 tokens.

```bash
python main.py +experiment=text8 \
  data.raw_dir=/path/to/text8/raw_data \
  data.cache_dir=/path/to/core_text8_cache
```

The raw directory accepts `text8.train.txt`, `text8.validation.txt`, and
`text8.test.txt`, or the official `text8.zip`. Set `data.download=true` to fetch
the archive when it is absent. Archive preparation uses the original 90M/5M/5M
character split. Existing legacy raw files can be reused; prepared caches are
self-contained, source-hashed, and keyed by sequence length and packing version.
The non-crop preprocessing preserves EOS insertion and BOS/EOS wrapping with
legacy groups of eight source chunks. It uses one deterministic preprocessing
shard; old multi-worker/cache-dependent chunk boundaries are not reproduced.
Random-crop and BPE variants are not included in this port.

Generation evaluation requires no likelihood from the generating method:

```bash
python main.py +experiment=text8 mode=evaluate \
  evaluation.checkpoint=/path/to/core/checkpoints/last.ckpt \
  data.raw_dir=/path/to/text8/raw_data \
  data.cache_dir=/path/to/core_text8_cache \
  task.num_eval_samples=128 'task.min_word_lengths=[3,5,7]'
```

`task.evaluation_split=validation` is the default; `test` is also supported.
The original loader used test data for its Text8 validation path; this Task
keeps training validation separate and makes generation's reference split
explicit. `task.evaluation_seed` fixes the reference subset. Generated samples
have no conditioning tokens and are paired with the same number of held-out
windows of equal token length. Actual character and special-token counts are
recorded because generated special tokens can change decoded character length.

Both sides are scored against a fixed vocabulary from the full raw **training**
split, even when loader limits are used for smoke tests. Vocabulary caching uses
the training source hash. No held-out words are added to that vocabulary.

`Text8Record` stores generated and held-out `Text8Sample` dataclasses. Each sample
contains text, character/special-token counts, and `WordCounts`:

- `total`: word occurrences, including repetitions.
- `valid`: occurrences found in the training vocabulary.
- `unique`: distinct words within the sample.
- `unique_valid`: distinct words found in the training vocabulary.

The same counts appear under `by_min_length`; thresholds are inclusive. A
minimum of 3 includes three-character words. Potentially truncated words at the
ends of character segments are excluded on both sides. Special tokens separate
segments instead of silently joining fragments into a word.

`Text8Aggregate` retains occurrence totals, corpus-wide unique counts (unions),
per-sample means, and derived rates for generated and held-out text. In `rates`,
`valid` is valid/total, `unique` is unique/total, and `unique_valid` is
unique_valid/unique. Undefined ratios are JSON null, with zero counts preserved.
The top-level primary metric `valid_word_rate` is the generated corpus's
valid/total ratio. Counts and diversity remain visible alongside it. Nested
aggregates are logged by the shared callback; NFE and runtime remain evaluator
statistics. Dataclasses are converted to dictionaries only for reporting.

Reusable word-count tools are in `core/tasks/metrics/text.py`. Text8 records and
aggregates are in `core/tasks/text8_metrics.py`. No external evaluator model,
perplexity calculation, or new dependency is required. This port validates
training and sampling execution; it does not claim pretrained Text8 checkpoint
parity or convergence. Legacy Text8 checkpoint import is not yet supported.

## OpenWebText

Install the optional language-task dependency in addition to the base requirements:

```bash
pip install -r core/requirements.txt -r core/tasks/requirements-language.txt
python main.py +experiment=openwebtext data.cache_dir=/path/to/owt_cache
```

`task=openwebtext` uses GPT2 tokenization, 1,024-token wrapped sequences, and
unconditional generation. The corpus defaults to `Skylion007/openwebtext` with
its final 100,000 documents reserved for validation. Documents receive EOS, then
groups of eight are packed with BOS/EOS. Local JSONL inputs with a `text` column
can be supplied through `data.files_train=[/path/train.jsonl]` and
`data.files_validation=[/path/valid.jsonl]`; existing encoded Dataset directories
use `data.prepared_train` and `data.prepared_validation`.

The original GPT2 layout is preserved: base vocabulary 50,257, added PAD and
absorbing mask both ID 50,257, and 50,258 model outputs. Wrapped training batches
have no padding positions. This intentional legacy layout differs from TinyGSM's
separate pad and mask IDs.

Generation evaluation does not load the training corpus:

```bash
python main.py +experiment=openwebtext mode=evaluate \
  evaluation.checkpoint=/path/to/core/checkpoint.ckpt \
  task.num_eval_samples=128 task.scorer.device=cuda
```

`CausalReferenceScorer` in `core/tasks/metrics/generative.py` is reusable and
independent of the generator. Its default reference is GPT2-large, configurable
through `task.scorer.model_name`. `task.scorer.local_files_only=true` restricts
reference loading to existing local files; `task.local_files_only=true` applies
to the generation tokenizer. CPU scoring is available with
`task.scorer.device=cpu`.

The metric mathematics match the original implementation:

- Generative perplexity is `exp(sum(masked next-token CE) / sum(scored tokens))`,
  rather than the mean of per-sample perplexities.
- Retokenization truncates to the configured generation length. Context chunks
  are nonoverlapping; each excludes its first token and scores through its first
  EOS, inclusive, subject to the attention mask. An EOS at position zero leaves
  no scored targets in that chunk.
- Entropy is the mean per-sample empirical token entropy in nats, including
  special tokens. It is not model predictive entropy.

`OpenWebTextRecord` stores text, entropy, NLL sum, and scored-token count;
`OpenWebTextAggregate` retains those sufficient statistics and the reference
model name. Zero scored tokens produce null perplexity; nonfinite results raise
an explicit error. Reference scoring retains its own precision rather than
inheriting the generator's autocast context. Legacy local auxiliary-model
checkpoint formats are not supported; HF-compatible or injected causal models
are supported.

## TinyGSM and GSM8K

`task=tinygsm` trains on TinyGSM question/code examples and evaluates generated
answers on the independent GSM8K dataset. Install the language dependency above.

```bash
python main.py +experiment=tinygsm data.cache_dir=/path/to/tinygsm_cache
python main.py +experiment=tinygsm mode=evaluate \
  evaluation.checkpoint=/path/to/core/checkpoint.ckpt \
  task.evaluation.scoring=code
```

The tokenizer is SmolLM-135M: 49,153 tokens including PAD ID 49,152, followed by
absorbing mask ID 49,153. Training preserves question/answer separation,
`data.train_on_prompt`, `data.train_on_pad`, filtering, wrapping, and seeded
train/validation splitting. Full streaming training is explicitly unsupported;
`data.max_examples` permits bounded preparation. Local training JSON records
(`question`, `code`) use `data.data_path`. Local GSM8K JSON records (`prompt`,
`response_ground_truth`) use `task.evaluation.data_path`.

GSM8K generation groups equal-length prompts, preserves per-example prompt
masks, and decodes completions up to their first EOS. Code scoring extracts the
`simple_math_problem` function, executes it in isolation, then uses its numeric
return value or captured numeric stdout. `task.evaluation.scoring=numeric`
selects answer extraction without code execution. Both retain the original
absolute tolerance of 0.001. Optional bootstrap statistics use
`task.evaluation.bootstrap_size` and `task.evaluation.seed`.

`core/tasks/sandbox.py` supplies reusable `execute_python(code, entrypoint=...,
timeout=...)` and typed `ExecutionResult`, independently of GSM8K extraction and
answer comparison. Code evaluation requires Bubblewrap (`bwrap`) and host support
for user namespaces. It uses a fresh isolated process per program, read-only
system libraries, an empty environment, no network namespace access, temporary
storage, and bounded CPU, memory, output, and wall time. Missing or failed
isolation is reported as `sandbox_error`; there is no unsandboxed fallback.
Check these diagnostics before interpreting code-evaluation accuracy.

`TinyGSMRecord` and `TinyGSMAggregate` retain correctness, parse/execution/timeout
and sandbox-error diagnostics, EOS/completion statistics, and bootstrap
accuracy uncertainty. Standalone GSM8K evaluation does not prepare TinyGSM
training data. Neither task port claims full-training convergence or historical
checkpoint-result parity; validation uses local fixtures and cached evaluator
models.

## Gaussian diffusion and representations

`GaussianDiffusion` combines a representation, Gaussian process, continuous DiT,
posterior parameterization, schedule, objective, and time sampler. All Gaussian
noise acts in the representation's continuous space. The process receives
already-encoded states and knows nothing about tokens, codebooks, or vocabulary.

Representations own `encode(tokens)`, `posterior_mean(probabilities)`, and
`decode(state)`:

| Representation | Encode | Posterior mean | Decode |
|---|---|---|---|
| One-hot | One-hot vectors | `p` | Argmax |
| Embedding | Lookup in codebook `E` | `p @ E` | Argmax of `state @ E.T` |

The embedding codebook is unit-normalized and fixed by default. Set
`representation.trainable=true` to optimize it; normalization is applied whenever
the table is used. It is saved in checkpoints as a buffer or parameter and moves
with the algorithm. Its dimension is independent of vocabulary and transformer
width. The DiT's learned input projection remains a separate architectural
component, not the representation codebook. Training uses unshaped token CE.

```bash
# One-hot representation with the generic Gaussian algorithm.
python main.py +experiment=gaussian_diffusion_sudoku

# Embedding representation, sphere-rank approximation, representation decoding.
python main.py +experiment=gaussian_diffusion_embedding_sudoku \
  representation.dimension=128

# Linear interpolation works with either representation.
python main.py +experiment=gaussian_diffusion_embedding_sudoku \
  noise_schedule=linear_interpolation
```

Schedules return `GaussianPath(alpha, sigma, dalpha_dt, dsigma_dt)` for
`x(t)=alpha(t)*encode(x0)+sigma(t)*epsilon`. Before training or evaluation, the
algo validates the schedule's declared requirements against the representation
kind, continuous dimension, vocabulary size, and normalization, and checks the
model's state/output dimensions. These are construction-time checks; components
do not hold references to each other.

| Schedule config | Compatible geometry | Path |
|---|---|---|
| `one_hot_rank` (alias `gaussian_rank`) | One-hot, matching vocabulary | alpha=1, rank-calibrated sigma |
| `embedding_sphere_rank` | Unit embeddings, matching continuous dimension | alpha=1-u, sigma=u |
| `linear_interpolation` | Either | alpha=1-t, sigma=t |

One-hot rank defaults remain 1,000 lookup points over sigma `[0.2,4.0]`, with
normalized rank percentiles `[0.01,0.4]` and the legacy training-time conversion
`(1-training_epsilon)*t`. Bounds and calibration buffers belong to the schedule.
The embedding schedule ports the original iid uniform unit-sphere approximation
and its Gauss-Jacobi quadrature. It is not recalibrated to a learned codebook.
Its default geometric rank interval is `[0.01,0.49]`, with linear boundary
segments of width 0.01 leading to exact clean/noise endpoints. The legacy rank
normalizations differ: one-hot uses `(V-1)/V` times pairwise outranking probability,
whereas the sphere schedule averages over competing coordinates. Their percentile
settings are not numerically interchangeable. `embedding_table_rank` remains
available as a diagnostic against an actual table.

Evolving encoder/decoder implementations can use the same representation
boundary when they supply a tokenwise continuous encoding, posterior-to-mean
mapping, and decoder. Contextual or sequence-compressing encoders need additional
mean-conversion and prompt-conditioning semantics; they are not implemented here.

Continuous solvers share initialization, posterior-to-velocity conversion,
prompt conditioning, final decoding, and NFE accounting. The process computes
`dalpha_dt*mean + (dsigma_dt/sigma)*(state-alpha*mean)`; solvers integrate in time.
They check for `PosteriorParameterization`; directly predicted velocity remains
deferred. Temperature, top-k, and top-p belong to the predictor and only affect
inference.

| Sampler config | Method | Defaults |
|---|---|---|
| `sampler=euler` | Euler | 180 steps, 181 NFE |
| `sampler=heun` | Heun | 90 steps, 181 NFE |
| `sampler=ode` | Dormand–Prince via `torchdiffeq` | rtol=1e-3, atol=1e-4 |

All solvers support both final-decoding policies:

- `sampler.decoding=posterior`: argmax of the final token posterior; the one-hot
  experiment default, preserving existing behavior.
- `sampler.decoding=representation`: decode the final predicted continuous clean
  mean through the representation; the embedding experiment default. This is
  `argmax((p @ E) @ E.T)` for embeddings and can differ from posterior argmax.

Both policies perform one final network call. Fixed-step samplers run from time
0.999 to 1e-5. Adaptive sampling uses the same endpoints, counts all field calls
including rejected steps, and reserves one call for decoding. `max_nfe=2000` and
`max_steps=1000` bound work; exhaustion raises an error. Each batch shares
adaptive steps, so batch size can affect NFE and numerical results. Prompts are
fixed at every internal stage regardless of representation.

```bash
python main.py +experiment=gaussian_diffusion_embedding_sudoku \
  mode=evaluate evaluation.checkpoint=/absolute/path/to/last.ckpt sampler=ode
```

Use the training representation, model, schedule, and EMA configuration when
loading a checkpoint. Existing one-hot weights load with the current generic
configuration; old saved configs referencing removed one-hot targets must be
updated to the generic targets. Switch representations or corruption paths by starting a new
training run. The earlier continuous port corrected legacy sigma endpoint
ordering and the sigma-squared divisor; no legacy continuous trajectory parity
is claimed. Measured results and regression checks are in `VALIDATION.md`.

## Baseline training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  +experiment=sudoku_hard \
  data.cache_dir=/path/to/data_cache \
  hydra.run.dir=core/outputs/mdlm_sudoku_hard
```

The default configuration matches the documented baseline recipe:

| Setting | Value |
| --- | --- |
| Dataset | Sudoku hard, seed 42, 48,000 train and 2,000 validation |
| Sequence | 180 tokens; 91 fixed prompt tokens and 89 solution tokens |
| Vocabulary | 12 task tokens plus absorbing mask ID 12 |
| DiT | 8 blocks, width 512, 8 heads, dropout 0.1 |
| Conditioning | No time conditioning; AdaLN disabled |
| Optimizer | AdamW, learning rate 0.0003, zero weight decay |
| Learning rate | 2,500 warmup steps, then constant |
| Effective batch | 256, accumulated over 4 microbatches |
| Duration | 20,000 optimizer steps |
| EMA | 0.9999, with the legacy update-count warmup |
| Noise | Log-linear keep rate, endpoint epsilon 0.001 |
| Training time | Antithetic uniform, lower bound 0.001 |
| Periodic generation | 128 validation puzzles every 1,000 optimizer steps |

The no-AdaLN choice is intentional. The saved baseline predates the legacy
config's newer `adaLN=true` default. The new defaults reproduce the saved
architecture rather than that later default.

The dataset cache uses the existing directory convention:
`sudoku_hard_train48000_valid2000_seed42`. An existing cache can be supplied as
data without depending on legacy code. If absent, the locally ported generator
creates and saves it. Generating all unique-solution puzzles can take substantial
time; `data.sudoku_num_workers` controls generation parallelism.

Training writes resolved `config.yaml`, CSV metrics, periodic checkpoints,
`checkpoints/last.ckpt`, the best validation-loss checkpoint, and subset-generation
records. Optional W&B logging can be enabled with:

```bash
python main.py '+wandb.project=core-mdlm' '+wandb.name=sudoku-hard'
```

## Full paired evaluation

Evaluate the same checkpoint with both samplers on all 2,000 validation puzzles:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --multirun \
  mode=evaluate sampler=ancestral,entropy_bounded \
  evaluation.checkpoint=/path/to/run/checkpoints/last.ckpt \
  data.cache_dir=/path/to/data_cache \
  hydra.sweep.dir=core/outputs/paired_evaluation
```

Each job writes `results.json` containing per-puzzle completions, exact accuracy,
cell and digit accuracy, Sudoku validity, clue consistency, and NFE. Cell
accuracy includes row separators to preserve the published metric; digit
accuracy excludes them. Evaluation uses EMA by default and seed 1. Use
`evaluation.use_ema=false` to evaluate raw weights.

The default ancestral trajectory matches the historical `ancestral_cache`
experiment: 180 reverse steps followed by one separate greedy denoising call,
for 181 NFE. The old implementation recomputed its prediction each step; this
port makes no prediction-cache speedup claim. Configure `sampler.final_denoise`
as `ancestral`, `greedy`, or `none`.

EB uses an entropy bound of 0.1 and greedy clean-token commitments. Acceptance
uses the full shaped posterior, before the greedy commitment projection. Its
`sampler.max_steps=180` is an evaluation cap, not a time-grid length. Setting
`parameterization.commitment=sample` enables categorical commitments instead.

Sampling temperature, top-k, and top-p are configured under `parameterization` and apply
only to inference. They are owned by the posterior object and cannot change training or validation loss. Avoid adding
these settings to the model or objective config.

For configurations that change architecture, load the resolved training config
and override only evaluation settings:

```bash
python main.py --config-path /absolute/path/to/run --config-name config \
  mode=evaluate evaluation.checkpoint=/absolute/path/to/run/checkpoints/last.ckpt \
  output_dir=/absolute/path/to/evaluation
```

The saved resolved config has no defaults list; to switch its sampler, use a
fresh default composition with the matching model settings or edit a copy of
the resolved sampler config. The default paired command above is intended for
the baseline architecture.

## Legacy checkpoint comparison

The importer accepts MDLM backbone and EMA weights, with strict parameter-shape
checks. It remaps the serialized Sudoku tokenizer locally, so even legacy
checkpoint evaluation works without the old packages present.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  mode=evaluate sampler=entropy_bounded evaluation.legacy=true \
  evaluation.checkpoint=/path/to/legacy/mdlm/checkpoints/last.ckpt \
  data.cache_dir=/path/to/data_cache \
  hydra.run.dir=core/outputs/imported_mdlm_eb
```

Use `sampler=ancestral` for the paired ancestral comparison. Legacy optimizer
and trainer-loop migration is not implemented. Native core checkpoints support
training resume, including optimizer, scheduler, EMA, and saved RNG state.

## Resume and sweeps

For pretrained evaluation grids, Pareto summaries, and plots, see
[Frontier sweeps](docs/FRONTIER_SWEEPS.md). For selectable learning-rate schedules
and DDP/FSDP presets, see [Training configurations](docs/TRAINING_CONFIGURATIONS.md).

```bash
python main.py \
  checkpoint.resume=/absolute/path/to/run/checkpoints/last.ckpt \
  data.cache_dir=/path/to/data_cache \
  hydra.run.dir=/absolute/path/to/run

python main.py --multirun seed=1,2,3 \
  data.cache_dir=/path/to/data_cache \
  hydra.sweep.dir=core/outputs/seed_sweep
```

Resume with the same architecture and optimization configuration. Resume
restores training state but does not promise bitwise continuation of the data
stream after an interruption inside an epoch; the ordinary shuffled DataLoader
does not persist a prefetch-aware cursor. Changing the number of devices also
changes random-stream partitioning.

For multiple GPUs, select visible devices and set `trainer.devices` and
`trainer.strategy=ddp`. Global batch size must divide evenly by the product of
world size and gradient accumulation. The launcher enables unused-parameter
handling because the baseline retains an unused time embedder for checkpoint
compatibility.

## Smoke tests

```bash
python main.py +experiment=smoke hydra.run.dir=core/outputs/smoke

python main.py +experiment=smoke mode=evaluate \
  evaluation.checkpoint=/absolute/path/to/core/outputs/smoke/checkpoints/last.ckpt \
  hydra.run.dir=core/outputs/smoke_evaluation

python -m pytest core/tests -q
```

The smoke experiment uses a tiny DiT, eight training puzzles, two validation
puzzles, two optimizer steps, and CPU execution. It verifies execution rather
than Sudoku quality. GPU regression checks run when CUDA is available.

## Sampling interface

The posterior exposes separate `training_posterior`, `sampling_posterior`, and
`greedy_posterior` methods. Temperature, top-k, top-p, and the EB commitment policy
are constructor settings under `parameterization`; the old top-level `shaping`
and `generation.greedy_commit` config fields have been removed. For example:

```bash
python main.py mode=evaluate evaluation.checkpoint=/path/to/model.ckpt \
  parameterization.temperature=0.8 parameterization.top_p=0.95 \
  parameterization.commitment=sample
```

Samplers orchestrate generation using the process, schedule, and predictor
available through Algo. They own initialization, time grids, reverse steps,
acceptance decisions, evaluation budgets, and final denoising. Algo connects the
network to the posterior through explicit prediction methods. EB keeps each
prediction locally and reuses it for entropy and commitment, with no extra
network call or hidden cache. Shaping settings remain solely on the posterior.

## Adaptive schedules

Select `noise_schedule=sflm_adaptive` or `noise_schedule=cdcd_adaptive` for
keep-rate algorithms. Gaussian diffusion uses `sflm_gaussian_adaptive` or
`cdcd_gaussian_adaptive`, with a separately configurable base path. S-FLM defaults
to the original sampled-time fitting coordinates; CDCD uses physical corruption
progress. See [adaptive schedule details](noise_schedules/adaptive/README.md) for
loss signals, configuration, checkpoint behavior, and boundary stabilization.

## Gaussian schedule options

`noise_schedule=one_hot_rank` (also `gaussian_rank`) now accepts:

- `noise_schedule.inversion=lookup` or `analytic`: interpolate the stored table,
  or invert the Gaussian CDF directly. The lookup is an approximation, not a
  separate exact formula. Both respect `sigma_min` and `sigma_max` (default
  0.2 and 4.0); the derivative is zero outside that interval.
- `noise_schedule.coupling=linear` or `geometric`: progression of target rank.
- `noise_schedule.normalization=vocabulary` or `competitors`: divide the expected
  outranking count by V or V-1. Defaults preserve the original vocabulary-scaled,
  linear lookup. Analytic inversion does not implicitly change normalization.

For example, append `noise_schedule=one_hot_rank
noise_schedule.inversion=analytic noise_schedule.coupling=geometric
noise_schedule.normalization=competitors` to a Gaussian experiment command.
Rank targets outside the available sigma interval saturate at its boundaries.

`noise_schedule=exponential_ve` supplies alpha=1 and
sigma(t)=sigma_min*(sigma_max/sigma_min)**t, with the corresponding derivative.
It works with one-hot and embedding representations and can serve as an adaptive
Gaussian wrapper's base. Its defaults are sigma_min=0.01 and sigma_max=10;
optional training_epsilon scales the training time once, as for the rank schedule.

`noise_schedule=embedding_sphere_rank` already supports `coupling=linear` and
`coupling=geometric`. These use the unit-sphere embedding rank approximation.
`noise_schedule=argmax_error` calibrates one-hot Gaussian noise to the probability
that argmax chooses the wrong token. It uses
P(error)=1-E[Phi(Z+1/sigma)**(V-1)] for standard-normal Z, evaluated using float64
Gauss-Hermite quadrature and inverted by a piecewise-linear lookup. This is
numerical calibration, distinct from expected outranking count.

The default sigma interval is [0.2, 4.0]. Null `error_min` and `error_max` use the
error probabilities attained at those endpoints; explicit values must be covered
by the sigma table. `coupling=linear` (default) or `geometric` controls progression
in error probability. `num_points=1000` and `quadrature_order=512` control lookup
and integration accuracy. Both table buffers are saved in checkpoints, and the
schedule returns the derivative of its actual interpolation for continuous
solvers. Only matching-vocabulary one-hot representations are supported: embedding
decoding error depends on codebook geometry and needs a separate calibration.

For example:

```bash
python main.py +experiment=gaussian_diffusion_sudoku \
  noise_schedule=argmax_error noise_schedule.coupling=linear
```

It also composes with adaptive Gaussian clocks via
`noise_schedule@noise_schedule.base=argmax_error`. Training retains alpha=1 and
uses the existing Gaussian process, posterior parameterization, and token CE.
As with other finite-sigma VE schedules, the sampler's Gaussian initial prior is
an approximation to the noisy data marginal, not an exact pure-noise endpoint.

## Composing Gaussian model, predictor, objective, and path choices

The additional model embedding transform and input projection are retained.
`model.input_preconditioning=gaussian|none` controls only coefficient scaling;
`model.time_conditioning` selects time, log sigma, effective noise, or log SNR
conditioning, alongside the original default. See
[model choices](models/GAUSSIAN_CHOICES.md).

`parameterization.softcap=30` applies `30*tanh(logits/30)` in both training and
inference, before constraints and inference-only temperature/top-k/top-p. The
null default disables it. See [predictor choices](parameterizations/README.md).

`objective.label_smoothing` defaults to zero. Select `objective=cross_entropy_power`
for an explicit time/noise weighting rule, or `objective=cross_entropy_learned`
for FLM-style learned weighting. The algo supplies time, alpha, and sigma without
putting loss mathematics into orchestration. Trainable objective parameters are
included in optimization, EMA and checkpoints. See [objective choices](objectives/README.md).

`noise_schedule=rescaled_gaussian` wraps a replaceable base Gaussian schedule;
`noise_schedule.path_type=ve|vp|linear` sets the coefficient normalization while
preserving effective noise sigma/alpha. The wrapper returns the full derivatives
of the rescaled path, so existing continuous samplers need no special branches.
Converting an exact pure-noise endpoint (alpha=0) to finite VE coordinates is
singular and is rejected. Equivalent paths still need consistent model
preconditioning/conditioning; finite-step numerical results need not be identical.

Example composition (configuration example, not a benchmarked recipe):

```bash
python main.py +experiment=gaussian_diffusion_embedding_sudoku \
  noise_schedule=rescaled_gaussian \
  noise_schedule@noise_schedule.base=exponential_ve \
  noise_schedule.path_type=vp \
  model.input_preconditioning=gaussian model.time_conditioning=log_snr \
  parameterization.softcap=30 \
  objective=cross_entropy_learned objective.label_smoothing=0.1
```

## Named Gaussian method experiments

Use `+experiment=onehot_gaussian`, `embedding_flow_matching`, `embedding_ve`,
`candi_embedding_pure_linear_rank`, `candi_embedding_pure_geometric_rank`,
`candi_rank_pure_linear`, `candi_rank_pure_geometric`, or the other named
[Gaussian presets](../configs/experiment/GAUSSIAN_PRESETS.md). The table documents
all 13 presets, configuration-only approximations, and the deliberately retained
sigma=4 cap for one-hot rank paths. Tasks remain independently configurable.

## D3PM absorbing and autoregressive models

```bash
python main.py +experiment=d3pm_absorb
python main.py +experiment=autoregressive
```

Both default to Sudoku Hard and support the existing task, trainer, checkpoint,
and generation evaluation interfaces. `+experiment=ar` is an alias for
`autoregressive`.

D3PM uses a discrete time sampler (`time_sampler.steps=1000`), the absorbing
process, and a log-linear keep schedule. Its objective includes the variational
term and a separate clean-input reconstruction forward pass, following the
original implementation. Its posterior retains the mask class and does not clamp
observed tokens. The ancestral sampler owns reverse sampling; its step count
defaults to the training step count but can be overridden independently.

Autoregressive training uses shifted next-token cross entropy and a causal DiT.
The sampler uses **KV caching by default**: it prefills the prompt once and then
passes one new token through the model at each step. Per-layer caches are
preallocated, use absolute rotary positions, and support mixed precision.
They belong to each generation call and are not checkpointed. Set
`sampler.use_cache=false` to use the uncached reference path.

Conditioning masks must describe contiguous prefixes; different prefix lengths
within a batch are grouped automatically. The task evaluator supplies tokenizer
BOS/EOS/pad IDs to the sampler; explicit sampler settings take precedence.
Unconditional generation requires a BOS ID and includes it in the returned
sequence. Generation stops at EOS and pads the remaining positions. Temperature,
top-k/top-p, softcap, and commitment remain predictor settings. For direct calls
to `algorithm.generate`, supply special-token IDs when constructing the sampler
or call its `configure_special_tokens` method first.

These are functional ports; training-quality parity has not been benchmarked.

## Spherical flow matching and permuted autoregressive

```bash
python main.py +experiment=sudoku_spherical_truncated_adaptive
python main.py +experiment=permuted_autoregressive_text8 data.download=true
```

Spherical flow matching combines unit-normalized trainable embeddings, SphereDiT,
spherical corruption, posterior prediction, and a manifold sampler. The
[spherical presets](../configs/experiment/SPHERICAL_PRESETS.md) preserve the
original Sudoku architecture and describe fixed, adaptive, and truncated-adaptive
schedules. The adaptive component defaults to original S-FLM time coordinates;
the named Sudoku presets explicitly preserve their physical-coordinate setting.

[Permuted autoregressive](docs/permuted_autoregressive.md) uses causal generation
in a sampled spatial order, with separate query/key rotary positions and KV
caching enabled by default. It reuses the AR predictor and cross entropy; it has
no diffusion process. Training and sampling order policies are independently
configurable. Generation fills all positions even if EOS is sampled early in the
generation order. Optional observed-token conditioning must be enabled for both
training and inference.
The Text8 preset supplies the 256-wide, four-layer, 10,000-step validation recipe;
`+experiment=permuted_autoregressive task=...` selects the generic method instead.
See [port validation](docs/PORT_VALIDATION.md) for the Sudoku results, Text8
generation metrics, training budgets, and saved samples.

## Self-conditioning

Enable feedback through configuration:

```bash
python main.py self_conditioning.enabled=true
python main.py self_conditioning.enabled=true self_conditioning.source=hidden
python main.py +experiment=sudoku_spherical_truncated_adaptive self_conditioning.enabled=true
```

The default source, `posterior_mean`, feeds the expected token embedding from
unshaped token probabilities. `hidden` feeds the previous final hidden state,
normalized by an affine LayerNorm in FP32 and added directly to the input.
The LayerNorm scale and bias start at zero; there is no hidden-feedback projection.
Expected-embedding feedback retains a zero-initialized projection for dimension matching.
Both use detached feedback. Expected embeddings
require a predictor that exposes token probabilities; unsupported predictor
configurations fail explicitly instead of interpreting arbitrary scores as
probabilities. Spherical expected embeddings are not renormalized to unit length.

During training, `self_conditioning.probability=0.9` independently selects examples
for a no-gradient bootstrap on the same corrupted input and time. Validation uses
`self_conditioning.evaluation_probability=1.0` by default. These are generic
defaults; original RePlaid uses a fixed-count training selection of
`ceil(0.25 * batch_size)`, which is a distinct policy for its eventual port.
Generation reuses previous predictions without extra bootstrap evaluations.

Both sources are integrated with MDLM ancestral/entropy-bounded sampling,
Gaussian Euler/Heun sampling, and spherical sampling. Other algorithm adapters
remain to be integrated; this is not a mathematical incompatibility. Adaptive
ODE sampling currently rejects enabled self-conditioning because rejected stages
and cached derivatives need explicit feedback handling. Heun holds feedback fixed
across both stages and updates it after a successful step.

Feedback belongs to each sampling call and is never carried between batches.
The disabled default preserves the original model parameter keys and forward
behavior; enabled checkpoints include the additional conditioning parameters.

### Continuous RePlaid

```bash
python main.py +experiment=replaid_sudoku_hard
```

The RePlaid preset uses trainable unit embeddings, a learned negative-log-SNR
schedule, adaptive reconstruction/diffusion allocation, the three-term likelihood
objective, explicit Gaussian output-prior correction, and Gaussian DDPM sampling.
Its fixed-count expected-embedding self-conditioning and separate gradient routes
reuse the shared prediction infrastructure. Other methods' defaults are unchanged.
See [the implementation and validation notes](docs/REPLAID.md) for component
ownership, precision policy, optimizer groups, masking, and reproduction limits.

### Simplex diffusion

`+experiment=sudoku_simplex` selects the copied simplex implementation: FP64
log-space Dirichlet transitions, expected-embedding input, posterior CE, and the
simplex bridge sampler. Variants `sudoku_simplex_argmax`,
`sudoku_simplex_self_conditioning`, and `sudoku_simplex_adaptive` select the input
adapter, shared hidden feedback, and binned adaptive time sampler respectively.
All use current core Sudoku Hard data. See [the port notes](docs/SIMPLEX_DIFFUSION.md)
for source provenance, component boundaries, optional guidance/confidence
sampling, validation, and differences from the source's paper protocol.
