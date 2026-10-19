# Running an algorithm on a task battery

`examples/task_battery.py` composes an algorithm with tasks and seeds, then passes
ordinary `TrainingJob` and `InferenceJob` objects to the existing Ray queue.
Ray owns scheduling, GPU allocation, trial storage, and restoration.

The two inputs are:

- `configs/algorithm_definition/masked.yaml`: component choices, model settings,
  modifiers, and inference policies.
- `configs/battery/small_tasks.yaml`: tasks, data, budgets, seeds, and evaluation
  variants. Task settings cannot override the algorithm's components.

The example has five tasks and five seeds: 25 independently trained models per
algorithm. Set `seeds: [1]` for five runs. Inference policies reuse each model's
checkpoint. The current policies and extra N-Queens clue-free evaluation produce
90 evaluation jobs with five seeds. These are configurable example budgets.

```bash
# Preview only; no data preparation, model loading, or Ray startup.
uv run python examples/task_battery.py

# Explicit launch; choose available GPUs and a fresh experiment name.
CUDA_VISIBLE_DEVICES=0,1 uv run --extra mazes --extra tracking \
  python examples/task_battery.py --run --name masked-small-tasks \
  --max-concurrent 2 --wandb-project task-battery
```

Each run's existing `config.yaml` and checkpoint hyperparameters contain the
algorithm definition and its configuration ID. Periodic and final results carry
that ID, task name, and training seed under `benchmark`. W&B receives the same
configuration when enabled. The ID hashes the declared algorithm configuration;
it is not a code version. The normal resolved run config records the actual
component settings. Git remains responsible for source versioning.

There is no source scanner, archive, custom manifest, result index, or separate
battery executor. Use Ray's `ResultGrid` and existing per-trial results; restore
experiments with `tune.Tuner.restore` and the existing `run_job` trainable, as in
[programmatic sweeps](programmatic-sweeps.md).

Outputs, Ray temporary files, and W&B staging default to `~/.cache/duo/`.
Data preparation follows each task's configuration. Text8 and TinyStories keep
downloads disabled by default; configure prepared data before launching. See the
[task suite](evaluation-suite.md) and [story judge](story-judging.md) for setup.
Judging remains a separate optional pass over TinyStories results.

Insertion/deletion requires a task supporting variable-length generation. All five battery tasks expose that interface. Sudoku and N-Queens count answers
with the wrong generated length as invalid; Text8 scores only generated positions. The example battery has standard Sudoku evaluation, not a
clue-removal variant. No full battery run is launched by tests or previews.

`examples/full_battery_comparison.py` runs seven approximately 20M-parameter
methods across all five tasks with one seed and 10,000 steps: masked and uniform,
each alone, with self-conditioning, and with self-conditioning plus insertion;
and autoregressive. Preview by default; `--run` submits 35 training jobs and 102
evaluations to Ray. GPU visibility is selected with `CUDA_VISIBLE_DEVICES`.
Insertion uses its own sampled/greedy endpoint policies and reports truncation at
the task capacity. Fixed-length diffusion uses native, native greedy, and EB.

`examples/self_conditioning_battery.py` applies the original Sudoku corruption
recipes to all five tasks: blurring, adaptive Gaussian, default spherical, simplex,
masked and uniform, each with hidden self-conditioning off/on. It uses one seed,
approximately 20M parameters, 10,000 steps, and BF16 mixed precision across families. The
60 training jobs generate 216 final evaluations, including clue-free N-Queens.
All three inference policies also run every 1,000 optimizer steps.
Blurring uses `terminal_mean=0.01` consistently across the battery so that
1024-token sequences satisfy its terminal leakage bound; the original Sudoku-only
recipe used 0.03. Gaussian and spherical vocabulary references resolve per task.

## Adding jobs to a running sweep

### Multiple GPUs per Tune trial

Reserve a whole number of GPUs on one node with Tune's resources:

```python
trainable = tune.with_resources(run_job, {"cpu": 112, "gpu": 8})
```

The wrapper sets `trainer.devices` from this reservation. Two or more GPUs use
`torchrun` and DDP, inheriting Ray's GPU visibility; only the Tune actor publishes
metrics and checkpoints. Use the default strategy or `strategy=ddp`. Multi-node,
fractional GPU, and multi-GPU evaluation trials are not supported. Batch size and
gradient accumulation stay as configured, so check their divisibility when changing
GPU count. Set `max_concurrent_trials=1` to run a batch sequentially.

Trials resume their own `run/checkpoints/last.ckpt` on recovery. Restore interrupted
Tune experiments with `resume_unfinished=True, resume_errored=True` and the same
resource reservation and shared storage path.

`job_tinygsm_ray_head.sh` reserves the TinyGSM reference resources for seven days
and starts a loopback Ray Jobs server on port 8265. It submits
`scripts/tinygsm_ray_queue.py --root <persistent-directory>` to run the eight
TinyGSM configurations sequentially, each using eight GPUs. The script accepts
`--gpus`, `--cpus`, and `--experiments` for subsequent batches. Submit another batch
through `ray job submit` on the allocated node with a new `--root`; reuse a root to
restore its saved Tune experiment. The head stays alive until the allocation ends.

### Live single-node submissions

The self-conditioning battery keeps a durable submission queue open after its
initial jobs finish. Ray still schedules every training and evaluation worker;
the submission file only holds job definitions. It is stored at
`~/.cache/duo/batteries/submissions/<experiment-name>.sqlite` on the local disk.

Submit ordinary `TrainingJob` objects from a separate Python process:

```python
from pathlib import Path
from core.sweeps.submissions import SubmissionQueue

queue = SubmissionQueue(Path.home() / ".cache/duo/batteries/submissions/self-cond-full-battery-20m-bf16-live-20261012.sqlite")
queue.submit(additional_jobs)  # An iterable of TrainingJob objects.
```

Alternatively, save a YAML list containing the existing `TrainingJob` fields
(`name`, `experiment`/`overrides` or `config`, and optional `inference`) and run:

```bash
python -m core.sweeps.submissions /absolute/path/to/queue.sqlite submit jobs.yaml
python -m core.sweeps.submissions /absolute/path/to/queue.sqlite status
# Optional: stop accepting new submissions, finish accepted work, then exit.
python -m core.sweeps.submissions /absolute/path/to/queue.sqlite close
```

Submissions are atomic and persist before the command returns. Resubmitting the
same name and identical configuration is harmless; conflicting definitions with
the same name reject the whole batch. Use a new name for a changed experiment.
The status command reports accepted submissions; Ray reports their execution.
Keep the queue on local storage, not an object store or network filesystem.

The driver stays alive while submissions are open, including when no jobs remain.
To recover after a driver interruption, run the launcher with
`--run --restore --name <experiment-name>`. Ray restores queued and completed work;
interrupted training resumes from its most recent local `last.ckpt` when available.
A job without a saved checkpoint starts again. No restart is needed to submit jobs.
Changing worker code or cluster resources is separate from submitting a new config.

## Insertion gap-token supervision ablation

`examples/insertion_head_sweep.py` builds 24 training jobs: masked/uniform ×
gap-token head off/on × hidden self-conditioning off/on × Text8/TinyStories/maze.
All use one seed, approximately 20M parameters, BF16, and 10,000 optimizer steps.
Masked births are masks; uniform births are uniform noise in both head settings.
The head is auxiliary: head-off removes its parameters and gap-token CE, while
keeping the same count loss and insertion law. The gap target is a bag of missing
tokens, not an ordered continuation.

Every 1,000 steps and at completion, the insertion sampler is evaluated with
127 transitions plus final decoding (128 model calls), using sampled or greedy
final decoding. Insertion counts remain stochastic in both policies. There are
48 final evaluation jobs. Length limits and truncation reporting are identical
across ablation arms; the fixed-length EB sampler is not used for insertion.
W&B project: `self-cond-sweep`, group: `insertion-gap-head-20m-20261012`.

```bash
# Preview configurations and parameter counts.
python examples/insertion_head_sweep.py
# Append to the running live Ray queue; no restart.
python examples/insertion_head_sweep.py --submit
```

Existing fixed-length masked/uniform runs provide separate no-insertion baselines.

## Current three-task run

The replacement run excludes mazes and N-Queens and starts from scratch in W&B
project `self-cond-three-tasks-20261012`. It includes the 12 corruption/SC
configurations and eight insertion/head/SC configurations on each of Sudoku hard,
Text8 and TinyStories: 60 training runs and 156 final evaluations, seed 1 only.
The existing 10,000-step, BF16, roughly 20M-parameter budgets are retained.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,5 python examples/self_conditioning_battery.py \
  --run --name three-tasks-20m-bf16-20261012 \
  --tasks sudoku_hard text8 tinystories \
  --wandb-project self-cond-three-tasks-20261012 --include-insertion
```

The live queue is
`~/.cache/duo/batteries/submissions/three-tasks-20m-bf16-20261012.sqlite`.
Previous results remain in their original directories and projects.

## Recovering the reporting and blurring failures

`examples/recover_failed_sweep.py` submits `/repair-v1` replacements for the two
recognized failure signatures: boolean CSV parsing after training, and blurring
span selection during self-conditioning. Completed insertion checkpoints resume
at the existing step budget, so their optimizer steps are not repeated; final
checkpoint publication then releases their evaluation jobs. A failure without a
checkpoint restarts training. Replacements retain the algorithm identity and
record the original Ray trial under `benchmark.recovery_of`.

```bash
python examples/recover_failed_sweep.py \
  ~/.cache/duo/batteries/three-tasks-20m-bf16-20261012 \
  --queue ~/.cache/duo/batteries/submissions/three-tasks-20m-bf16-20261012.sqlite \
  --watch
```

`--watch` catches the same reporting failure in original workers that imported
the old code before the fix. It exits once the original training jobs are all
terminal. Submission is idempotent; unfamiliar failures stop recovery for review.
Historical failed trials remain visible; recovered trials and evaluations have
separate names. No healthy trial is stopped.

## TinyGSM compilation and performance profiles

The queue can apply an override document to both new jobs and restored Tune
snapshots using `--config-overrides`. The profiles are opt-in; loading an old
checkpoint does not silently enable compilation.

- `configs/performance/tinygsm_compile.yaml`: compile the backbone in place,
  compile the uniform objective, skip unused vocabulary projection in hidden
  self-conditioning bootstrap passes, and use fused AdamW. Model/EMA checkpoint
  keys and optimizer parameter order are preserved. BF16 fused AdamW clipping
  explicitly preserves the configured norm/value clipping because the installed
  Lightning AMP plugin otherwise rejects this optimizer.
- `configs/performance/tinygsm_compile_memory.yaml`: additionally recompute
  transformer block activations during backward. RNG state is preserved for
  dropout. Eager checkpointing disables early-stop because the legacy
  TorchScript residual helpers wrap its internal control-flow exception.

Both profiles use four loader workers per rank and validation batches of 16
instead of 64, with the validation batch cap increased from 500 to 2,000 to
preserve the same maximum example count. They retain global training batch 512,
accumulation 8, BF16, EMA, and the existing generation/evaluation settings.
Smaller validation batches reduce allocator reservation spikes; `nvidia-smi`
includes reserved cache and is not the same as peak live tensor memory.

Full-size synthetic-batch H100 measurements on PyTorch 2.5.1+cu121 (2026-10-17),
sequence length 512, microbatch 8, self-conditioning enabled, optimizer/EMA every
8 microbatches, excluding compilation warmup:

| Model | Eager ms/microbatch | Compiled speed profile | Compiled memory profile | Peak live GiB: eager / speed / memory |
|---|---:|---:|---:|---:|
| Masked | 50.3 | 29.0 | 33.1 | 9.66 / 8.49 / 6.83 |
| Uniform | 60.4 | 28.9 | 31.8 | 9.65 / 8.12 / 6.44 |

These measure loss, backward, optimizer, and EMA on a single GPU, not full-job
throughput or dataset-loading/DDP overhead. Compilation warmup took roughly
1–2 minutes for these profiles; new shapes and validation paths can compile
additional graphs. Inductor can change dropout's random-number stream, so
compiled and eager training trajectories are not bitwise identical.

The profiler confirmed native FlashAttention forward/backward. Forcing cuDNN
attention failed with an unsupported execution-plan error on this installed
stack, so the profiles retain automatic SDPA dispatch. PyTorch documents backend
selection through [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

Reproduce an isolated GPU benchmark (on an allocated GPU, with `PYTHONPATH` set
and `RCAC_SCRATCH` available):

```bash
python scripts/benchmark_tinygsm_performance.py \
  --experiment tinygsm_uniform_sc --variant optimized --batch 8 \
  --iterations 64 --output /tmp/uniform-compiled.json
```

For a **new** queue, two independent four-GPU jobs can be requested with:

```bash
python scripts/tinygsm_ray_queue.py --root /path/to/new-queue \
  --gpus 4 --cpus 56 --max-concurrent-trials 2 \
  --config-overrides configs/performance/tinygsm_compile.yaml
```

With global batch 512 and accumulation 8, four GPUs imply microbatch 16 per GPU.
Do not reuse an existing queue root to change concurrency: Tune restores its
saved scheduler configuration. Select unfinished experiments and explicit resume
checkpoints when constructing a replacement queue; the example starts new runs.
No production training queue is started by the benchmarks or profiles themselves.

Uniform's compiled microbatch sweep measured 142k/176k/203k/216k tokens/sec per
GPU at microbatches 8/16/32/64, with peak live memory 8.1/12.8/22.1/40.6 GiB.
A single-GPU uniform run therefore fits global batch 512 with accumulation 8 in
this synthetic training benchmark. That is not yet a capacity guarantee for
Gaussian/insertion workloads or their generation callbacks. Two four-GPU jobs
are a conservative starting configuration; validate actual data/evaluation and
aggregate throughput before increasing concurrency.

Validation included all eight TinyGSM configurations through AOT compilation,
full-size two-GPU Inductor training with EMA validation and checkpoint resume,
and BF16 CUDA eager/compiled numerical comparisons (relative gradient L2 error
below 0.05% for masked and uniform). Existing algorithm/SC tests passed. Detailed
local measurements are in `logs/tinygsm-performance-summary.json`.

### Partition-equivalence audit

Changing GPU count is **not strictly objective-equivalent** in the current
training implementation, even with global batch 512 and microbatch 8 preserved.
The returned loss is normalized by each microbatch's target-token count, then
Lightning averages accumulated gradients and DDP averages ranks. TinyGSM excludes
variable-length prompts from the loss. DistributedSampler regroups examples when
world size changes, so unequal token counts can change example weights. Equal
counts are equivalent; matching global batch alone is insufficient.

Uniform antithetic sampling still covers all 512 global quantile strata for
8 ranks x 8 accumulation x 8 examples, 4 x 8 x 16, and 4 x 16 x 8. Adaptive
loss-derivative sampling is more sensitive: it refits during the *first*
microbatch at eligible optimizer steps, after gathering 64/64/32 observations
respectively, rather than after all 512. Even the two layouts with 64 observations
supply different strata at that point and can fit different CDFs.

Optimizer/LR/EMA/clipping and GSM8K/checkpoint cadence are per optimizer step.
Validation-loss cadence uses microbatches: with accumulation 16, set
`trainer.val_check_interval=20000` to retain the original 1,250 optimizer-step
interval. Achieving partition-independent updates would require an explicitly
reviewed token-normalization policy and adaptive refits at optimizer boundaries,
or preserving the original logical microbatch grouping and refit protocol.
Neither policy was changed as part of this audit. Regression coverage is in
`core/tests/test_batch_partition_equivalence.py`.

### TinyGSM simplex queue (2026-10-17)

The simplex additions are `tinygsm_simplex_sc`,
`tinygsm_simplex_masked_sc`, `tinygsm_simplex_uniform_sc`,
`tinygsm_simplex_gaussian_sc`, and `tinygsm_simplex_sc_insertion`.
“Discrete” means the separate masked and uniform hybrids; insertion means
plain simplex plus the insertion modifier. All five use hidden self-conditioning,
adaptive time sampling, and the common TinyGSM training/evaluation recipe.
Use `configs/performance/tinygsm_compile.yaml` for compiled training.

The default queue excludes masked/uniform Gaussian insertion combinations.
To cancel already-pending trials without interrupting active training, put their
exact W&B IDs in `{"cancelled_wandb_ids": ["..."]}` in
`<Tune experiment storage>/queue-control.json`. Fresh trial workers check this
shared file before training and report `cancelled=true`. This does not stop a
trial that has already started and does not remove its row from Tune history.
Use this for standalone training jobs without dependent inference children.
The current queue uses fresh actors (`reuse_actors=false`); existing workers
must import the updated `core.sweeps.tune` module for this check to apply.

### Continuous-only adaptive time for Gaussian hybrids (2026-10-18)

The masked/uniform Gaussian TinyGSM presets now set
`algorithm.adaptive_time_views: [0]` (the Gaussian view). Draw a uniform
coordinator time `u`; the discrete branch uses `u`, while Gaussian uses the
adaptive sampler's quantile map `F_inverse(u)`. These clocks remain coupled,
but only the continuous marginal changes as the sampler adapts. Loss feedback
is indexed by the mapped continuous time. Generation uses the same branch-time
mapping; the discrete reverse trajectory remains on the uniform coordinator grid.
The default with no selected views retains the previous shared-clock behavior.
Corrected runs use a new W&B group `tinygsm-gaussian-continuous-time-20261018`
to distinguish them from the old shared-adaptive-clock training.

The simplex + masked/uniform presets also select continuous view 0 with
`adaptive_time_views: [0]`. Their discrete clock stays uniform while the simplex
clock uses the learned quantile map in training and generation. Corrected fresh
runs use `tinygsm-simplex-continuous-time-20261018`; the old shared-clock simplex
hybrid trials are cancelled. Plain simplex is unaffected.

### Simplex corruption precision and benchmark (2026-10-18)

New standalone/hybrid simplex configs select `process.precision: float32`.
`DirichletProcess(..., precision="float64")` remains available as a reference;
the Python constructor's default stays float64 for historical callers. Small
schedule tensors may remain float64, but coefficients are cast before expansion
so corruption, prior and reverse log states stay in the selected precision.
Hybrid clean states use float32, avoiding promotion of the large noisy state.
Existing serialized configs need an explicit precision override when resumed.

Run `python -m scripts.benchmark_simplex_precision --device cuda --output result.json`
on a reserved GPU to compare FP32/FP64 forward corruption and churn=1 reverse
sampling at batch16, sequence512, vocabulary49153. This is a corruption benchmark,
not end-to-end training throughput. A small CPU smoke benchmark was run; no GPU
benchmark was launched while the Gaussian hybrids own the allocation.
All simplex jobs are stopped for now, leaving masked/uniform Gaussian hybrids
running concurrently. Their precision/settings were not changed by this patch.
