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
