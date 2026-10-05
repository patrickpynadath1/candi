# Programmatic sweeps with Ray Tune

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) owns the sweep API,
trial scheduling, GPU allocation, results and experiment restoration. Hydra
composes the existing presets. The only repository helpers are
[`compose`](../../core/sweeps/configuration.py) and
[`run_trial`](../../core/sweeps/tune.py), an adapter to the existing
training/evaluation runner. There are no custom Trial, Sweep or executor classes.
Ray Tune is installed by `uv sync` with the other project dependencies.

For a ready-made catalogue covering Gaussian schedules, simplex, spherical flow,
uniform, MDLM, RePlaid and blurring, see the
[base algorithm sweep](base-algorithm-sweep.md).

## Sweep complete method presets

Save this as `sweep.py` at the repository root. Run it with
`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 CUDA_VISIBLE_DEVICES=2,3 uv run python sweep.py` to restrict a new local Ray
instance to those GPUs. Disabling Ray's automatic uv runtime packaging makes local
workers use the same uv environment and checkout, including ignored data-cache
symlinks. Without it, Ray packages a copy of the project and creates a worker uv
environment, which requires separately provisioning the data paths:

```python
from pathlib import Path

import ray
from ray import tune
from core.sweeps import run_trial

if __name__ == "__main__":
    ray.init()
    trainable = tune.with_resources(run_trial, {"cpu": 4, "gpu": 1})
    tuner = tune.Tuner(
        trainable,
        param_space={
            "experiment": tune.grid_search([
                "autoregressive", "blurring_diffusion", "duo",
                "replaid", "spherical_flow_matching", "simplex",
            ]),
            "overrides": {
                "trainer.max_steps": 2_000,
                "seed": tune.grid_search([1, 2, 3]),
            },
        },
        tune_config=tune.TuneConfig(max_concurrent_trials=2),
        run_config=tune.RunConfig(
            storage_path=str(Path("core/outputs/ray").resolve()),
            name="method-comparison",
            log_to_file=True,
        ),
    )
    results = tuner.fit()
    print(results.get_dataframe())
    if results.num_errors:
        raise RuntimeError(f"{results.num_errors} trials failed; inspect results.errors")
    ray.shutdown()
```

This runs every preset/seed combination. Presets retain their own optimizer,
architecture, task and schedule choices; equal training steps do not constitute
a controlled benchmark. Provision task data as for ordinary training.

## Edit configs or enumerate compatible combinations

`compose()` accepts a preset and Hydra group/field overrides. Mapping values are
literal values; a list of strings accepts advanced Hydra grammar. Compose outside
an already active Hydra initialization context. Interpolations stay live while
you edit the configuration.

```python
from omegaconf import OmegaConf
from ray import tune
from core.sweeps import compose, run_trial

trials = []
for representation in ["one_hot", "embedding"]:
    for sampler in ["euler", "heun"]:
        config = compose("gaussian_diffusion_sudoku", {
            "representation": representation,
            "noise_schedule": "exponential_ve",
            "sampler": sampler,
        })
        config.optimizer.lr = 3e-4
        config.trainer.max_steps = 2_000
        config.sampler.steps = 64
        trials.append({"config": OmegaConf.to_container(config, resolve=False)})

# Use this trainable/param_space in Tuner, with RunConfig as above.
trainable = tune.with_resources(run_trial, {"cpu": 4, "gpu": 1})
param_space = {"trial": tune.grid_search(trials)}
```

Each list item can instead be `{"experiment": "duo", "overrides": {"seed": 1}}`.
Use either a config snapshot or experiment/overrides, not both. The outer `trial`
key is solely for sweeping complete specifications: put all options inside it.
For independent axes, nested `tune.grid_search` values form a Cartesian product.
Ordinary Python loops let you enumerate only meaningful combinations.

Use a full preset when changing methods. Changing only `algorithm` leaves the
previous method's other components in place. Composition does not validate every
mathematical combination; component compatibility checks still apply when the
model is instantiated. Setting `mode=evaluate` or `mode=sample` and
`evaluation.checkpoint` uses the same adapter for sampling/evaluation sweeps.

## A dynamic queue of training and inference jobs

The runnable [example](../../examples/train_then_evaluate.py) uses one `Tuner`
with [`TrainingInferenceSearch`](../../core/sweeps/queue.py), a small Tune
`Searcher` extension. Ray still owns GPU allocation, workers and trial state.
Each successful training completion immediately adds that checkpoint's inference
jobs to the queue; other training jobs can still be running.

Define concrete jobs with Python loops using the
[`TrainingJob` and `InferenceJob` dataclasses](../../core/sweeps/jobs.py). The queue
also uses typed records internally; conversion to dictionaries happens at the
Tune/persistence boundary. A training job accepts either experiment/overrides or
an edited `config`, plus its compatible inference settings:

```python
from ray import tune
from core.sweeps import InferenceJob, TrainingJob
from core.sweeps.queue import TrainingInferenceSearch, run_job

jobs = [TrainingJob(
    name="mdlm-seed-1",
    experiment="sudoku_hard",
    overrides={"seed": 1},
    checkpoint="last.ckpt",  # Or best.ckpt, if validation produced it.
    inference=(
        InferenceJob(sampler="ancestral", overrides={"sampler.steps": 64}),
        InferenceJob(sampler="ancestral", overrides={"sampler.steps": 128}),
        InferenceJob(sampler="entropy_bounded", overrides={"sampler.entropy_bound": 0.1}),
    ),
)]

trainable = tune.with_resources(
    run_job,
    lambda job: {"cpu": 4 if job["stage"] == "train" else 2, "gpu": 1},
)
tune_config = tune.TuneConfig(
    search_alg=TrainingInferenceSearch(jobs),
    num_samples=-1,
    max_concurrent_trials=2,
)
# Pass trainable, tune_config and RunConfig(storage_path=..., name=...) to Tuner.
```

`num_samples=-1` lets the searcher signal when the queue is finished. Do not use a
finite training-job count: inference jobs are additional trials. Define concrete
training specs (using ordinary loops); do not embed unresolved `grid_search`
objects inside these jobs. `max_concurrent_trials` limits the combined number of
issued jobs. Newly generated inference takes priority over training still in the
searcher's queue; it does not preempt running jobs or jobs already given to Tune.
Resource requirements can depend on `job["stage"]`, as above.

Inference loads the **persisted training config and checkpoint**, not temporary
worker files. It replaces the sampler and applies only inference overrides:
`sampler.*`, `evaluation.max_examples`, `evaluation.use_ema`, `seed`,
`loader.eval_batch_size`, and `trainer.precision`. Omitting `sampler` preserves
the saved sampler. The model, task and parent checkpoint remain fixed. Sampler
compatibility is still checked by the existing components. Shaping changes to
predictor/model configuration are not supported by this queue adapter.

Result configurations contain `stage`, `name`, and `parent` (the training trial
ID) so metrics can be grouped by checkpoint. The example includes MDLM with
ancestral/entropy-bounded sampling and Gaussian with Euler/Heun sampling.

The queue persists pending training and inference jobs, issued trial IDs,
completion status and which parents have already generated descendants through
Tune's searcher checkpoint hooks. `Tuner.restore(..., trainable=trainable)`
restores unfinished sweeps and does not regenerate completed descendants. The
example automatically restores an existing experiment; use a different experiment
name for changed jobs. Restoring the same folder does not append newly edited
job definitions.

Only successful training jobs generate inference. Configure automatic retries
with `RunConfig(failure_config=tune.FailureConfig(max_failures=...))`; once retries
are exhausted, failure stays visible in the result grid and independent jobs
continue. After the queue is fully finished, retry failed parents in a new
experiment. Reopening terminal errors with `resume_errored=True` is not supported
for generating new descendants after Tune has closed its search.

The dynamic adapter currently requires a local/shared filesystem accessible to
the coordinator and every worker. Remote object-store checkpoints must first be
staged on shared storage. Automatic partial-training checkpoint resume and
intermediate metric reporting remain outside this adapter.

## GPU allocation and execution

Ray assigns each trial its requested logical resources and sets
`CUDA_VISIBLE_DEVICES` in the worker. The adapter uses local GPU 0, or CPU when no
GPU is requested. Use `tune.with_resources(run_trial, {"cpu": 4})` for CPU trials.
Tune limits concurrency by available resources and `max_concurrent_trials`.
Resource counts are scheduling reservations, not hard CPU/VRAM limits. Select
GPUs allocated to you; Ray does not account for unrelated processes' memory use.

The adapter supports zero or one whole GPU per trial and requires one trainer
device, one node and the auto strategy. Multi-GPU training inside a trial needs
a separate distributed-training integration. Ordinary multi-GPU training through
`main.py` is unchanged. Actor reuse should remain disabled (Tune's default), so
trials do not share model or CUDA state.

Workers preserve repository-relative input paths; outputs always go into the
absolute Tune trial directory under `run/`. Ray may execute on an existing
cluster, but every worker must have this project, its environment and data, and
multi-node sweeps require shared storage. The executed checks cover local Ray.
The existing [frontier runner](frontier-sweeps.md) remains separate and sequential;
Tune evaluation sweeps do not automatically calculate Pareto frontiers.

## Metrics, outputs and restoration

`Tuner.fit()` returns Ray's `ResultGrid`. Inspect `result.error`, `result.metrics`
and `result.path`, or `results.get_dataframe()`. Failures remain visible in the
result grid; check `results.num_errors` if a script should fail when any trial fails.

Each trial writes normal resolved `run/config.yaml`, checkpoints, CSV training
metrics and, for evaluation, `run/results.json`. Tune stores its own experiment
state and reported metrics. `log_to_file=True` captures worker stdout/stderr.
The reported `output_dir` points to the worker's temporary `run/` directory and
`cuda_visible_devices` records the worker's assigned devices. On success the
adapter registers the entire `run/` directory as a Tune checkpoint, preserving
weights, config, metrics and generated records in `RunConfig.storage_path`.
Access durable artifacts through `result.checkpoint.as_directory()`:

```python
with result.checkpoint.as_directory() as saved:
    checkpoint_path = Path(saved) / "checkpoints/last.ckpt"
    # Load/use the file here (evaluation-only trials have results.json instead).
```

Failed trials have no final Tune checkpoint. Their local worker directory may
still contain partial outputs; `sync_config=tune.SyncConfig(sync_artifacts=True)`
in `RunConfig` additionally enables syncing worker artifacts to persistent storage.

The adapter reports **final metrics only**. Training uses the latest nonempty
value of each CSV metric, which can come from different steps. Evaluation reports
numeric aggregates flattened with slash-separated keys; generated records stay
in `results.json`. Early-stopping schedulers such as ASHA need intermediate
reporting and are not enabled by this adapter.

Restore a saved experiment explicitly with the same trainable:

```python
restored = tune.Tuner.restore(
    "/absolute/path/to/core/outputs/ray/method-comparison",
    trainable=trainable,
    resume_errored=True,
)
results = restored.fit()
```

Tune restores the original search space and leaves completed trials completed.
Use a new experiment name for changed configurations. `FailureConfig` in
`RunConfig` can configure automatic retries. Retried/interrupted trials restart
training from the beginning unless `checkpoint.resume` was explicitly set;
this adapter does not register intermediate Lightning checkpoints with Tune.
Ray restoration is distinct from Lightning optimizer/checkpoint restoration.
Old custom-executor manifests are not Ray experiment state; start a new Tune
experiment when migrating an old sweep.

## Inspect saved metrics without launching jobs

From the repository root, read the experiment directory (after trials have
reported results):

```python
from pathlib import Path
from ray import tune

path = Path("core/outputs/ray/base-algorithms").resolve()
frame = tune.ExperimentAnalysis(str(path)).dataframe()
print(frame.to_string(index=False))
frame.to_csv("sweep-results.csv", index=False)
```

Each row is a trial, with configuration columns such as `config/stage` and
`config/name`, and reported metrics such as `train/loss`, `val/nll`, and task
aggregate metrics. The adapter reports final results only, so running jobs have
no live training curves in Tune. Native trial directories also contain Tune's
`result.json` and `progress.csv`. Use W&B for live training and periodic generation
tracking. The base algorithm sweep enables it by default; custom job definitions
must supply a non-null `wandb` configuration themselves.
