# Programmatic sweeps

Use Python to compose complete method presets, replace Hydra component groups,
edit individual configurations, and queue runs across local GPUs. The API lives
in [`core.sweeps`](../../core/sweeps/programmatic.py). Composition/planning live in `programmatic.py`;
[`LocalExecutor`](../../core/sweeps/local.py) owns scheduling and persistence.
It uses the existing
[training/evaluation runner](../../core/training/runner.py) in a fresh subprocess
for every trial. Algorithms and models need no sweep-specific code.

## Sweep complete methods

Save this as `sweep.py` at the repository root and run `uv run python sweep.py`:

```python
from core.sweeps import Resources, Sweep, Trial

trials = [
    Trial(
        experiment=method,
        overrides={"trainer.max_steps": 2_000, "seed": seed},
    )
    for method in [
        "autoregressive",
        "blurring_diffusion",
        "duo",
        "replaid",
        "spherical_flow_matching",
        "simplex",
    ]
    for seed in [1, 2, 3]
]

sweep = Sweep(
    trials,
    output_dir="core/outputs/method-comparison",
    resources=Resources(gpus=1, cpus=4, max_parallel=2),
)
for trial in sweep.plan():
    print(trial["name"], trial["config"]["algorithm"]["_target_"])

results = sweep.run()
for result in results:
    print(result.name, result.status, result.metrics)
```

These presets assemble compatible components, using the default Sudoku task/data
unless the preset overrides them. They retain their own optimization, schedule,
and architecture settings; the same number of steps does not make this a
controlled benchmark. Provision the task data as for ordinary training. A plan
resolves configuration but does not instantiate models, check every mathematical
combination, prepare data, or launch jobs.

## Edit configs and replace components

`compose()` accepts a preset plus the same group/field overrides as Hydra. Mapping
values are literal values. A list of strings permits advanced Hydra grammar.
Composing must happen outside an already active Hydra initialization context.

```python
from core.sweeps import Resources, Trial, compose, run_sweep

trials = []
for representation in ["one_hot", "embedding"]:
    for sampler in ["euler", "heun"]:
        config = compose(
            "gaussian_diffusion_sudoku",
            {
                "representation": representation,
                "noise_schedule": "exponential_ve",
                "sampler": sampler,
            },
        )
        config.optimizer.lr = 3e-4
        config.trainer.max_steps = 2_000
        config.sampler.steps = 64
        trials.append(Trial(config=config, name=f"{representation}-{sampler}"))

plans = run_sweep(
    trials,
    output_dir="core/outputs/gaussian-comparison",
    resources=Resources(gpus=1, cpus=4),
    dry_run=True,
)
```

`run_sweep` also accepts `DictConfig` objects directly. Each config is copied;
interpolations such as `${model}` remain live while you edit. At planning/execution,
all values are resolved into independent snapshots. Runtime paths are captured
without changing Hydra's global resolver. `output_dir` is assigned per trial.

Use a full method preset rather than only overriding `algorithm`: changing only
the class leaves the previous method's other components in place. Group overrides
replace groups; `++some.new_field` can add a new field. When editing composed
configs directly, assign an entire component config to replace it. Component
compatibility checks remain authoritative at instantiation.

For a Cartesian product with seeds:

```python
from core.sweeps import grid

trials = grid(
    ["autoregressive", "duo"],
    axes={"optimizer.lr": [1e-4, 3e-4]},
    seeds=[1, 2, 3],
    overrides={"trainer.max_steps": 2_000},
)
```

For conditional choices, use ordinary Python loops and explicit trials. No
special search-space language is required.

## Resources and GPU visibility

`Resources(gpus=1, cpus=4, gpu_ids=[2, 3])` assigns each active trial one distinct
GPU. Explicit IDs are physical IDs or UUIDs, and must belong to an inherited
`CUDA_VISIBLE_DEVICES` allowlist if one exists. Without an explicit list, the
executor uses the visible devices. It does not inspect memory use or coordinate
with unrelated jobs; select GPUs allocated to your sweep.

Each child receives exactly one visible GPU and uses local device 0. Resources
set the trainer accelerator and evaluation device. Configs requesting distributed
strategies, multiple devices, or multiple nodes are rejected. CPU execution uses
`Resources(gpus=0, cpus=1, max_parallel=2)`. Concurrency is limited by CPU affinity,
CPUs per trial, available GPU slots, and `max_parallel`.

`cpus` sets OMP/MKL thread limits and determines queue capacity, not an OS CPU
reservation. Data-loader workers and library subprocesses consume additional CPU
resources; tune `loader.num_workers` accordingly. Set `max_parallel` explicitly
for CPU-heavy trials.

This is a **local, single-machine executor**, with zero or one GPU per trial. It
does not yet provide Ray/Slurm execution, multi-GPU trials, Bayesian search, or
ASHA. The existing [frontier runner](frontier-sweeps.md) remains separate and
sequential; this API can queue explicit evaluation configs but does not compute
Pareto frontiers itself.

## Outputs, failures and restart

Each sweep has a `manifest.json` and coordinator lock. Each trial has its own
`trial.yaml`, `run.log`, normal runner `config.yaml`, metrics and checkpoints.
Training metrics report the latest nonempty value of each CSV metric (potentially
from different steps); evaluation metrics come from `results.json`, flattened with slash-separated keys
(e.g. `word_accuracy/by_length/4/count`). Only numeric aggregates enter the
manifest; generated records stay in the per-trial `results.json`.

Rerunning the same sweep skips successful matching trials and retries failed or
interrupted trials. All trial names, order and resolved config fingerprints must
match. Changes require a new output directory; `resume=False` rejects an existing
manifest. Trial directories without a manifest are also rejected to avoid
silently overwriting unrelated runs. Fingerprints cover resolved configuration,
not code revisions, checkpoint contents or dataset contents: use a new sweep
folder when those change.

A retry starts the runner again with the same config. It does **not** automatically
resume training from a partially written checkpoint; set `checkpoint.resume`
explicitly in a new sweep if that is desired. Logs append across attempts.
`SweepFailed` exposes all `results`, including successful trials; remaining trials
continue after an individual failure. Interrupting the coordinator terminates
its active child process groups and records interrupted status. SIGKILL or host
failure cannot run cleanup; ensure surviving workers are stopped before restart.

Only one coordinator may operate on a sweep directory at a time. Distinct sweep
directories do not share a resource registry, so give concurrent sweeps disjoint
GPU allocations.
