# Configuration

A [recipe](recipe.md) combines an information path, denoising task, and inference
settings. Hydra component groups below are its implementation, while an
[experiment configuration](../guides/experiments.md) adds tasks, budgets, seeds,
and hyperparameter search spaces.

Hydra composes YAML from root [`configs/`](../../configs/). The default training
configuration is [`config.yaml`](../../configs/config.yaml); the frontier entrypoint
uses [`frontier.yaml`](../../configs/frontier.yaml). Run commands from the repository
root with uv.

## Select components and recipes

```bash
uv run python main.py --cfg job
uv run python main.py sampler=ancestral sampler.steps=180 --cfg job
uv run python main.py +experiment=gaussian_diffusion_sudoku --cfg job
```

A config group such as `sampler=ancestral` selects `configs/sampler/ancestral.yaml`.
Scalar overrides change fields of the selected config. `+experiment=...` appends
an experiment to the defaults list; experiment files use the global package to
select compatible components and override training/model settings.

Hydra recursively instantiates `_target_` values. For example, an algorithm config
references `${model}`, `${parameterization}`, `${process}`, and other selected
objects. Constructors receive their explicit dependencies, not the entire root
configuration. The runner shares one instantiated Task with the data module and
periodic task-evaluation callback.

Prefer a complete experiment recipe when switching mathematical families.
`algorithm=gaussian_diffusion` alone does not automatically replace an absorbing
process, discrete model, or incompatible sampler. Independently selecting a task
also requires checking vocabulary, sequence length, and model memory needs.

## Important groups

| Group/section | Purpose |
| --- | --- |
| `algorithm`, `model`, `parameterization`, `prediction` | Algorithm assembly, architecture, output interpretation, prediction coordinator. |
| `process`, `noise_schedule`, `time_sampler`, `representation`, `objective` | Mathematical components. |
| `sampler` | Generation algorithm and its budget/settings. |
| `task`, `data`, `token_space` | Dataset/task semantics and vocabulary. |
| `datamodule`, `evaluator`, `callback` | Shared infrastructure implementations. |
| `lr_scheduler`, `optimizer`, `strategy`, `trainer` | Optimization and Lightning execution. |
| `self_conditioning`, `block_diffusion` | Shared modifier configuration. |
| `evaluation`, `checkpoint` | Evaluation weights and native training resume. |

Some groups exist only in particular recipes, such as a curriculum or generation
order. Refer to the selected YAML rather than assuming every option is global.

## Inspect before executing

`--cfg job` prints the composed configuration without training or allocating a
model. Saved runs contain a resolved `config.yaml`; this is the best starting point
for evaluating the same architecture later:

```bash
uv run python main.py --config-path /absolute/path/to/run --config-name config \
  mode=evaluate evaluation.checkpoint=/absolute/path/to/run/checkpoints/last.ckpt \
  output_dir=/absolute/path/to/evaluation hydra.run.dir=/absolute/path/to/evaluation
```

Override **both** output paths when using a resolved saved config: its `output_dir`
is a literal old directory rather than a live Hydra interpolation. That saved
config may no longer have group defaults, so selecting new config groups can fail;
edit the resolved config explicitly or recompose the matching original recipe.

## Overrides and sweeps

```bash
uv run python main.py --multirun seed=1,2,3 \
  hydra.sweep.dir=core/outputs/seed_sweep
```

An ordinary comma override is a Hydra multirun axis. A frontier grid is configured
through `frontier.axes`, a list of parameter/value lists; see
[frontier sweeps](../guides/frontier-sweeps.md). Quote braces and brackets in shell
commands. Hydra merges dictionary overrides, whereas replacing the frontier axes
list replaces that list as a whole. Use `+` to add absent fields, `++` to add or
replace, and `~` to remove a config field/group when appropriate.

Default artifacts are under `core/outputs/`, with dataset caches under
`core/data_cache/`. These are storage conventions, not code dependencies. Existing
cache symlinks are supported; clean clones do not contain the cached contents.

## Adding a config

Add a YAML file in the relevant group with an importable `_target_` and explicit
constructor arguments. Add an experiment recipe only when several component
choices must travel together. Preserve existing recipes unless their behavior is
intentionally changing. [Adding a component](../extending/adding-a-component.md)
and [adding a method](../extending/adding-a-method.md) describe the implementation
and validation steps.
