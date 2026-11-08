# Preparing sweeps

Experiments run in two steps. `core.sweeps.prepare` expands a Hydra sweep
config into a plan of cases, and [`core.sweeps.nested_tuning`](nested-tuning.md)
runs that plan's manifest on Ray with Optuna selection. Hydra owns the grid,
Ray owns scheduling and GPU allocation, and Optuna owns the search; there are no
custom Trial, Sweep or executor classes. For Python callers,
[`compose`](../../core/sweeps/configuration.py) composes a preset with overrides.

## Hydra sweep configs

Hydra's built-in `BasicSweeper` owns Cartesian expansion and its defaults list
composes each recipe. The [sweep configs](../../configs/sweep/) define ordinary
`hydra.sweeper.params`; there is no separate grid configuration language.
For example, [hybrids.yaml](../../configs/sweep/hybrids.yaml) sweeps:

```yaml
hydra:
  sweeper:
    params:
      information_path/gaussian_state: one_hot,embedding
      information_path/path_schedule: adaptive,rank_linear
      modifiers/sc: off,on
      information_path/discrete_view: masked,uniform
```

The shared [common.yaml](../../configs/sweep/common.yaml) adds the tasks, seed,
training budget, worker resources and inference search spaces. Native config
groups select components through Hydra defaults; their small preparation labels
preserve readable run names.

## Prepare and collect

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare --multirun +sweep=hybrids \
  preparation.output=$SDDC_STORAGE_ROOT/plans/my-experiment
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare collect $SDDC_STORAGE_ROOT/plans/my-experiment
```

Run several multiruns into the same `preparation.output`, then `collect` once
after every grid succeeds; it validates the exported cases but cannot know which
grids you intended to include. CLI overrides can narrow or extend a sweep, for
example `task=battery/text8 seed=1,2 modifiers/sc=on`. On the GPU server, prepare through a
submitted job so manifests contain server paths (see
[remote execution](remote-execution.md)).

[`prepare.py`](../../core/sweeps/prepare.py) only exports the already-composed
configuration and its tuning specification. Although Hydra prints "Launching
jobs," these are preparation calls: they do not train, initialize Ray, or log to
W&B. Existing case/config files are refused rather than overwritten, and every
interpolation must resolve at export.

Each case produces `configs/<name>.yaml` and `cases/<name>/case.json`; `collect`
writes `manifest.json` for nested tuning. Hydra saves its configs and overrides
under `hydra/<family>/`.

A task's evaluation spec (tuning/test indices, score metric, direction, seeds and
task overrides) comes from `preparation.evaluation_file`, a committed JSON
(see `experiments/*/evaluation/`), plus any `preparation.evaluation` fields;
preparation refuses a file drawn from a different task. Each
inference policy inherits the composed sampler unless it supplies `sampler`;
`sampler_overrides` changes sampler fields. Callback NFE allocation and
algorithm identifiers are recorded at export. Recipe identity excludes the
preparation metadata.
