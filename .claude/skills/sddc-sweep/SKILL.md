---
name: sddc-sweep
description: Write an SDDC sweep config (recipes x tasks x seeds, with validation-only tuning and a held-out test split), prepare it into a plan of cases and check the manifest without training. Use when setting up an experiment or comparison, adding an arm to an existing one, or before submitting nested tuning.
---

# Write a sweep and prepare its plan

Terms (experiment, case, study, trial): `docs/guides/experiments.md`. A sweep is a Hydra
multirun over a recipe; `core.sweeps.prepare` turns each combination into a **case** with a
pinned config and tuning spec. Preparing never trains or starts Ray.

## 1. Write `configs/sweep/<name>.yaml`

Copy the closest worked example (`configs/sweep/sc_vs_depth.yaml` is the reference):

```yaml
# @package _global_
defaults:
- /experiment: corruption_factorial      # the base recipe
- common                                 # shared protocol: configs/sweep/common.yaml
- /ablation_path: gaussian_cdcd          # default arm of each axis
- /modifiers/feedback: none
- /inference/inference_policies: hybrid_renoising
- override /task: battery/sudoku_hard
- _self_
preparation:
  name: <name>
  output: ${oc.env:SDDC_STORAGE_ROOT,${oc.env:HOME}/.cache/sddc}/plans/<name>
  evaluation_file: experiments/<name>/evaluation/${preparation.task_name}.json
  family: <name>
  recipe_name: ${preparation.path_name}-${preparation.feedback_name}
hydra:
  sweeper:
    params:
      task: battery/sudoku_hard,battery/text8
      ablation_path: gaussian_cdcd,masked_elbo,uniform
      modifiers/feedback: none,expected_embedding
```

What to decide:

- **Axes** (`hydra.sweeper.params`): group paths, the same names as on the command line.
  Each value must be a file in that group.
- **Case names**: `common.yaml` builds `${preparation.task_name}-${preparation.recipe_name}-seed-${seed}`.
  Make `recipe_name` distinguish every arm, or two cases will collide (`collect` refuses
  duplicates).
- **Evaluation file** per task: JSON with `task`, `score {kind: metric, metric}`,
  `direction`, `seeds`, `test_seeds`, `tuning_indices`, `test_indices`. Tuning and test
  indices, and the two seed sets, must be disjoint. Reuse a committed file
  (`experiments/*/evaluation/`) when comparing with published numbers.
- **Inference policies** (`preparation.inference`): each has a `name`, a search `space`
  (only `sampler.*` keys other than `sampler.steps`/`max_steps`, plus
  `algorithm.predictor.temperature` and `algorithm.predictor.commitment`), `trials`,
  `budget_parameter: steps`, and optionally its own `sampler` or `sampler_overrides`. Shared
  sets live in `configs/inference/inference_policies/`.
- **Training search** (`preparation.training.space`, `trials`): keys must start with
  `optimizer.`, `model.`, `algorithm.`, `objective.`, `noise_schedule.`, `time_sampler.`,
  `training.`, `self_conditioning.` or `lr_scheduler.`. Default: one fixed candidate.
- **Budget**: `preparation.evaluation.nfe` (default 128 network calls, terminal included).
- **Resources**: `preparation.local.resources_per_job` (0 or 1 whole GPU per worker).

## 2. Prepare and collect

```bash
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare -m \
  +sweep=<name> preparation.output=$SDDC_STORAGE_ROOT/plans/<name>
uv run --locked --extra tuning --extra scoring python -m core.sweeps.prepare \
  collect $SDDC_STORAGE_ROOT/plans/<name>
```

The plan holds `configs/<case>.yaml`, `cases/<case>/case.json` and `manifest.json`. Existing
case files are never overwritten: a changed protocol needs a new `preparation.name` or
output directory. Prepare on the machine that will run the cases, so paths in the manifest
resolve there (for a server, submit the prepare command as a job; skill `sddc-remote`).

## 3. Check

```bash
uv run --locked --extra tuning --extra scoring python -m core.sweeps.nested_tuning \
  $SDDC_STORAGE_ROOT/plans/<name>/manifest.json --check \
  [--output $SDDC_STORAGE_ROOT/runs/<name>]
```

It validates every case (the same checks a run performs), prints one row per case (training
trials, policies, inference trials, NFE) and totals, and with `--output` reports whether each
case is new or resumes, and refuses cases whose stored `protocol.json` changed. It needs no
Ray and trains nothing.

## Done when

`--check` exits 0, the case count and names are what you expected, and a test like
`core/tests/test_experiment_sc_vs_depth.py` (expand the grid, run one tiny case on CPU)
passes for a new worked example.
