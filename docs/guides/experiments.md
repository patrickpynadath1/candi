# Experiments, cases, studies, and runs

An **experiment** is a comparison designed to answer a question. It combines
recipes, a task suite, seeds, training/search budgets, and an evaluation protocol.
For example: how does self-conditioning interact with corruption type across
Sudoku, Text8, and TinyStories?

These terms describe the existing workflow. They do not introduce new executor
classes or a second configuration system.

| Term | Meaning in this repository |
| --- | --- |
| Recipe | Information path, denoising task, and inference settings |
| Task suite | Tasks and their data, conditioning, and scoring conventions |
| Experiment configuration | Inputs that define the comparison, including budgets and search spaces |
| Experiment case | One recipe × task × training-seed pipeline, including training and inference selection |
| Manifest | JSON file listing the expanded case specifications to execute |
| Study | One Optuna optimization problem: the outer training search or an inner sampler search |
| Trial | One candidate configuration within an Optuna study |
| Run | One tracked training or evaluation execution; it may be a W&B run |
| Job | A schedulable unit of work executed by a Ray worker |

A generation seed is not another training seed. A single inference trial currently
runs evaluation for each configured generation seed, so it can produce several
W&B runs. Consequently, W&B run counts, Optuna trial counts, and completed case
counts are different measures of progress.

## How the pieces fit

```text
Experiment
  Manifest: list of experiment cases
    Case: recipe × task × training seed
      Outer Optuna study: training candidates
        Trial: acquire or train one checkpoint
          Inner Optuna study for each sampler
            Trials: evaluate inference settings on validation
      Select training candidate, sampler, and settings on validation
      Evaluate the selected winner on reserved final examples
```

Inference-only tuning has one outer candidate pointing to an existing checkpoint.
Fresh training with no training-parameter search has one outer candidate that
trains a model. Both use the same nested execution path.

The [nested tuning guide](nested-tuning.md) defines the manifest schema and saved
artifacts. [Task suites](task-batteries.md) explain composing tasks with recipes.
[Programmatic sweeps](programmatic-sweeps.md) document the lower-level Ray job API;
those jobs are not automatically nested hyperparameter searches.

## Execution and naming

`run_case` executes a case. `--case-concurrency` controls how many independent
cases can progress together; `--concurrency` controls inference candidates within
each case. Ray assigns CPUs/GPUs to their actual workers. Optuna proposes and
scores candidates. W&B records runs; these are not W&B-managed Sweeps.

For existing callers, `run_study` and `--study-concurrency` remain compatibility
aliases. Existing artifact names (`training/`, `inference/`, `optuna.sqlite`,
`selected.json`, `final/result.json`) and manifest fields are unchanged.

Some established paths still use `battery` and `algorithm_definition`, including
`configs/battery/`, `configs/algorithm_definition/`, and the example launchers.
In the documentation these mean **task suite** and **recipe definition**. Retaining
those paths keeps existing configs, commands, and saved experiment provenance
usable; a terminology cleanup does not require a new parallel implementation.
