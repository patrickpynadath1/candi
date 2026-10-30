# CANDI documentation

CANDI supports controlled experiments on diffusion language model recipes. Its
purpose is to make component choices and their interactions explicit, not to
prescribe one diffusion algorithm or provide a large-scale pretraining stack.

## The recipe

Read the [recipe overview](concepts/recipe.md), then its three parts:

1. [Information path](concepts/information-path.md): representation, corruption
   process, and noise schedule.
2. [Denoising task](concepts/denoising-task.md): backbone, output parameterization,
   objective, and training time sampler.
3. [Inference settings](concepts/inference-settings.md): initialization, inference
   time/noise grid, update rule, and terminal decoding.

[Modifiers](concepts/recipe.md#modifiers-across-the-three-parts) explain how
self-conditioning, multiple views, insertion/deletion, and block diffusion affect
these choices. Compatible component composition is the goal; arbitrary combinations
are not guaranteed to be mathematically valid.

## Start here

- [Getting started](getting-started.md): install with uv and run a CPU smoke test.
- [Architecture](concepts/architecture.md): ownership and training/generation flow.
- [Component contracts](concepts/component-contracts.md): states, predictions, masks,
  and compatible mathematical meanings.
- [Configuration](concepts/configuration.md): compose a method, task, and sampler.

## Run experiments

Start with [experiments, cases, studies, and runs](guides/experiments.md).
A manifest lists cases; each case can contain nested Optuna studies. Both training
and inference choices are selected on validation before final evaluation.

- [Nested training and inference tuning](guides/nested-tuning.md)
- [Inference-only tuning as a fixed-checkpoint case](guides/inference-tuning.md)

- [Training](guides/training.md)
- [Sampling and evaluation](guides/sampling-and-evaluation.md)
- [Small-task evaluation suite](guides/evaluation-suite.md)
- [Recipe definitions and task suites](guides/task-batteries.md)
- [TinyStories judging](guides/story-judging.md)
- [Programmatic sweeps](guides/programmatic-sweeps.md)
- [Base algorithm sweep](guides/base-algorithm-sweep.md)
- [Corruption × self-conditioning sweep](guides/corruption-self-conditioning.md)
- [Frontier sweeps](guides/frontier-sweeps.md)
- [Checkpointing](guides/checkpointing.md)
- [Distributed training](guides/distributed-training.md)

## Component reference

| Recipe part | Component references |
| --- | --- |
| Information path | [Representations](components/representations.md), [processes](components/processes.md), [noise schedules](components/noise-schedules.md) |
| Denoising task | [Backbones/models](components/models.md), [output parameterizations](components/predictors.md), [objectives](components/objectives.md), [training time sampling](components/time-sampling.md) |
| Inference settings | [Samplers](components/samplers.md), with compatible process updates and predictor shaping |
| Coordination | [Algorithms](components/algorithms.md) connect the components; [architecture](concepts/architecture.md) describes execution ownership |

Architectural/sequence modifiers have dedicated references:
[self-conditioning](modifiers/self-conditioning.md),
[block diffusion](modifiers/block-diffusion.md), and
[insertion/deletion and variable-length diffusion](modifiers/variable-length.md).
The insertion reference explains per-process birth states, shared hybrid edits,
block-local generation, adaptive-time feedback, and the current Text8 evidence.

## Method recipes

- [Runnable paper presets](guides/paper-recipes.md): configuration names, adaptations and limits.

- [BlockGen fixed-block recipes](methods/blockgen.md): masked/uniform baselines and full-method limitations.
- [Discrete diffusion](methods/discrete-diffusion.md): MDLM, uniform, SEDD, DUO, D3PM.
- [GIDD mask/uniform corruption](methods/gidd.md): a single-view mixture with a CE baseline.
- [Gaussian diffusion](methods/gaussian-diffusion.md): one-hot/embedding recipes.
- [Spherical flow matching](methods/spherical-flow-matching.md)
- [Rank-linear schedules](methods/rank-linear-schedules.md): simplex and spherical calibration.
- [Simplex diffusion](methods/simplex-diffusion.md)
- [Blurring diffusion](methods/blurring-diffusion.md)
- [RePlaid](methods/replaid.md)
- [Autoregressive and permuted autoregressive](methods/autoregressive.md)
- [Hybrids](methods/hybrids.md): parallel noisy views of the same clean sequence.

## Tasks

See [task conventions](tasks/overview.md), then
[Sudoku](tasks/sudoku.md), [Text8](tasks/text8.md),
[N-Queens](tasks/nqueens.md), [TinyStories](tasks/tinystories.md),
[Mazes](tasks/maze.md), [OpenWebText](tasks/openwebtext.md), or [TinyGSM](tasks/tinygsm.md).

## Extend and assess

- [Add a component](extending/adding-a-component.md)
- [Add a method](extending/adding-a-method.md)
- [Add a task](extending/adding-a-task.md)
- [Coverage and limitations](validation/coverage.md)
- [Reproduction results](validation/reproduction-results.md)

An available config does not establish benchmark parity. The validation pages
separate execution checks from numerical comparisons and trained-model results.
Generated artifacts under `core/outputs/` and cached datasets are local, ignored
files; a clean clone does not contain them. Legacy implementations remain in Git
history before the `candiv2.1` cleanup.
