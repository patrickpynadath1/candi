# CANDI documentation

CANDI composes models, prediction parameterizations, corruption processes,
schedules, objectives, and samplers for discrete data. Source lives in `core/`,
Hydra configurations in root `configs/`, and executable entrypoints in `main.py`
and `frontier.py`. This documentation describes the current implementation.

## Start here

- [Getting started](getting-started.md): install with uv and run a CPU smoke test.
- [Architecture](concepts/architecture.md): ownership and training/generation flow.
- [Component contracts](concepts/component-contracts.md): states, predictions, masks,
  and compatible mathematical meanings.
- [Configuration](concepts/configuration.md): compose a method, task, and sampler.

## Run experiments

- [Training](guides/training.md)
- [Sampling and evaluation](guides/sampling-and-evaluation.md)
- [Programmatic sweeps](guides/programmatic-sweeps.md)
- [Frontier sweeps](guides/frontier-sweeps.md)
- [Checkpointing](guides/checkpointing.md)
- [Distributed training](guides/distributed-training.md)

## Component reference

| Component | Responsibility |
| --- | --- |
| [Algorithms](components/algorithms.md) | Connect training and prediction operations. |
| [Models](components/models.md) | Neural architecture and conditioning. |
| [Representations](components/representations.md) | Map tokens to/from continuous states. |
| [Predictors](components/predictors.md) | Give model outputs mathematical meaning and apply inference shaping. |
| [Processes](components/processes.md) | Forward corruption and reverse-transition mathematics. |
| [Noise schedules](components/noise-schedules.md) | Map time to corruption coefficients. |
| [Time sampling](components/time-sampling.md) | Draw training times and, where supported, construct inference grids. |
| [Objectives](components/objectives.md) | Compute mathematical training losses. |
| [Samplers](components/samplers.md) | Own generation state and update loops. |

Architectural/sequence modifiers have dedicated references:
[self-conditioning](modifiers/self-conditioning.md) and
[block diffusion](modifiers/block-diffusion.md).

## Method recipes

- [Discrete diffusion](methods/discrete-diffusion.md): MDLM, uniform, SEDD, DUO, D3PM.
- [Gaussian diffusion](methods/gaussian-diffusion.md): one-hot/embedding recipes.
- [Spherical flow matching](methods/spherical-flow-matching.md)
- [Simplex diffusion](methods/simplex-diffusion.md)
- [Blurring diffusion](methods/blurring-diffusion.md)
- [RePlaid](methods/replaid.md)
- [Autoregressive and permuted autoregressive](methods/autoregressive.md)
- [Hybrids](methods/hybrids.md): parallel noisy views of the same clean sequence.

## Tasks

See [task conventions](tasks/overview.md), then
[Sudoku](tasks/sudoku.md), [Text8](tasks/text8.md),
[OpenWebText](tasks/openwebtext.md), or [TinyGSM](tasks/tinygsm.md).

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
