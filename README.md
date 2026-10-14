# CANDI v2.1

A framework for testing what makes diffusion language models work. Compose recipes
from an **information path**, a **denoising task**, and **inference settings**, then
compare them across tasks with explicit training budgets and validation-based
hyperparameter selection.

The aim is controlled, affordable research into component interactions—not
large-scale pretraining or introducing a single preferred diffusion algorithm.
Compatible components can be exchanged without rewriting an entire method;
mathematical constraints remain explicit.

| Recipe part | Components |
| --- | --- |
| [Information path](docs/concepts/information-path.md) | Representation, corruption process, noise schedule |
| [Denoising task](docs/concepts/denoising-task.md) | Backbone, output parameterization, objective, training time sampler |
| [Inference settings](docs/concepts/inference-settings.md) | Initialization, inference time/noise grid, update rule, terminal decoding |

Self-conditioning, multiple views, block diffusion, and insertion/deletion are
modifiers that can affect several parts. See the [recipe overview](docs/concepts/recipe.md).
Tasks define data, conditioning, and metrics independently of corruption.

Implementations and tests live in `core/`, Hydra configs in `configs/`, and
[documentation](docs/index.md) explains both the research model and code contracts.

## Quick start

```bash
uv sync --locked
uv run python main.py +experiment=smoke
```

Python 3.10+ is required; the lock uses the PyTorch CUDA 12.1 index. The smoke recipe
runs on CPU. Datasets, pretrained weights, and reference scorers are external
inputs. Output/cache directories are ignored by Git.

```bash
# MDLM on Sudoku hard
uv run python main.py

# Two-GPU training
CUDA_VISIBLE_DEVICES=0,1 uv run python main.py strategy=ddp trainer.devices=2

# Evaluate matching pretrained weights
uv run python main.py mode=evaluate evaluation.checkpoint=/path/to/last.ckpt

# Pretrained Sudoku frontier grid
uv run python frontier.py evaluation.checkpoint=/path/to/last.ckpt
```

## Experiments

An experiment expands recipes, tasks, and training seeds into **cases**. Each case
uses an outer Optuna study for training choices and inner studies for inference
settings. Both selection stages use validation; only the selected winner reaches
final evaluation. An existing checkpoint is the one-candidate special case of
this same workflow. Ray schedules the training/evaluation workers and W&B records
runs.

Install and launch from the same locked environment:

```bash
uv sync --locked --extra tuning --extra scoring --extra tracking
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning --help
```

See [experiment terminology](docs/guides/experiments.md),
[task suites](docs/guides/task-batteries.md), and
[nested tuning](docs/guides/nested-tuning.md) for configuration and execution.

## Documentation

- [Getting started](docs/getting-started.md) and [configuration](docs/concepts/configuration.md)
- [Architecture and ownership](docs/concepts/architecture.md)
- [Training](docs/guides/training.md), [distributed training](docs/guides/distributed-training.md), and [checkpointing](docs/guides/checkpointing.md)
- [Sampling/evaluation](docs/guides/sampling-and-evaluation.md) and [frontier sweeps](docs/guides/frontier-sweeps.md)
- [All component, method, modifier, and task references](docs/index.md)
- [Coverage and limitations](docs/validation/coverage.md) and [recorded results](docs/validation/reproduction-results.md)

For development, run `uv run pytest -q core/tests` and
`uv run ruff check core main.py frontier.py`. Legacy implementations remain in Git
history before the `candiv2.1` cleanup.
