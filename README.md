# CANDI v2.1

Composable diffusion models for discrete data. Implementations and tests live in
`core/`, Hydra configs in `configs/`, and documentation in [`docs/`](docs/index.md).
The root `main.py` runs training/evaluation; `frontier.py` evaluates pretrained
sampling grids. No removed legacy source checkout is required.

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
