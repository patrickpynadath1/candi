# CANDI v2.1

Composable diffusion models for discrete data. Implementations, tasks, Hydra
configs, and tests live in `core/`; the root scripts provide the entrypoints.
There are no dependencies on the removed legacy algorithm folders.

## Setup

```bash
uv sync --locked
# Optional experiment tracking:
uv sync --locked --extra tracking
```

Python 3.10+ is required. The checked-in lock uses the PyTorch CUDA 12.1 index.
Datasets, pretrained checkpoints, and reference scoring models are external
inputs; they are not included in the repository. Outputs and data caches are
ignored by Git. Existing cache symlinks can be reused locally.

## Train, evaluate, and sample

```bash
# CPU smoke run
uv run python main.py +experiment=smoke

# MDLM on Sudoku hard
uv run python main.py

# Two-GPU DDP
CUDA_VISIBLE_DEVICES=0,1 uv run python main.py strategy=ddp trainer.devices=2

# Choose a task/recipe and learning-rate schedule
uv run python main.py +experiment=openwebtext lr_scheduler=cosine_decay_warmup

# Standalone evaluation: use the checkpoint's architecture/recipe
uv run python main.py mode=evaluate evaluation.checkpoint=/path/to/last.ckpt

# Training sweep
uv run python main.py --multirun seed=1,2,3
```

`mode=sample` uses the same generation and task-metric pipeline as evaluation.
For resume, pass `checkpoint.resume=/path/to/last.ckpt` with matching training
settings. Select experiments and component groups from `core/new_configs/`.

## Pretrained frontier sweeps

```bash
uv run python frontier.py \
  evaluation.checkpoint=/path/to/last.ckpt \
  hydra.run.dir=core/outputs/frontiers/my_sweep
```

The default preset sweeps Sudoku EB temperature and entropy bounds. Text8 and
OpenWebText presets, custom axes, restart behavior, and generated JSON/CSV/SVG
reports are described in [Frontier sweeps](core/docs/FRONTIER_SWEEPS.md).

## Tests and documentation

```bash
uv run pytest -q core/tests
uv run ruff check core main.py frontier.py
```

- [Detailed recipes and component guide](core/README.md)
- [Architecture and ownership](core/ARCHITECTURE.md)
- [Training schedules, DDP, and restricted FSDP](core/docs/TRAINING_CONFIGURATIONS.md)
- [Validation evidence](core/VALIDATION.md)

Currently implemented tasks are Sudoku, Text8, OpenWebText, and TinyGSM. FSDP has
restrictions documented above; DDP is the general multi-GPU training path.
Historical implementations and scripts remain available in Git history.
