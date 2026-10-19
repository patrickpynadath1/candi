# Getting started

Run commands from the repository root. The project uses Python 3.10+ and uv.

```bash
uv sync --locked
uv run python main.py --cfg job
```

The lockfile uses the PyTorch CUDA 12.1 index. A CUDA-capable environment is needed
for the default GPU recipe; the smoke recipe below runs on CPU. The project does
not require FlashAttention or a sibling source checkout. For optional W&B logging:

```bash
uv sync --locked --extra tracking
```

For nested training/inference tuning, including BLEU and W&B, use the same extras
for installation, the Ray launcher, and the driver:

```bash
uv sync --locked --extra tuning --extra scoring --extra tracking
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning --help
```

Do not depend on temporary `PYTHONPATH` package directories. See
[experiment terminology](guides/experiments.md) and [nested tuning](guides/nested-tuning.md).

Dataset files, pretrained checkpoints, and external reference scoring models are
provided separately. See the individual [task pages](tasks/overview.md) for data
sources and offline options.

## First training run

```bash
uv run python main.py +experiment=smoke \
  hydra.run.dir=core/outputs/getting_started
```

This trains a tiny MDLM for two optimizer steps on eight Sudoku puzzles, validates
on two puzzles, and saves checkpoints. It tests execution, not generation quality.
The task can generate its own small dataset cache; no pretrained model is needed.

The run directory contains the resolved `config.yaml`, CSV metrics, and
`checkpoints/last.ckpt`. Data caches default to `core/data_cache/`; outputs default
to `core/outputs/`. Both paths are ignored by Git and may be symlinks to another
storage volume. Neither symlink target is part of the repository.

## Evaluate the saved model

Reuse the architecture settings from training:

```bash
uv run python main.py +experiment=smoke mode=evaluate \
  evaluation.checkpoint=core/outputs/getting_started/checkpoints/last.ckpt \
  hydra.run.dir=core/outputs/getting_started_evaluation
```

The output is `results.json`, including task metrics, generated records, and
network-evaluation counts. `mode=sample` uses this same evaluation pipeline; it is
not a separate metrics-free export path.

## Move to a real experiment

The default `uv run python main.py` is the 20,000-step MDLM Sudoku-hard training
recipe on one GPU. Inspect settings before launching:

```bash
uv run python main.py --cfg job
uv run python main.py +experiment=openwebtext --cfg job
uv run python main.py +experiment=gaussian_diffusion_sudoku --cfg job
```

Use [configuration](concepts/configuration.md) to understand overrides,
[training](guides/training.md) to choose optimizer and schedule settings, and
[validation coverage](validation/coverage.md) to see what has been verified.

## Development checks

```bash
uv run pytest -q core/tests
uv run ruff check core main.py frontier.py
```

CUDA-specific tests require visible GPUs. Core unit/integration tests use small
fixtures; they are not substitutes for a full training benchmark. The documented
[pretrained OpenWebText demonstration](validation/reproduction-results.md) uses
external cached weights and a real reference scorer.
