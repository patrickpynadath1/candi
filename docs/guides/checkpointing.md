# Checkpointing and weight selection

Training writes native Lightning checkpoints containing model/component state,
optimizer and scheduler state, loop progress, core format metadata, and EMA when
enabled. The core wrapper also saves Python, NumPy, CPU Torch, and current-device
CUDA RNG state. The runner writes a resolved `config.yaml` beside the checkpoints.

The runner saves periodic `step_*.ckpt` files, a best-loss checkpoint monitored on
`val/nll`, and `last.ckpt`. A best-loss checkpoint is not automatically the best
checkpoint for task generation quality.

## Continue training

```bash
uv run python main.py --config-path /absolute/path/to/training_run --config-name config \
  checkpoint.resume=/absolute/path/to/training_run/checkpoints/last.ckpt \
  trainer.max_steps=30000 \
  output_dir=/absolute/path/to/resumed_run hydra.run.dir=/absolute/path/to/resumed_run
```

Use the original architecture, modifiers, optimizer groups, and schedule recipe.
For horizon-dependent schedules, changing `max_steps` can change the future LR
curve: keep the original explicit scheduler horizon when you require continuation
of the same curve. The ordinary shuffled DataLoader does not persist a prefetch-aware
cursor, so exact mid-epoch data-stream replay is not guaranteed. Changing device
count also changes random-stream partitioning.

This path resumes training. The evaluation loader described below only loads
weights; it does not restore an optimizer or convert legacy training-loop state.

## Evaluate saved weights

```bash
uv run python main.py --config-path /absolute/path/to/training_run --config-name config \
  mode=evaluate evaluation.checkpoint=/absolute/path/to/training_run/checkpoints/last.ckpt \
  output_dir=/absolute/path/to/evaluation hydra.run.dir=/absolute/path/to/evaluation
```

A saved resolved config has a literal `output_dir`; overriding only `hydra.run.dir`
is insufficient to move its artifacts. Root runners do not automatically select
the saved architecture based on `evaluation.checkpoint` alone.

By default `evaluation.use_ema=true`. Set it to false to load raw/final weights,
including checkpoints trained with `training.ema=0`. When the configured module
has EMA enabled, loading rejects missing EMA state rather than silently falling
back. With `training.ema=0`, there is no EMA object and evaluation uses raw weights
even if `evaluation.use_ema=true`; select false explicitly to make that intent clear. Native state dicts are loaded
strictly, and EMA loading validates parameter counts/shapes. `results.json`
records the checkpoint path and whether EMA weights were used under `weights`.

Strict parameter loading cannot see parameter-free choices such as the noise
schedule, process or time conditioning. Native evaluation therefore compares the
checkpoint's saved `algorithm`, `block_diffusion`, `variable_length`,
`token_feedback` and `encoder_decoder` sections with the current config and
raises on any difference. Sampling knobs (`predictor.temperature`, `top_k`,
`top_p`, `commitment`), the time sampler and the training objective may differ.
Set `evaluation.allow_config_mismatch=true` only for a deliberate cross-recipe
evaluation. Legacy imports are not checked.

Run configs and checkpoints saved by a family class that has since become a preset
(the REPLAiD classes) are migrated automatically: the runner rebuilds the config
from the preset, keeping the saved task, data, trainer, evaluation and sampler
settings, and the loader renames moved state keys and migrates the checkpoint's
saved config before the recipe check ([legacy checkpoints](../../core/training/legacy_checkpoints.py)).

## Import supported legacy weights

| Setting | Supported import |
| --- | --- |
| `evaluation.legacy=false` | Native core checkpoint (`core_format_version=1`). |
| `evaluation.legacy=true` | Legacy MDLM backbone and optional EMA. |
| `evaluation.legacy=candi` | Legacy CANDI import with shared-embedding and EMA handling. |

These are explicit weight importers, not universal checkpoint converters. Match
the original model architecture and method recipe. The imported MDLM OpenWebText
checkpoint, for example, needs the 768-wide/12-layer model with AdaLN and its
original schedule endpoint settings.

The legacy unpickler maps known old Sudoku tokenizers and treats the removed
Transformers GPT2TokenizerFast pickle class as inert checkpoint metadata. The Task
constructs the actual tokenizer; it does not tokenize using that old object.
Checkpoint loading is intended for trusted local training artifacts.

## FSDP and sharing

The restricted FSDP preset writes full checkpoints that the standalone evaluator
can load without a distributed process group. It requires EMA disabled, so use
`evaluation.use_ema=false`. Its other limits are described in
[distributed training](distributed-training.md).

For a reproducible handoff, keep the checkpoint, resolved config, dependency lock,
code revision, task dataset/version, and evaluation settings together. Generated
checkpoints and caches are ignored by Git; absolute paths mentioned in historical
validation evidence are not bundled data.

Implementation: [checkpoint loader](../../core/training/checkpoints.py),
[training module](../../core/training/module.py),
[EMA](../../core/training/ema.py), and
[CANDI importer](../../core/training/candi_checkpoint.py).
