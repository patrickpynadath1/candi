# Training

The root runner constructs the Task, shared data module, and Lightning
[`TrainingModule`](../../core/training/module.py), then trains the selected
algorithm. Config composition is described in [configuration](../concepts/configuration.md).

```bash
uv run python main.py
uv run python main.py +experiment=smoke
uv run python main.py +experiment=openwebtext lr_scheduler=cosine_decay_warmup
```

The default is MDLM on Sudoku hard: 20,000 optimizer steps, effective batch 256,
BF16 mixed precision, AdamW at `3e-4`, 2,500 warmup steps, and EMA decay `0.9999`.
The smoke experiment deliberately overrides these settings for a tiny CPU check.
Task availability and preparation requirements are listed in [tasks](../tasks/overview.md).

For Python-defined method grids and local GPU scheduling, see
[programmatic sweeps](programmatic-sweeps.md).

## Batches and optimization

`loader.global_batch_size` is the effective batch across devices and accumulated
microbatches. Per-device batch size is

$$
B_{\mathrm{local}}=B_{\mathrm{global}}/(\mathrm{world\ size}\times\mathrm{accumulation}).
$$

The division must be exact. `loader.eval_batch_size` controls each evaluation
batch separately. `trainer.max_steps` counts optimizer steps; gradient accumulation
is configured by `trainer.accumulate_grad_batches`.

AdamW fields are `optimizer.lr`, `beta1`, `beta2`, `epsilon`, and `weight_decay`.
Recipes can add `optimizer.groups` with parameter-name glob patterns and group
options. Every pattern must match, groups cannot overlap, and unmatched parameters
remain in the default group. See the [RePlaid recipe](../methods/replaid.md) for a
method using distinct optimizer groups.

## Learning-rate schedules

Learning-rate scheduling is independent of diffusion noise scheduling:

| Selection | Behavior |
| --- | --- |
| `lr_scheduler=constant_warmup` | Default: linear warmup from zero, then constant. Uses `training.warmup_steps`. |
| `lr_scheduler=constant` | Fixed LR from the first optimizer step. |
| `lr_scheduler=step_scheduler` | Warmup then halving every `lr_scheduler.n_halve_steps`; defaults 500/10,000 steps. |
| `lr_scheduler=cosine_decay_warmup` | Linear warmup then one cosine decay to an absolute floor. |

```bash
uv run python main.py lr_scheduler=cosine_decay_warmup \
  trainer.max_steps=20000 lr_scheduler.warmup_steps=2000
```

Cosine warmup defaults to 10% of `trainer.max_steps`; `warmup_steps` overrides that
fraction. Both `warmup_lr_init` and `lr_min` default to `1e-6`. Groups retain their
own peak LR and share those absolute endpoint values. After the horizon the floor
is held. The implementation is local: the original preset referred to an absent
`utils.CosineDecayWarmupLRScheduler`, so exact parity with that missing class is not
claimed. The step-halving formula matches the original implementation.

Implementation: [schedules.py](../../core/training/schedules.py). Tests cover values,
parameter groups, endpoints, and resume. Older native configs with no
`lr_scheduler` field retain the original constant-with-warmup behavior.

## Validation, generation, and logs

Training logs `train/loss`; validation aggregates token-weighted loss under
`val/nll`. That field is a historical name: for general objectives it is not
necessarily a negative log likelihood. Task generation metrics are separate.

- `trainer.val_check_interval` controls loss validation in Lightning's batch units.
  With the default `check_val_every_n_epoch=null`, it is not a simple synonym for
  optimizer-step count under gradient accumulation.
- `callback.every_n_steps` controls periodic generation in optimizer-step units;
  zero disables it. `callback.subset_size` and `callback.seed` fix its subset.
- `checkpoint.every_n_steps` saves periodic checkpoints, while the runner also
  retains the best `val/nll` checkpoint and writes `last.ckpt` at completion.
  Best loss and best task-generation accuracy are not necessarily the same model.

CSV logs live under the run's `metrics/` directory. Periodic task results live in
`task_callback/`. W&B is optional:

```bash
uv sync --locked --extra tracking
uv run --extra tracking python main.py '+wandb.project=candi' '+wandb.name=sudoku-hard'
```

Use `uv run --extra tracking` for W&B commands so uv keeps that optional dependency
selected. `training.ema=0` disables EMA; otherwise evaluation normally uses EMA.
Some method recipes intentionally choose different EMA settings.

## Resume and sweeps

```bash
uv run python main.py checkpoint.resume=/absolute/path/to/last.ckpt
uv run python main.py --multirun seed=1,2,3 \
  hydra.sweep.dir=core/outputs/training_seeds
```

Resume requires the matching original architecture and optimizer/scheduler recipe.
See [checkpointing](checkpointing.md) for saved configs, raw-vs-EMA selection, and
mid-epoch replay limits. Use [distributed training](distributed-training.md) for
DDP/FSDP and [frontier sweeps](frontier-sweeps.md) to evaluate pretrained models
without training.
