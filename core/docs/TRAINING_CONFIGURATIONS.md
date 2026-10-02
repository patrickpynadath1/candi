# Training configurations

Learning-rate scheduling is separate from the diffusion noise schedule. Select:

| Config | Behavior |
| --- | --- |
| `lr_scheduler=constant_warmup` | Existing default: linear warmup from zero using `training.warmup_steps`, then constant LR. |
| `lr_scheduler=constant` | Fixed optimizer LR from the first step. |
| `lr_scheduler=step_scheduler` | Warmup, then halve every `lr_scheduler.n_halve_steps` optimizer steps. Matches the original LRHalveScheduler. |
| `lr_scheduler=cosine_decay_warmup` | Linear warmup, then one cosine decay to an absolute LR floor. |

The cosine defaults use 10% of `trainer.max_steps` for warmup and absolute
`warmup_lr_init=1e-6`, `lr_min=1e-6`, matching the intent of the original preset.
Its old `utils.CosineDecayWarmupLRScheduler` target is absent in this checkout;
this is a local implementation, not a claim of parity with that missing class.
Set `lr_scheduler.warmup_steps` to override the fraction. Parameter groups keep
their configured peak learning rates and share the absolute floor. All schedulers
advance per optimizer step, not per accumulated microbatch. AdamW settings and
optional parameter-name groups remain under `optimizer`.

```bash
uv run python main.py lr_scheduler=cosine_decay_warmup \
  trainer.max_steps=20000 lr_scheduler.warmup_steps=2000

CUDA_VISIBLE_DEVICES=0,1 uv run python main.py strategy=ddp \
  trainer.devices=2 loader.global_batch_size=256 trainer.accumulate_grad_batches=4
```

Strategy presets are `strategy=ddp`, `strategy=single_cpu`, `strategy=single_gpu`,
and restricted `strategy=fsdp`. The default leaves Lightning's
`trainer.strategy=auto` in effect. Existing `trainer.strategy=ddp` overrides still
work. A selected `strategy` group takes precedence over `trainer.strategy`.
Match `trainer.accelerator` and `trainer.devices` to the strategy. DDP retains
unused-parameter handling and the existing EMA, periodic generation, and global
batch partitioning. Global batch must divide by world size times accumulation.

## Restricted FSDP

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python main.py strategy=fsdp \
  trainer.devices=2 training.ema=0 callback.every_n_steps=0 \
  trainer.gradient_clip_val=0
```

The preset uses `SHARD_GRAD_OP`, original parameters, and full state-dict
checkpoints. It currently requires:

- EMA disabled: the existing EMA implementation does not track shards.
- Periodic generation disabled: rank-zero-only sampling cannot execute the
  collectives needed by sharded weights. Evaluate saved full checkpoints in a
  separate process instead, with `evaluation.use_ema=false`.
- Gradient clipping disabled: the installed Lightning FSDP precision plugin
  rejects norm clipping.
- No `post_optimizer_step` parameter projection hooks. These require full
  parameters; methods with those hooks should use DDP.

These restrictions are checked before fitting. A tiny MDLM integration check is
not evidence that every algorithm/modifier supports FSDP. DDP remains the general
multi-GPU choice.

Validation: a two-GPU MDLM DDP run completed four optimizer steps with cosine
scheduling, EMA, periodic generation, and checkpoint saving. The restricted
FSDP preset completed two steps, resumed to four, and its full checkpoint loaded
successfully in the standalone evaluator without a distributed process group.

## Resume and older checkpoints

Use `checkpoint.resume=/absolute/path/to/last.ckpt` with the original training
configuration. Lightning restores optimizer and scheduler state; changing the
scheduler recipe while resuming is not supported. Older native configs without
an `lr_scheduler` group retain constant-with-warmup behavior. Mid-epoch exact
DataLoader replay and import of legacy optimizer state remain outside core's
current resume guarantees.
