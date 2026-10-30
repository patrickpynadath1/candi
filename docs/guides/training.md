# Training

The root runner constructs the Task, shared data module, and Lightning
[`TrainingModule`](../../core/training/module.py), then trains the selected
algorithm. Config composition is described in [configuration](../concepts/configuration.md).

```bash
uv run python main.py
uv run python main.py +experiment=smoke
uv run python main.py +experiment=openwebtext lr_scheduler=cosine_decay_warmup
```

The default is MDLM on Sudoku hard: 200,000 training puzzles, 10,000 validation
puzzles, 50,000 optimizer steps, effective batch 256,
BF16 mixed precision, AdamW at `3e-4`, 2,500 warmup steps, and EMA decay `0.9999`.
The smoke experiment deliberately overrides these settings for a tiny CPU check.
Task availability and preparation requirements are listed in [tasks](../tasks/overview.md).

For Python-defined method grids and GPU scheduling through Ray Tune, see
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

When W&B is configured, periodic task evaluations log scalar metrics to every
logger, plus a `<task>/samples` table and the complete evaluation JSON as a W&B
artifact. The resolved training configuration and final checkpoint/step summary
are recorded too. Install the optional SDK with `uv sync --extra tracking`, and
retain `--extra tracking` when invoking `uv run`.

Metric names are not identical to the legacy trainer: training loss is
`train/loss` (formerly `trainer/loss`), while periodic generation metrics use the
task name as their prefix. `val/nll` is the validation objective's historical
label; it is not necessarily a log likelihood for every method.
## Intermediate token feedback

An optional single-pass token-feedback branch can be enabled with
`token_feedback.enabled=true`. After backbone block `token_feedback.depth`
(zero based), one additional transformer block predicts auxiliary clean-token
logits through the model's shared token output head. Their softmax expectation
under the input token embedding table is added to the residual stream through
the same zero-initialized, bias-free linear projection used by posterior-mean
self-conditioning (`ConditioningProjection`). No LayerNorm is applied to this
feedback. The main objective is
unchanged and training adds masked clean-token cross entropy weighted by
`token_feedback.loss_weight`. Prompt and padding positions remain excluded by
the existing `loss_mask`.

The branch stays active during inference, while its auxiliary loss is only
computed for training loss calls. Gradients flow through the softmax and token
embedding by default; `token_feedback.detach=true` provides a feedback-path
ablation without detaching the auxiliary loss. The current implementation uses
a fixed configured depth for deterministic behavior across distributed ranks;
random training depth is not supported. Models whose output is a score or
velocity require an explicit token readout and are rejected rather than treated
as token logits. Cached causal decoding is not currently supported.

### Self-conditioning gradient routing

`self_conditioning.gradient_mode` controls training-time bootstrap gradients:

- `detached` (default): only the conditioned consumer pass receives gradients.
- `both`: gradients flow through the bootstrap, expected embedding, and consumer.
- `producer`: conditioned examples train the bootstrap through a consumer whose
  parameter values are detached. Its input Jacobian remains differentiable.
- `mixed`: randomly assign half the SC-selected examples in each microbatch to
  `producer`, with the remainder using `detached` (an odd extra example uses
  `detached`). Examples not selected for SC train normally.

These modes retain the existing SC selection probability and do not change
sampling. Producer/mixed execution evaluates frozen and trainable consumers;
its training compute is greater than ordinary detached SC. Frozen here means
no direct parameter gradients from that consumer call, not a separate model
held fixed over the entire training run: both passes share current weights.

A producer-only branch cannot learn through a zero-initialized SC projection.
For hybrid/single-view models, `model.conditioning_initialization=identity`
provides an initially active projection. Compare initialization as well as
routing when interpreting results against zero-initialized baselines.

### Compilation

The shared runner enables `compile: true` by default, including for older queued
configs that omit the field. Each transformer block uses in-place
`Module.compile(dynamic=False)` with PyTorch's default Inductor backend. This
preserves parameter identities and checkpoint keys, including for resumed
training and EMA. It applies to training and generation; set `compile: false`
for eager execution. Already-running workers retain their execution mode until
their job ends; newly started Ray workers pick up the runner setting.
