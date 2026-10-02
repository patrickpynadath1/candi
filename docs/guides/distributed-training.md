# Distributed training

DDP is the general multi-GPU path. The data module partitions the effective batch
by world size and gradient accumulation; Lightning distributes the DataLoader.
Training-time samplers receive rank and accumulation context. Adaptive schedules
with distributed feedback gather observations in their own implementations.

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python main.py strategy=ddp \
  trainer.devices=2 loader.global_batch_size=256 trainer.accumulate_grad_batches=4
```

That example uses local microbatch 32 on each of two GPUs. The global batch must
divide exactly by `world_size * accumulate_grad_batches`. Dataset size must support
one distributed microbatch. The DDP preset enables unused-parameter handling,
including the unused time embedder retained by some baseline architectures.

## Strategies and devices

| Selection | Behavior |
| --- | --- |
| No strategy group | Uses `trainer.strategy`, which defaults to `auto`. |
| `strategy=ddp` | Explicit Lightning DDP with `find_unused_parameters=true`. |
| `strategy=single_cpu` | Single-device CPU strategy; also select `trainer.accelerator=cpu`. |
| `strategy=single_gpu` | Single-device `cuda:0`, relative to visible devices. |
| `strategy=fsdp` | Restricted FSDP path described below. |

An explicitly selected `strategy` group takes precedence over `trainer.strategy`.
The older `trainer.strategy=ddp` override remains supported. Match the accelerator,
device count, and precision to the strategy. `trainer.num_nodes` is passed to
Lightning, but a multi-node launcher/environment is still required; no cluster
launcher or multi-node integration result is supplied by this repository.

Validation loss is summed across ranks before token-weighted reduction. Periodic
task generation runs on rank zero while other ranks wait at a barrier. That is
not distributed generation. Standalone evaluation also uses one
`evaluation.device`; assigning multiple trainer devices does not parallelize it.

## Restricted FSDP

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python main.py strategy=fsdp \
  trainer.devices=2 training.ema=0 callback.every_n_steps=0 \
  trainer.gradient_clip_val=0
```

The preset uses `SHARD_GRAD_OP`, original parameter objects, and full state dicts.
Current safeguards require:

- **No EMA:** the current EMA implementation does not track parameter shards.
- **No periodic generation:** rank-zero-only generation cannot execute sharded
  parameter collectives. Evaluate a saved full checkpoint separately.
- **No gradient clipping:** the installed Lightning FSDP precision plugin rejects
  norm clipping. The current guard requires the clipping value to be zero.
- **No post-optimizer projection hooks:** methods with `post_optimizer_step`
  constraints require full parameters and should use DDP.

These restrictions are checked before fitting. Standalone evaluation of FSDP
checkpoints needs `evaluation.use_ema=false`. This is not a claim that every
algorithm, adaptive schedule, or modifier combination supports FSDP.

## Verified scope

A tiny MDLM two-GPU DDP run completed four optimizer steps with cosine LR scheduling,
EMA, periodic generation, and checkpoint saving. Restricted FSDP completed two
steps, resumed to four, and produced a full checkpoint that loaded in the standalone
evaluator without a distributed process group. These are execution checks, not
multi-GPU convergence benchmarks.

When introducing a new component with its own parameters/buffers or distributed
feedback, test backward, optimizer updates, checkpointing, and all-rank participation.
A single-GPU success does not establish distributed compatibility. See
[coverage](../validation/coverage.md), [checkpointing](checkpointing.md), and the
[strategy safeguards](../../core/training/strategies.py).
