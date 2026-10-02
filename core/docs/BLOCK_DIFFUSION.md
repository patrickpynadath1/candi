# Block diffusion modifier

Block diffusion composes with an existing diffusion algorithm. The underlying
algorithm continues to own corruption, schedules, posterior parameterization,
objectives, and reverse-time mathematics. Enable it with
`block_diffusion.enabled=true` and set `block_diffusion.block_size`.

The model evaluates a causal clean stream and a noisy stream with the same
transformer weights. A noisy position attends every noisy position in its own
block and clean positions in strictly earlier blocks. Clean position `i` sees
only clean positions through `i`, and never sees noisy positions. Both streams
reuse original position IDs. Consequently a block at least as large as the
sequence has the same noisy-stream computation as the unmodified noncausal
model. Future prompt positions follow the same strict block-causal rule and are
invisible until their block becomes active.

`clean_loss=autoregressive` adds shifted next-token cross entropy from raw clean
stream vocabulary logits. `clean_loss_weight` controls its contribution and
defaults to `1.0`. Task
loss masks select valid shifted targets. With `clean_loss=none`, the diffusion
loss still backpropagates through earlier clean blocks.

Generation wraps any compatible sampler and completes blocks from left to
right. Only the active block evolves; prompt and completed positions remain
fixed. Self-conditioning state is local to each inner sampler invocation and is
therefore reset between blocks. Clean tokens are re-encoded through the clean
stream; noisy-state cache entries are never reused as clean context.

The initial supported models are `DiT` (including per-position adaLN conditioning)
and `ContinuousDiT`, covering MDLM and Gaussian diffusion. During generation,
`cache_clean_prefix=true` materializes clean-stream keys and values once per block;
the uncached mode recomputes them on every model evaluation. Two-stream attention
roughly doubles sequence length during training,
so dense attention can use about four times the attention memory.

RePlaid is rejected explicitly in this first version: its singleton-expanded
loss, detached embedding gradient routes, and named optimizer groups require a
dedicated integration. Other model classes without the explicit stream methods
are rejected at construction rather than run with an approximate adapter.
