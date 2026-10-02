# Block diffusion

Block diffusion wraps a compatible algorithm instead of defining a separate
corruption method. The underlying algorithm retains its process, schedule,
predictor, and objective. The modifier changes model visibility, provides clean
context, and optionally adds a clean autoregressive loss.

```bash
uv run python main.py block_diffusion.enabled=true block_diffusion.block_size=32
uv run python main.py block_diffusion.enabled=true block_diffusion.block_size=32 \
  block_diffusion.clean_loss=autoregressive block_diffusion.clean_loss_weight=1.0
```

Defaults are disabled, block size 32, `clean_loss=none`, weight 1, and
`cache_clean_prefix=true`. Construction happens in the shared training/checkpoint
module, so restored evaluation uses the same wrapper structure.

## Training visibility

The model evaluates clean and noisy streams with shared transformer weights and
original position IDs:

- A noisy position sees every noisy position in its own block and clean positions
  in strictly earlier blocks.
- Clean position `i` sees clean positions through `i`, never noisy positions.
- Future prompt positions follow the same block-causal rule; conditioning does not
  make them visible early.

A block covering the whole sequence gives the noisy stream no clean-prefix
attention and the same visibility as the unmodified noncausal model. The two-stream
training sequence is roughly twice as long, so dense attention can use about four
times the attention memory.

`clean_loss=autoregressive` computes shifted next-token CE from raw clean-stream
logits. Task loss masks select shifted targets, and `clean_loss_weight` scales its
contribution. `none` adds no auxiliary objective, but diffusion gradients still
flow through earlier clean context. This loss is distinct from sampler-time
posterior shaping.

## Sampling and state

[BlockSampler](../../core/modifiers/block_diffusion/sampler.py) completes blocks
left-to-right using the inner method's sampler. Only the active block evolves;
prompt and completed tokens remain fixed. NFE is accumulated from the block runs.
Self-conditioning sessions reset between inner sampler invocations.

With clean-prefix caching, the model materializes clean-stream keys and values
once per block. Without it, clean context is recomputed on each model call.
Completed outputs are encoded through the clean stream; noisy KV entries are
never recycled as clean context. This is separate from the ordinary autoregressive
model's KV cache.

## Sequence-aware processes and hybrids

[BlockDiffusion](../../core/modifiers/block_diffusion/training.py) supplies a
[SequenceLayout](../../core/sequences.py) when an algorithm advertises
`supports_sequence_layout`. Blurring uses these boundaries to apply its spectral
corruption independently to each block. The process receives a general sequence
layout rather than learning about block diffusion inside its mathematics.

For hybrids, the wrapper surrounds the fused model/algorithm: all branch states
for the active block are handled together, with shared block visibility and one
self-conditioning vector. Adding a branch does not create a separate clean stream.

## Current constraints and extension

The [model wrapper](../../core/modifiers/block_diffusion/model.py) accepts DiT,
ContinuousDiT, and HybridModel implementations (including supported subclasses)
with the required fused-stream interfaces. Unsupported architectures fail during
construction. This check is an integration boundary, not proof that every
algorithm/model combination has been numerically validated.

RePlaid is explicitly rejected: its specialized embedding-gradient routes and
optimizer parameter groups require a dedicated integration. Do not bypass the
check by wrapping only its attention layer.

To support a new architecture, implement clean/noisy stream embedding, hidden
processing, output projection, position handling, and cache semantics. For a
sequence-coupled process, consume explicit sequence boundaries and test that one
block cannot influence another through corruption. Verify causal visibility,
clean/noisy cached equivalence, prompt preservation, optional-loss masking, and
self-conditioning composition. See [modifier implementation](../../core/modifiers/block_diffusion/)
and [coverage](../validation/coverage.md).
