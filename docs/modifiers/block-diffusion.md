# Block diffusion

Block diffusion wraps a compatible algorithm instead of defining a separate
corruption method. The underlying algorithm retains its process, schedule,
predictor, and objective. The modifier changes model visibility, provides clean
context, and optionally adds a clean autoregressive loss.

```bash
uv run python main.py block_diffusion.enabled=true block_diffusion.block_size=32
uv run python main.py block_diffusion.enabled=true block_diffusion.block_size=32 \
  block_diffusion.clean_loss=autoregressive block_diffusion.clean_loss_weight=1.0
uv run python main.py +experiment=insertion_continuous task=text8 \
  block_diffusion.enabled=true block_diffusion.block_size=32 \
  block_diffusion.max_length=64
```

Defaults are disabled, block size 32, `clean_loss=none`, weight 1, and
`cache_clean_prefix=true`. Construction happens in the shared training/checkpoint
module, so restored evaluation uses the same wrapper structure.

## Training visibility

For fixed-length block diffusion, the model evaluates clean and noisy streams
with shared transformer weights and original position IDs:

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

With the variable-length modifier, source blocks are logical edit domains.
Training and validation corrupt every source block independently and process
all blocks in parallel through one shared backbone pass. Noisy positions see
only their own noisy block and clean tokens from earlier blocks. Every target
contributes to the same token-normalized loss; there is no single-block training
mode. Blocks from one example share its sampled global time, preserving
antithetic/DDP/gradient-accumulation partitioning in original-example units.
Self-conditioning uses the same isolation in its bootstrap and final passes.
Existing checkpoint weights remain loadable, but resuming an older block-insertion
checkpoint now uses all-block training; there is no compatibility switch for the
old cycling behavior. Training throughput and supervised tokens per step therefore
differ from historical one-block runs. During
generation, reverse insertion and content denoising act only inside the active
block; its result is frozen before the next block begins. Deletion is the forward
training corruption, not a learned removal operation during generation. `block_diffusion.max_length` is the capacity of
each generated block (default: `block_size`), rather than the full sequence
capacity. Sampling reports `block_lengths` for each example, and their sum is the
reported generated length. The inner insertion sampler's `max_length` remains
an additional per-invocation bound; use the smaller of the two limits for the
active block, rather than treating either as a total-output limit.

A block is completed when its reverse time grid reaches the endpoint and final
content decoding runs. There is no learned block-stop decision and no enforced
terminal token count. The clock resets for the next block. Thus `block_size=32`
does not guarantee a generated length of 32: it defines the clean training span
and the number of logical blocks in the requested generation window. The Text8
presets run eight blocks for a 256-token window, each capped at 64 tokens. A
fixed 32-token terminal block would require a length-conditioned insertion rule
that is not currently implemented.

The [insertion/deletion reference](variable-length.md) explains the structural
hazard, per-process birth laws, adaptive-time feedback and Text8 pilot evidence.
Use these self-conditioned Text8 recipes:

```bash
uv run python main.py +experiment=text8_block_insertion_masked_sc_22m
uv run python main.py +experiment=text8_block_insertion_gaussian_sc_22m
```

With clean-prefix caching, the model materializes clean-stream keys and values
once per block. Without it, clean context is recomputed on each model call.
Completed outputs are encoded through the clean stream; noisy KV entries are
never recycled as clean context. This is separate from the ordinary autoregressive
model's KV cache.

Variable blocks currently re-encode their completed prefix on every model call,
even when `cache_clean_prefix=true`. Earlier edits shift physical positions, so
reusing a fixed-position prefix cache would be incorrect. Generation runs each
example independently to keep its clean prefix free of storage padding. Fully
protected blocks are copied without a model call or insertion opportunity.

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

Variable-length block training currently requires packed sequences: every input
position must be valid. Ragged batches fail explicitly rather than embedding
padding as clean context. The optional clean autoregressive loss is also rejected
for variable blocks because its shifted targets are not defined across changing
block lengths.

To support a new architecture, implement clean/noisy stream embedding, hidden
processing, output projection, position handling, and cache semantics. For a
sequence-coupled process, consume explicit sequence boundaries and test that one
block cannot influence another through corruption. Verify causal visibility,
clean/noisy cached equivalence, prompt preservation, optional-loss masking, and
self-conditioning composition. See [modifier implementation](../../core/modifiers/block_diffusion/)
and [coverage](../validation/coverage.md).
