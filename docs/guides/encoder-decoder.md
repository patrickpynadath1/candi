# Encoder–decoder block diffusion

`EncoderDecoderModel` wraps an existing denoising backbone with a separate clean
encoder. `BlockDiffusion` still owns training layout, visibility, and blockwise
generation. Neither wrapper changes the underlying corruption or objective.

```yaml
# Compose +experiment=e2d2, then choose the task independently.
model:
  n_blocks: 2                 # decoder depth
encoder_decoder:
  _target_: core.modifiers.encoder_decoder.EncoderDecoderModel
  encoder_layers: 10
block_diffusion:
  enabled: true
  block_size: 32
  x0_causal: false
```

The encoder uses independent transformer layers with the decoder's width, head
count, dropout, and standard MLP. Input embeddings and the output projection stay
on the wrapped backbone. Each decoder layer projects the encoder's final states
into its own K/V space and attends jointly to clean context and noisy states.
Only earlier clean blocks are visible; clean encoder states cannot access noisy
inputs. All blocks train in parallel with gradients through the encoder.

This implements E2D2's final-hidden-state information flow on the local DiT,
not its Qwen3 backbone or shared-KV/pretrained variant. The current integration
supports fixed-length block layouts, standard MLP decoder layers, and no clean
auxiliary objective. Activation checkpointing applies to encoder and decoder.
The existing sampler controls the diffusion updates; this is not a new sampler.

During evaluation, caches persist across blocks for both ordinary block diffusion
and E2D2. Ordinary block diffusion appends each layer's clean K/V only for newly
completed tokens. E2D2 appends encoder K/V and projects only the new final encoder
states into each decoder layer's context K/V. Denoising steps reuse these caches;
the evolving noisy block is recomputed. A new generation starts a fresh cache.
Changed/truncated prefixes and extensions of incomplete bidirectional blocks
invalidate the cache. Caches are used only in evaluation with gradients disabled.
Training remains parallel and fully differentiable. Compilation targets tensor
training paths rather than mutable cache/layout bookkeeping.

For text8 and TinyStories, compare 10/2, 6/6, and 2/10 encoder/decoder depths at
fixed width. With the unconditioned DiT these have exactly equal parameter
counts. Hold data, block size, loss, training steps, effective batch, and sampler
search budget fixed. Record both validation loss and tuned generation scores,
as well as measured memory/time. Equal depth does not imply identical FLOPs:
decoder layers also project the encoder context.

BlockSampler reports total sequence NFE, summing calls over generated blocks.
A per-block sampling budget must not be reported as a whole-sequence budget.

### Raw embedding context (zero encoder)

`+experiment=raw_context` uses eight decoder layers and
`encoder_decoder.encoder_layers=0`. This is the zero-depth setting of
`EncoderDecoderModel`, not a separate execution path. For masked/token inputs,
clean context is the shared learned vocabulary embedding table. Decoder layers
normalize it, apply their own K/V projections and positional encoding, and use
joint attention over allowed clean context and noisy states. There is no extra
cross-attention sublayer or contextual encoder. Clean embeddings receive gradients.

The same block visibility mask hides current/future clean answers. Incremental
inference caches each decoder layer's clean K/V and projects only newly completed
blocks. The wrapper also supports the existing continuous input adapters; in that
case zero depth means the adapter's clean input representation, not necessarily a
vocabulary embedding lookup.

The [block-encoder-decoder](../../experiments/block-encoder-decoder/README.md) example
trains it (`denoising_task/block=raw_context`) next to the E2D2 depth arms, with the same training
and validation-only sampler tuning.

### Query-dependent mixtures of actual encoder K/V

`+experiment=encoder_kv_depth` selects `EncoderDepthAttention` inside the same
`EncoderDecoderModel` wrapper. It works with the existing masked, uniform and
continuous input adapters; neither the wrapper nor attention knows the corruption
process, objective, or vocabulary. Like the existing wrapper, it requires compatible
DDiTBlock widths/heads and positionwise MLP sublayers, with fixed-length blocks.

```yaml
encoder_decoder:
  _target_: core.modifiers.encoder_decoder.EncoderDecoderModel
  encoder_layers: 6
  context_attention:
    _target_: core.modifiers.encoder_decoder.attention.EncoderDepthAttention
    query_chunk_size: 32
```

The encoder returns its actual attention keys (already rotated with RoPE) and
values at every layer. No decoder-side context projections or additional rotation
are applied. Every decoder query/head selects depths separately for each context
position. If `s[j,l] = dot(q, k[j,l]) / sqrt(head_dim)`, the depth weights are
`a[j,:] = softmax(s[j,:])`. The context-position score is `sum_l a[j,l]*s[j,l]`.
These scores compete with the decoder's ordinary noisy-position scores in one
position softmax. The output sums encoder values with position weight times depth
weight, plus the noisy-position values. This is equivalent to constructing the
query-specific mixed K/V before position attention; it is not a joint softmax
across flattened position/depth pairs.

Block visibility is applied to position scores, preventing current/future clean
answers from contributing. The encoder's own block mask prevents indirect leakage
through deeper context representations. At generation time, encoder K/V grow only
when clean context grows; one shared depth cache serves every decoder layer.
Queries and depth mixtures are recomputed on every denoising pass.

This is a correctness-oriented PyTorch implementation. Query tiling limits the
size of each score allocation; it is not a fused FlashAttention kernel and does
not promise a speedup. Training should retain activation checkpointing because
otherwise autograd saves intermediate tiles. The optional attention uses FP32
scores/reductions and returns the query dtype under BF16 autocast.

Important depth semantics: layer l's K/V are projected from its **input**, before
that layer's attention/MLP updates. Thus those updates influence layer l+1's K/V.
The final encoder stage is therefore an `EncoderKeyValueReadout`: normalization
and K/V projection only. It has no query projection, attention-output projection,
MLP, or residual update. With `encoder_layers=L`, this variant has L-1 full encoder
blocks plus that terminal readout, still exposing L K/V levels. The readout follows
the same block execution contract, including checkpointing and incremental caching.
Standard final-hidden-state E2D2 retains L full blocks. At least one encoder level
is required. This changes parameter/state-dict layout relative to the initial
untrained depth-attention implementation; no existing training jobs use it.

Tests compare outputs and gradients with explicitly materialized query-dependent
mixtures, check the single-depth ordinary-attention limit, all-masked rows, answer
leakage, actual K/V provenance, cache parity, and loss/gradient execution for
masked, uniform and Gaussian corruption. This recipe has not been added to a
training sweep automatically.

### MDLM modulation and controlled comparisons

The MDLM recipe explicitly enables decoder `model.time.adaLN`. Its default
`algorithm.time_conditioning=false` supplies a constant timestep to the learned
modulation and residual gates; it does not remove those layers. E2D2, raw context,
and encoder-K/V recipes inherit this setting. Clean encoder blocks remain
unconditioned. Consequently, shifting a layer from encoder to decoder also adds
that decoder layer's modulation parameters; equal total layer counts are not
exactly equal parameter counts.

The November 1 block/E2D2 campaigns inherited `adaLN=false`. Their artifacts are
preserved as that variant. The November 2 `*-adaln-20k-*` campaigns restart from
scratch with AdaLN; they do not resume the incompatible older checkpoints.
