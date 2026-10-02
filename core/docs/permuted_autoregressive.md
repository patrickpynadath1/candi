# Permuted autoregressive

`PermutedAutoregressive` trains a causal token model in a sampled spatial order.
It has no masking process or diffusion schedule. The public component and
experiment name is `permuted_autoregressive`.

```bash
python main.py +experiment=permuted_autoregressive task=text8 data.download=true
```

The default reproduces the reference random permutation training: retain the
first token as BOS, permute all remaining targets, and shift the permuted targets
to obtain model inputs. Loss masks follow targets. A token at spatial position
`i` is a key at position `i`; its query uses the position of the *next target*.
Attention is causal in generation order, independently of these RoPE positions.

## Components

- `generation_order.GenerationOrder` owns sequential, random, and blockwise
  random policies. `ar_probability` mixes sequential rows into a policy.
- `algorithms.permuted_autoregressive` builds training inputs, positions, and
  masks, then coordinates the causal model, posterior, and cross entropy.
- `models.causal.CausalDiT` accepts distinct query/key position IDs. Its ordinary
  autoregressive path is unchanged. Cached keys are already position-rotated.
- `samplers.permuted_autoregressive` owns order selection, model calls, token
  scattering, and the KV cache. Predictor shaping is reused without duplication.

Training and sampling orders are independent:

```bash
python main.py +experiment=permuted_autoregressive task=text8 \
  generation_order.mode=block generation_order.block_size=16 \
  generation_order.ar_probability=0.1 \
  sampler.order.mode=block sampler.order.block_size=16
```

`sampler.use_cache=true` is the default. Sampling evaluates BOS once and then
one new token at each step. `sampler.use_cache=false` recomputes the entire
ordered prefix as a correctness/reference option. Both fill all requested spatial
positions: EOS does **not** terminate arbitrary-order generation. The task's
normal token decoding and word metrics interpret the completed sequence.

Prompt conditioning is an explicit extension, disabled in the reference default.
Enable `algorithm.condition_on_observed=true` for both training and evaluation.
The training corruption mask identifies observed tokens; the order policy moves
these ahead of predicted tokens. Sampling likewise visits fixed prompt tokens
first, retaining their values, and then fills the remaining positions. Position
zero supplies BOS (or the caller's fixed token). This supports noncontiguous
observations; callers must use matching conditioning semantics during training.

## Validation

Tests cover policy permutations/block boundaries/sequential mixture, shifted
training CE and masking against the reference equations, an independent explicit
matrix-attention reference, arbitrary query/key positions with chunked KV caches,
causal visibility, cached-versus-uncached generation, prompt preservation, and
checkpoint state restoration. CUDA checks include bfloat16.

The real Text8 validation run uses the unmodified random-order policy, a 256-wide
four-layer model, length 256, batch 128, learning rate 0.0003, 200 warmup steps,
and EMA decay 0.999. An initial 3,000-step checkpoint was continued to 10,000
steps with optimizer, scheduler, and EMA restored. Its artifacts are under
`core/outputs/permuted_autoregressive_text8/`; this is a bounded training check,
not a quality-equivalent reproduction of a large-model training budget.
