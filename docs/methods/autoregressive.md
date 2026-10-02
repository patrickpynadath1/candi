# Autoregressive models

Ordinary and permuted autoregressive models share CausalDiT, posterior shaping, CE, and a KV-cached sampling implementation. They do not need a diffusion process or noise schedule. Their training and positional semantics differ.

## Ordinary autoregressive

```bash
uv run python main.py +experiment=autoregressive
uv run python main.py +experiment=autoregressive task=text8
```

Training predicts `tokens[:, 1:]` from `tokens[:, :-1]` under causal attention, applying the correspondingly shifted loss mask. The sampler prefills the prompt and then processes new tokens using cached attention keys/values. `sampler.use_cache=false` provides a full-prefix reference path. Shaping remains in the posterior parameterization.

## Permuted autoregressive

```bash
uv run python main.py +experiment=permuted_autoregressive task=text8 data.download=true
uv run python main.py +experiment=permuted_autoregressive_text8
```

Position zero supplies BOS. Remaining targets are permuted and shifted to construct training inputs; the loss mask follows the target permutation. Attention is causal in generation order. RoPE keys identify the current token's spatial position, while queries identify the next target's spatial position. Cached keys are already rotated.

The order component supports sequential, random, and blockwise random policies, with `ar_probability` mixing sequential rows. Training and sampling policies are independent:

```bash
uv run python main.py +experiment=permuted_autoregressive task=text8 \
  generation_order.mode=block generation_order.block_size=16 \
  generation_order.ar_probability=0.1 \
  sampler.order.mode=block sampler.order.block_size=16
```

This ordering policy is different from the [block diffusion modifier](../modifiers/block-diffusion.md).

The cached sampler evaluates BOS once and one new token per step, scattering sampled tokens into spatial positions. EOS does not terminate arbitrary-order generation: every target position must be filled. Uncached generation recomputes the entire ordered prefix and serves as a correctness check.

## Prompt conditioning

The reference permuted preset leaves observed-token conditioning disabled. Set `algorithm.condition_on_observed=true` consistently for training and inference to enable it. Training uses the corruption mask to place observed tokens before targets; inference visits fixed prompt tokens first and preserves their values. Noncontiguous observations are supported by this explicit extension. Ordinary AR remains causal in its standard sequence order.

These algorithms do not use the diffusion prediction-session API; do not assume the diffusion self-conditioning modifier applies merely because both use a posterior. See [component contracts](../concepts/component-contracts.md).

## Validation

Tests check explicit attention equations, arbitrary query/key positions, chunked caches, causal visibility, cached/uncached equality, masking, generation orders, prompt handling, restoration, and BF16 execution.

Permuted AR trained on real Text8 for 10,000 steps with a width-256, four-layer model and generated all 255 target positions using KV caching. Of 5,500 generated words, 3,751 were valid against the training vocabulary (68.20%); the rate for words of length at least seven was 9.07%. Samples were accepted as an initial port check, not a large-model quality benchmark. No Sudoku training check was requested for this method. See [reproduction results](../validation/reproduction-results.md).

Sources: [ordinary algorithm](../../core/algorithms/autoregressive.py), [permuted algorithm](../../core/algorithms/permuted_autoregressive.py), [causal model](../../core/models/causal.py), [generation orders](../../core/generation_order/).
