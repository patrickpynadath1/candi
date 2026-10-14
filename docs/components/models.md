# Models

Recipe context: [denoising task](../concepts/denoising-task.md).

Models own neural architecture and learned parameters. They consume prepared
states and conditioning coordinates and return raw vocabulary logits. Posterior
constraints, softcapping, temperature, and token filtering belong to
[predictors](predictors.md).

The common transformer code is in [layers.py](../../core/models/layers.py).
Models that support self-conditioning can return
[ModelPrediction](../../core/models/prediction.py), containing logits and the
final hidden state. Ordinary calls return logits; feature requests support
modifiers without requiring an extra inference pass.

## Configured architectures

| `model` selector | Input and purpose |
| --- | --- |
| `dit`, `dit_conditioned` | Token IDs through DiT embeddings; the latter enables AdaLN time conditioning |
| `continuous_dit` | `[B,L,D]` continuous states, time, sigma, and alpha |
| `sphere_dit` | Spherical states and spherical conditioning |
| `simplex_dit` | Simplex state input, with expectation/argmax input options in the implementation |
| `causal_dit` | Causal token prediction with explicit KV-cache support |
| `hybrid_candi`, `hybrid_uniform_simplex` | Representation-owned input mappings, fusion, and one shared DiT backbone |

See [model configs](../../configs/model/) and
[implementations](../../core/models/). Hidden width, head count, block count,
dropout, and state dimension are architecture choices; a checkpoint must have
matching parameter shapes. The task vocabulary and absorbing mask vocabulary
are distinct and supplied by configuration.

## Gaussian input choices

[ContinuousDiT](../../core/models/continuous_dit.py) by default retains both a learned state
embedding transform and a subsequent input projection for one-hot and embedding
representations. Neither is the representation's embedding table.

`model.input_preconditioning=gaussian` scales transformed state features by
`1 / sqrt(alpha**2 + sigma**2)` before the affine input projection.
`none` disables this scaling, retaining the learned transforms. Alpha and sigma
cannot both vanish.

`model.time_conditioning` chooses the scalar entering the timestep embedder:

| Value | Coordinate |
| --- | --- |
| `log_one_minus_time` (default) | `-log(1-time)` |
| `time` | Supplied time |
| `log_sigma` | `log(sigma)` |
| `log_effective_noise` | `log(sigma)-log(alpha)` |
| `log_snr` | `2*(log(alpha)-log(sigma))` |
| `negative_log_snr` | Negative of the preceding coordinate |

Log coordinates clamp coefficients at epsilon, and the default time coordinate
clamps below one. Away from these clamps, Gaussian preconditioning combined with
`log_effective_noise` or `log_snr` gives the same model input under common positive
rescaling of state, alpha, and sigma. `log_sigma` changes under that rescaling.
These choices do not change process coefficients or their derivatives.

RePlaid additionally selects `input_architecture=linear`,
`output_architecture=adaln`, `time_input=true`, and
`input_preconditioning=unit_embedding`. That preconditioner uses
`1/sqrt(alpha**2/D + sigma**2)` for state dimension `D` and requires the linear
input architecture. `conditioning_scale=unit_embedding` scales expected-embedding
feedback by `sqrt(D)`. These are explicit recipe choices; the standard Gaussian
presets keep their additional embedding projection. Architecture changes alter
checkpoint shapes and keys, unlike merely switching a conditioning coordinate.

## Caches and modifiers

[CausalDiT](../../core/models/causal.py) owns KV-cache mechanics. The autoregressive
samplers own append order and lifetime; permuted generation passes query/key
positions separately from causal generation order.
[Block diffusion](../modifiers/block-diffusion.md) needs explicit clean/noisy
stream interfaces and has its own clean-prefix cache. Do not reuse noisy keys
as clean context. [Self-conditioning](../modifiers/self-conditioning.md) injects
one detached feedback tensor using the model's conditioning adapter; a hybrid
also uses one shared feedback vector, not one per branch.

To add a model, expose the inputs its algorithm prepares and return the expected
vocabulary dimension. Implement optional hidden-feature and stream interfaces
only when their semantics are defined, and test cached/uncached equivalence where
applicable. A new architecture alone does not establish compatibility with every
modifier.

## Shared backbone and replaceable components

[TextDiT](../../core/models/text_dit.py) shares transformer execution across the
public construction recipes. [Components](../../core/models/components.py)
separate token/simplex/dense input mapping, time embedding, block execution, and
output projection. `DiT`, `ContinuousDiT`, `SphereDiT`, and `CausalDiT` retain their
public signatures and learned parameter layouts. `SimplexDiT` selects a simplex
input component, and hybrid fusion continues to enter the backbone through
`forward_hidden`.

The constructors accept `input_adapter`, `output_adapter`, and `block_execution`;
noncausal models also accept `time_adapter`. Hydra recursively instantiates these
arguments. For example, ordinary DiT can consume simplex log probabilities without
a new subclass:

```yaml
_target_: core.models.dit.DiT
vocab_size: 10
hidden_size: 128
n_heads: 4
n_blocks: 4
input_adapter:
  _target_: core.models.components.SimplexExpectationInput
```

This changes the model input only; select an algorithm/process that supplies
simplex states separately. It does not make MDLM's integer input a simplex state.

Replacement components follow the relevant call signature:

| Component | Inputs |
| --- | --- |
| Token/simplex input | State and the model's `vocab_embed` module |
| Continuous input | State, projection, embedding tensor, optional preconditioner |
| Time adapter | Prepared scalar coordinate and timestep embedder |
| Layer output | Hidden state, condition, and output layer |
| Embedding output | Hidden state, condition, projection, embedding table, bias |
| Diffusion block execution | Hidden state, blocks, RoPE, condition, attention mask |
| Causal block execution | Hidden state, blocks, RoPE, KV cache, query RoPE |

The defaults are parameter-free strategies: learned tensors stay registered at
historical model paths, avoiding checkpoint migrations and preserving shared
embedding identities. A custom learned adapter can register its own parameters;
that intentionally changes the checkpoint schema. Input/output replacements must
match the learned architecture selected by the recipe. For example, an embedding
output adapter needs the projection and table created by
`output_architecture=embedding_projection`.

The shared backbone preserves differences that matter: continuous execution
turns off autocast weight caching for detached bootstraps; spherical models can
retain their learned temperature output head; causal execution uses a distinct
block policy and explicit query/key positions. There is no attempt to substitute
ordinary diffusion attention for causal KV-cache behavior. Representation
encoding/decoding remains outside these input components, and Gaussian models
retain the additional input projection.

### Intermediate token-feedback gradients

Set `token_feedback.detach=true` to stop the main loss from backpropagating
through the intermediate prediction used as feedback. The feedback uses
`softmax(auxiliary_logits).detach()`; the auxiliary cross-entropy still uses the
attached logits and trains the extractor and shared output head. The default is
`false`, which lets both losses train the intermediate prediction.

This detaches the predicted probabilities, not the entire feedback branch: the
embedding table and feedback gate still receive main-loss gradients, and the
backbone and shared output head still train through their normal final-output
paths. Changing this flag does not change forward values or inference behavior.
