# Models

Recipe context: [denoising task](../concepts/denoising-task.md).

Models own neural architecture and learned parameters. They consume prepared
states and conditioning coordinates and return raw vocabulary logits. Posterior
constraints, softcapping, temperature and token filtering belong to
[predictors](predictors.md).

Every diffusion recipe uses one network, [`ModularDiT`](../../core/models/modular.py):
a DiT transformer whose input, time conditioning, self-conditioning and output are
**slots** chosen in YAML. The autoregressive recipes use
[`CausalDiT`](../../core/models/causal.py), which keeps its own KV-cache path. Both
share the transformer execution in [`TextDiT`](../../core/models/text_dit.py) and the
layers in [`layers.py`](../../core/models/layers.py).

## Presets

| `model=` | Input slot | Output slot | Used by |
| --- | --- | --- | --- |
| `dit` | token ids | AdaLN head | masked, uniform and other discrete recipes |
| `dit_conditioned` | token ids | AdaLN head, time conditioning on | discrete recipes that condition on time |
| `continuous_dit` | embedded projection of a continuous state | embedding head | Gaussian, blurring, RePlaid |
| `simplex_dit` | simplex log-state (expectation or argmax) | AdaLN head | simplex diffusion |
| `sphere_dit` | linear projection | AdaLN head, optional temperature | spherical flows, vMF |
| `single_view`, `hybrid`, `hybrid_candi`, `hybrid_uniform_simplex` | one or several corrupted views, fused | AdaLN head | view-based (hybrid) recipes |
| `causal_dit` | token ids, causal | AdaLN head | autoregressive, permuted AR |

The presets are in [`configs/denoising_task/model/`](../../configs/denoising_task/model/). The slot groups are
`configs/denoising_task/model/input/` (`token`, `simplex`, `linear`, `embedded_projection`, `revealed_context`) and
`configs/denoising_task/model/output/` (`adaln`, `embedding`); a recipe swaps one with
`override /denoising_task/model/input@model.input: linear`, for example.

## Size

`model.hidden_size`, `n_blocks`, `n_heads`, `cond_dim` and `dropout` size the network
(defaults 512, 8, 8, 128, 0.1). `model.state_dim` is the width of a continuous state;
`model.output_vocab_size` slices the logits when the model vocabulary includes extra
tokens. The task vocabulary and the absorbing mask vocabulary are distinct and come
from configuration (`token_space`). A checkpoint must match the parameter shapes.

## Slots

```yaml
_target_: core.models.modular.ModularDiT
vocab_size: ${token_space.model_vocab_size}
hidden_size: 512
n_blocks: 8
input:                       # configs/denoising_task/model/input/token.yaml
  _target_: core.models.modular.TokenInput
  _partial_: true
time:
  _target_: core.models.modular.TimeConditioning
  _partial_: true
  coordinate: time
  adaLN: true
output:                      # configs/denoising_task/model/output/adaln.yaml
  _target_: core.models.modular.AdaLNHead
  _partial_: true
self_conditioning:
  _target_: core.models.modular.SelfConditioning
  _partial_: true
  enabled: ${self_conditioning.enabled}
  source: ${self_conditioning.source}
```

Slots are Hydra partials. The network fills in the dimensions each slot accepts
(`vocab_size`, `hidden_size`, `cond_dim`, `state_dim`) and owns their weights.

| Slot | Classes | Options |
| --- | --- | --- |
| `input` | `TokenInput`, `SimplexInput`, `LinearInput`, `EmbeddedProjectionInput`, `RevealedContextInput`, or `views` (see below) | `SimplexInput.mode`: `expectation`, `argmax`; `preconditioning` on the continuous inputs |
| `time` | `TimeConditioning` | `coordinate`, `adaLN`, `epsilon`, `input_projection`, `time_source` (`global` or the first view's `point` time) |
| `self_conditioning` | `SelfConditioning` | `source` (`posterior_mean`, `hidden`), `space` (`hidden` or `state`), `initialization`, `scale` (a number or `unit_embedding`) |
| `output` | `AdaLNHead`, `EmbeddingHead` | `adaLN`, `temperature` (AdaLN head); `output_dim` (embedding head) |

`model.time.adaLN` turns time conditioning on in the time embedding, every block
(`block_adaLN`) and the output head together.

### Time coordinates

`model.time.coordinate` chooses the scalar that enters the timestep embedder:

| Value | Coordinate |
| --- | --- |
| `time` | Supplied time |
| `log_one_minus_time` | `-log(1-time)` (continuous default) |
| `negative_log_keep`, `negative_log_keep_floor`, `negative_log_keep_exact` | `-log` of a discrete view's keep rate |
| `log_sigma` | `log(sigma)` |
| `log_effective_noise` | `log(sigma)-log(alpha)` |
| `log_snr` | `2*(log(alpha)-log(sigma))` |
| `negative_log_snr` | Negative of the preceding coordinate |

Log coordinates clamp coefficients at `epsilon`. Away from the clamps, Gaussian
preconditioning combined with `log_effective_noise` or `log_snr` gives the same
model input under a common positive rescaling of state, alpha and sigma;
`log_sigma` changes under that rescaling. These choices do not change process
coefficients or their derivatives.

### Continuous inputs

`EmbeddedProjectionInput` multiplies the state by a normalized learned embedding,
then projects it; `LinearInput` projects the state directly. Both take
`preconditioning`:

- `gaussian`: scale by `1 / sqrt(alpha**2 + sigma**2)`;
- `unit_embedding`: scale by `1 / sqrt(alpha**2 / D + sigma**2)` for state dimension
  `D` (RePlaid; linear input only);
- `none`.

RePlaid combines `override /denoising_task/model/input@model.input: linear`, the AdaLN head,
`model.time.input_projection=true` and `unit_embedding` preconditioning, with
self-conditioning `scale: unit_embedding` (`sqrt(D)`). Changing a slot changes the
checkpoint's keys and shapes; changing a time coordinate does not.

### Views (hybrid recipes)

`model.views` lists the corrupted views, each a `DiffusionView` with its own process,
schedule and representation, and `model.fusion` how their embeddings combine
([hybrids](../methods/hybrids.md), [`fusion.py`](../../core/modifiers/hybrid/fusion.py)).
The views replace the input slot (`ViewsInput`); one shared transformer and one
self-conditioning vector serve all views. A view held by the algorithm
(`algorithm.views=`) is bound to the network at construction with
`ModularDiT.bind_views`.

## Block sublayers and depth mixing

Two groups change the transformer blocks without changing the slots:

- `+block_sublayer=` replaces the position-wise MLP in every block
  ([`sublayers.py`](../../core/models/sublayers.py)): `mlp` (default 4x GELU),
  `expanded_attention` (a second attention sublayer with `expansion` times the width),
  or `centered_mlp` (subtracts a learned, gated share of the sequence mean before the
  GELU; the gate starts at zero, which is the plain MLP; full unmasked sequences only).
- `+depth_mixing=` changes how blocks read earlier blocks' outputs
  ([`core/models/depth_mixing/`](../../core/models/depth_mixing/)): `baseline` (residual
  stream), `dca` (DeepCrossAttention), `mudd` (MUDD connections) or `attention`
  (content-based per-position attention over depth). The non-baseline options shrink
  `mlp_hidden_size` to keep the parameter count close to the baseline.

Both change checkpoint keys.

## Caches and modifiers

`CausalDiT` owns KV-cache mechanics; the autoregressive samplers own append order
and lifetime, and permuted generation passes query and key positions separately.
[Block diffusion](../modifiers/block-diffusion.md) has its own clean-prefix cache and
explicit clean/noisy streams; noisy keys are never reused as clean context.
[Self-conditioning](../modifiers/self-conditioning.md) injects one detached feedback
tensor through the self-conditioning slot. Models return
[`ModelPrediction`](../../core/models/prediction.py) (logits and the final hidden
state) when a modifier asks for features, so feedback needs no extra pass.

`token_feedback.detach=true` stops the main loss from backpropagating through the
intermediate prediction used as feedback (`softmax(auxiliary_logits).detach()`); the
auxiliary cross-entropy still trains the extractor and shared head. It does not
change forward values or inference.

## Adding a model

Most new architectures are a new slot: a module that accepts the dimensions it needs
and maps a state to hidden vectors (input) or hidden vectors to logits (output), plus
a YAML file in the slot group. Return the expected vocabulary dimension, implement
hidden-feature and stream interfaces only when their meaning is defined, and test
cached against uncached execution where a cache exists. A new architecture alone does
not make it compatible with every modifier; see the
[compatibility table](../validation/compatibility.md).

### Revealed clean context

`core.models.revealed_context.RevealedContextInput` combines a continuous input
slot with a projection of clean embeddings at positions revealed by an absorbing
view. Unknown positions contribute zero context. Both use the continuous view's
codebook, so there is no second learned token table. The extra parameter cost is
`state_dim * hidden_size`; `context_scale=unit_embedding` multiplies context by
`sqrt(state_dim)`. This is independent of posterior self-conditioning.
Select it with `denoising_task/model/input@model.input: revealed_context` and put the
two views under `model.input.views` (as `replaid_hybrid` does); the continuous part is
the nested `continuous` slot (a linear input by default).
