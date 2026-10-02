# Component contracts

Contracts currently use explicit constructor arguments, conventions, and runtime
validation. The attempted strict-typing migration was reverted. Tensor annotations
or dataclasses in particular modules do not imply a project-wide static guarantee
that arbitrary Hydra selections are compatible.

## Common tensor meanings

Use these symbols when reading the component references:

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size on the current device. |
| `L` | Sequence length, including the task's special/separator positions. |
| `V` | Relevant token vocabulary; distinguish data and model vocabularies. |
| `D` | Continuous representation dimension or model hidden width, as specified. |
| Tokens `[B, L]` | Integer token IDs, not embeddings. |
| Token distribution `[B, L, V]` | Probabilities or log probabilities; the interface must specify which. |
| Continuous state `[B, L, D]` | A noisy continuous representation, not necessarily normalized. |
| Global time `[B]` | Time sampled for training examples; schedules convert it to coefficients. |

Dimensions alone do not establish compatibility. A plug-in estimate and a clean-token
posterior can both have shape `[B, L, V]` but enter different reverse equations.
A one-hot rank calibration cannot be substituted for an embedding calibration
merely because both return Gaussian coefficients.

## Masks and task boundaries

- `loss_mask`: positions contributing to the training reduction; some methods may
  apply additional weighting internally.
- `corruption_mask`: positions eligible for corruption. A prompt can be excluded
  from loss and still be corruptible, depending on the task recipe.
- `prompt_mask`: `True` means the supplied token is fixed during generation.
- `padding_mask`: task collation may provide it, but the shared training wrapper
  does not forward it to `algorithm.loss`. Padding support must be expressed through
  the actual loss/corruption semantics or an explicit integration change.

The shared training wrapper forwards `input_ids`, `loss_mask`, and
`corruption_mask`; it does not forward extra task batch fields such as
`prompt_mask`. Generation receives its prompt mask through a separate interface.

Task `prepare_generation` must not pass held-out answer tokens as fixed input.
The method/process must preserve fixed prompts. Vocabulary config specifies whether
an absorbing mask is appended, already native to the tokenizer, or absent.

## Prediction meaning

| Family | Meaning and consumer |
| --- | --- |
| Posterior | Clean-token distribution conditioned on noisy state(s); used by posterior-based reverse transitions and posterior means. |
| Plug-in | Normalized estimate used in the uniform/DUO plug-in equations; not an unconstrained clean-token posterior. |
| Score | Score-ratio parameterization for SEDD; not automatically usable as posterior probabilities. |
| Gaussian-prior correction | Combines neural logits with a state-dependent representation prior before forming a posterior. |

Training and sampling prediction pathways are separate so temperature, top-k,
top-p, and commitment policy do not silently alter the training objective. Model
softcapping can intentionally belong to both pathways. Explicit precision
promotion around entropy or transition math does not redefine the prediction.

Expected-embedding self-conditioning requires a probability interpretation.
Hidden-state self-conditioning requires compatible hidden-state output and input
support. Neither should reinterpret arbitrary scores as probabilities.

## Loss and sampler results

Most training algorithms expose `loss(tokens, loss_mask, corruption_mask=..., ...)`
and return a mapping with differentiable `loss`, detached `loss_sum`, and
`num_tokens`. The training wrapper logs validation reduction under `val/nll`, even
for objectives that are **not** likelihoods. Interpret that name as a historical
loss label unless the method supplies a likelihood interpretation.

Samplers return token sequences and metadata including `nfe_per_sample`; some also
provide `nfe`. NFE counts model evaluations according to that sampler's accounting,
not simply the number of requested steps. The common evaluator stores mean/min/max
NFE separately from task metrics.

## Compatibility boundaries

| Combination | Requirement |
| --- | --- |
| Gaussian schedule + representation | Coefficient/geometry compatibility, including rank-calibration assumptions. |
| Sampler + process + predictor | Matching reverse-transition meaning and prediction interface. |
| Hybrid branches | Compatible clean-token posterior and shared vocabulary/layout; each branch retains its own state. |
| Expected-embedding self-conditioning | A supported probability-producing predictor and embedding source. |
| Block diffusion | Compatible model attention/layout and method-specific sequence handling. |
| FSDP + algorithm | No unsupported EMA, rank-zero generation, norm clipping, or post-update parameter projection hooks. |

Not all combinations are tested, and runtime checks do not cover every possible
misconfiguration. Start from a named recipe, consult the relevant component page,
and add a meaningful integration test when changing a boundary. See
[coverage](../validation/coverage.md) for the actual verification level.
