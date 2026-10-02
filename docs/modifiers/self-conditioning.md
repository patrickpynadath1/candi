# Self-conditioning

Self-conditioning adds detached prediction feedback to the network input. Its
implementation lives in [core/modifiers/self_conditioning](../../core/modifiers/self_conditioning/),
with architectural adapters in [models/prediction.py](../../core/models/prediction.py).
Algorithms prepare ordinary inputs and prediction conversions; the shared
coordinator handles bootstrapping. MDLM has no self-conditioning-specific loss.

```bash
uv run python main.py self_conditioning.enabled=true self_conditioning.source=hidden
```

## Sources and injection

| `self_conditioning.source` | Feedback | Injection |
| --- | --- | --- |
| `posterior_mean` (default) | Expected clean input embedding from unshaped posterior probabilities | Bias-free projection into model width, zero initialized by default |
| `hidden` | Final contextual hidden state before the output head | FP32 affine LayerNorm, then direct addition; scale and bias start at zero |

Hidden feedback has no dense projection. Both default adapters contribute exactly
zero initially. Missing hidden feedback is represented by zeros through the same
LayerNorm, so its learned bias may contribute later. Expected-embedding feedback
retains the representation's geometry; one-hot means are probability vectors,
and spherical mean embeddings are **not** renormalized to unit length.

Feedback probabilities include mathematical constraints and any softcap, but
exclude temperature, top-k/top-p, and greedy commitment. The algorithm supplies
the expected-embedding conversion explicitly. The coordinator never discovers
an embedding table by reaching into another component. Both sources are detached.
RePlaid can override conditioning initialization and selection in its recipe;
inspect its experiment config rather than assuming generic defaults reproduce it.

## Training and validation

`self_conditioning.probability=0.9` selects training examples independently for a
no-gradient bootstrap pass on the **same corrupted state and time**. That pass has
no incoming feedback. Selected examples use its detached prediction in the main
pass; others use zero feedback. The bootstrap does not compute an objective or
update adaptive observations.

`evaluation_probability=1.0` separately controls validation-loss bootstrapping.
It does not control trajectory reuse during generation. The coordinator also
supports `selection=fixed_count`, selecting `ceil(probability*batch_size)` examples,
for recipes such as RePlaid. Its specialized embedding/schedule gradient routes
are separate from this general architectural mechanism.

Adapters are built before optimizer and EMA creation and included in checkpoints.
Disabled models retain the original state-dict keys and initialization behavior.
Older enabled checkpoints with a different feedback architecture require matching
historical code; replacing the old projection is not a transparent load migration.

## Generation

Supported samplers create a local
[PredictionSession](../../core/modifiers/self_conditioning/session.py). The first
prediction has no feedback. Later calls reuse detached features from the preceding
prediction without an extra network evaluation. Heun freezes feedback across both
stages and commits it only after the step succeeds; failed staged work is discarded.
The model and reusable sampler retain no trajectory state.

Calling raw `model_logits` or calling a prediction endpoint without a session stays
stateless. Block sampling starts a new session for each block. A hybrid model has
one conditioning vector for its fused backbone, not a separate vector per branch.
Gaussian DDPM additionally supports a bootstrap-at-current-state policy with its
own extra model-call cost.

## Compatibility and extension

Integrated posterior paths include MDLM, Gaussian fixed-step sampling, spherical,
simplex, posterior-uniform, blurring, RePlaid, and hybrids, subject to their own
sampler/model constraints. This is not a claim that every combination has a quality
benchmark. Adaptive ODE sampling explicitly rejects enabled feedback until rejected
steps and cached derivatives have consistent lifecycle semantics.

The plug-in/score discrete base currently rejects enabled conditioning; causal
methods need position-alignment and cache integration. Posterior-mean conditioning
requires `training_posterior` and `probabilities`, and construction reports a
missing-capability error rather than treating scores as a posterior. Hidden
conditioning does not mathematically require probabilities, but still needs an
integrated model/prediction/sampler path.

To extend support, provide optional model features and a conditioning adapter,
route predictions through the coordinator, and give the sampler a local session.
Test detached bootstrap gradients, unchanged corruption/time, unshaped feedback,
failed-step cleanup, and checkpoint/EMA behavior. See
[self-conditioning tests](../../core/tests/test_self_conditioning.py) and
[reproduction results](../validation/reproduction-results.md) for evidence; old
unnormalized and L2-normalized experiments describe previous architectures.
