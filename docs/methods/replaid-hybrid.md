# Conditional REPLAiD hybrid

The `replaid_hybrid` preset combines native reusable components; it does not
import historical code. Its trained token table lives in a 16-dimensional
unit-normalized space. Gaussian noise applies only to unrevealed tokens;
revealed tokens also enter a separate clean-context projection. Posterior
self-conditioning is disabled. The learned shared clock determines both
Gaussian log noise-to-signal and absorbing-mask probability. Its time-conditioning
floor is `1e-30`, preserving the large log-noise values learned by the original
hybrid checkpoint rather than clipping them at the general model default.

Components:
- `PiecewiseInverseCDFClock`, `ClockGaussianSchedule`, `ClockKeepSchedule`;
- `ForcedTargetMask` conditional proposal;
- `HybridELBO` reveal/Gaussian/prior objective;
- `RevealedContextInput` architecture slot;
- Gaussian-prior posterior correction on the Gaussian view;
- composed `revealed_hybrid` sampler, which decodes the final token state.

Training uses a second-moment score-function estimator for density logits;
endpoints and embeddings receive ELBO gradients. Self-conditioning and
block-diffusion combinations are not implied by this recipe. Optimizer groups
apply the special learning rate to the codebook and clock without decay.
Native checkpoints include model, representation, shared clock and EMA.

For an exported evaluation bundle:

```bash
uv run --locked python main.py --config-path /absolute/path/to/bundle --config-name config
```

`config.yaml` selects EMA; `config-raw.yaml` selects raw weights. Set
`evaluation.device=cuda` only when GPUs have been allocated. Sampling/evaluation
uses the same task/tokenizer specified by the saved config. Data paths are local
and must be available on the target machine. Exported historical bundles are
for evaluation or an explicit warm start; they do not claim to restore the old
Lightning optimizer/loop/RNG state for exact training resumption.

Validation covers independent estimator/gradient checks, two CPU Lightning
updates, shared-state restoration, prompt preservation and actual model-call
NFE counts. Full-checkpoint port validation is recorded beside the exported
artifacts, rather than depending on historical code at runtime.
