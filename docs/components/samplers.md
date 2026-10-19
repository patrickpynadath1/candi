# Samplers

Recipe context: [inference settings](../concepts/inference-settings.md).

Samplers orchestrate generation: initialization, time grids, model requests,
process updates, stopping, decoding, prompt preservation, and trajectory state.
They consume parameterized predictions, not raw logits. Temperature and filtering
remain predictor settings.

The common external shape is
`sample(algo, prompt_tokens, prompt_mask) -> (tokens, metadata)`. Metadata includes
`nfe` and, where needed, `nfe_per_sample`; evaluation uses these to measure cost.
A prompt mask fixes tokens regardless of whether they appear in a training loss.

## Configured families

| `sampler` | Intended process / policy |
| --- | --- |
| `ancestral` | Absorbing posterior reverse steps |
| `entropy_bounded` | Absorbing entropy-bounded commitments |
| `analytic` | SEDD absorbing-score analytic transition |
| `uniform_ancestral`, `uniform_entropy_bounded` | Uniform plug-in or posterior transition with matching predictor/process |
| `euler`, `heun`, `ode` | Gaussian posterior-derived velocity integration |
| `gaussian_ddpm` | Stochastic Gaussian reverse posterior steps |
| `spherical` | Posterior-derived spherical tangent flow |
| `simplex` | Dirichlet bridge sampling, optional churn/guidance |
| `blur_ancestral` | Spectral blurring reverse updates |
| `hybrid`, `hybrid_entropy_bounded` | Shared posterior feeding multiple branch updates |
| `autoregressive`, `permuted_autoregressive` | Incremental causal decoding with KV cache |

See [configs](../../configs/sampler/) and [implementations](../../core/samplers/).
Choose a matching experiment first: an arbitrary sampler switch is not a proof of
mathematical compatibility.

## Continuous integration

[PosteriorMeanVelocity](../../core/samplers/continuous.py) requests a clean-state
mean through the algorithm and delegates the field formula to the Gaussian
process. The current adapters require `PosteriorParameterization`; directly
predicted velocity is not supported. Representation decoding and posterior argmax
are explicit final policies, so integrators do not branch on one-hot vs embedding.

Euler uses one field evaluation per step; Heun uses two. Both add a final model
call for decoding. The default interval stops short of singular endpoints:
`start_time=0.999`, `epsilon=1e-5`. Adaptive Dormand–Prince (`ode`) exposes `rtol`,
`atol`, `max_nfe`, and `max_steps`; exhausted budgets or nonfinite fields fail
explicitly. Its NFE budget includes final decoding. It currently rejects enabled
self-conditioning because rejected-step and derivative-cache semantics need a
separate integration.

Gaussian DDPM is a stochastic reverse-moment solver, not another name for Euler.
Its prior and prompt modes and previous-feedback versus bootstrap policy are
explicit settings. Spherical sampling averages token-directed tangent vectors;
it does not substitute the Gaussian posterior-mean velocity formula.

## Entropy, hybrid updates, and feedback

EB computes entropy of the uncommitted prediction, chooses positions, then asks
for the predictor's configured commitment probabilities. Entropy is not computed
after a greedy point-mass projection. Uniform EB must use the selected uniform
reverse contract, including posterior bridge normalization when applicable.

Hybrid sampling tracks a state for every branch and queries one fused model.
Each branch receives the shared posterior and its own schedule values. Branch
clocks map from global time; noisy states need not map into each other. The
CANDI-specific EB recipe is not a generic guarantee for arbitrary branch sets.

Supported samplers create a local self-conditioning prediction session. Heun
freezes incoming feedback across its stages, committing after a successful step.
Trajectory state never lives in a shared sampler instance. Block diffusion wraps
sampling left-to-right and starts a fresh session per block; see
[modifiers](../modifiers/block-diffusion.md).

When adding a sampler, keep reverse mathematics in the process and orchestration
here. Test prompt preservation, NFE including final calls, local state cleanup on
failure, deterministic limit cases, and predictor compatibility. Config knobs
that shape logits belong in the predictor, not duplicated in the sampler.

## Composing loops and update rules

[FixedGridLoop](../../core/samplers/loop.py) runs an update callable over a supplied
time grid. It does not inspect predictions or states, so the same loop executes
tensor states and tuples of hybrid branch states. Fixed ancestral, analytic,
Gaussian DDPM, blurring, spherical, simplex, and hybrid samplers use this loop.
Their initialization and terminal policies remain explicit: spherical and simplex
use their last iteration for decoding, whereas Gaussian and ancestral recipes
can evaluate an additional terminal prediction. No extra model evaluation is
introduced by sharing the loop.

[DiscreteSampler](../../core/samplers/discrete.py) provides a composable discrete
lifecycle. Its rule supplies `initialize`, `update`, and `finalize`; the latter two
return `(state, nfe)`, so a custom rule can report multiple model evaluations.
`AbsorbingAncestralUpdate`, `UniformAncestralUpdate`, and `AbsorbingAnalyticUpdate`
request predictions through the algorithm and delegate transition equations to
the appropriate process. Existing Hydra sampler targets are presets for these
rules, and retain their previous constructor defaults.

```python
from core.samplers.discrete import DiscreteSampler
from core.samplers.ancestral import AbsorbingAncestralUpdate
from core.samplers.loop import LinearTimeGrid

sampler = DiscreteSampler(
    update_rule=AbsorbingAncestralUpdate(final_denoise="greedy"),
    time_grid=LinearTimeGrid(start=1.0, end=1e-5, steps=180),
)
# tokens, metadata = sampler.sample(algo, prompt_tokens, prompt_mask)
```

Continuous `FixedStepSampler` likewise accepts an `update_rule` (`EulerUpdate` or
`HeunUpdate`) and optional `time_grid`. The rule may evaluate the field multiple
times; the field counts actual calls. A prediction-session context freezes feedback
across all stages of an accepted step. Custom grids must retain the configured
`start_time` and `epsilon` endpoints, which also control initialization and decoding.

The shared loop is not a universal stopping policy. Entropy-bound completion,
adaptive ODE acceptance/rejection, and autoregressive cache traversal retain
specialized loops. Their APIs and behavior are unchanged. Likewise, specialized
DDPM/blur/hybrid updates currently use local update callables rather than the
discrete rule protocol; sharing execution does not assert mathematical
interchangeability between those processes.
