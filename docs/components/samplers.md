# Samplers

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
