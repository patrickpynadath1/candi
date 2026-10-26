# von Mises–Fisher spherical flows

This implements the alternative conditional path in *Spherical Flows for Sampling
Categorical Data*, Chemseddine, Kornhardt and Steidl, arXiv:2605.05629v4,
Sections 3–4 and Appendix C. It does not replace the geodesic `spherical` view.

## Information path

Select `hybrid_view@model.views.content=vmf` in a composed recipe. The view uses
learnable unit-normalized token embeddings and

```
q(x | token, t) ∝ exp(kappa(t) * dot(embedding(token), x))
kappa(t) = kappa_max * (1 - t)
```

Our time runs from data (0) to noise (1), opposite to the paper. At time 1 the
prior is uniform on the sphere. Time 0 has finite concentration, not a point mass;
terminal token decoding completes generation. The default is dimension 512 and
kappa_max 250; the paper uses 15 for Sudoku and 250 for its language tasks.
When changing the concentration, set **both**
`model.views.content.process.kappa_max` and
`model.views.content.schedule.kappa_max` (the latter cannot exceed the table).
When changing dimension, also update the representation's state and embedding dimensions.

The forward kernel samples the radial cosine by interpolated inverse CDF and a
uniform tangent direction. Deterministic CPU quadrature tables are cached and
copied to the device on first use. Default grids contain 4097 radial/quantile
points and 257 concentration points. Sampling and velocities are numerical
approximations; increase these resolutions for more demanding accuracy checks.
Dimensions below 3 are currently unsupported.

## Denoising task

The existing token cross-entropy objective learns the posterior. Standard hidden
or expected-embedding self-conditioning and parallel discrete views compose with
this path. This is a path/sampler implementation in our backbone, not a complete
reproduction of the paper's tied spherical output classifier or training recipe.

## Inference

The paper's Algorithm 2 predictor uses the full token posterior, all positions,
no commitments, and a per-view `VMFProbabilityFlowUpdate`. For each token its
contribution is `psi(cosine, kappa) * tangent_projection(embedding)`; the posterior
weighted sum is multiplied by the source schedule derivative and time step.
The step uses normalization retraction, as in the paper, rather than an
exponential-map step. The scalar solves Eq. (14); the implementation evaluates
the flux integral in log space from either boundary to avoid tail cancellation.

The existing `hybrid` sampler therefore runs the ODE predictor directly. To add
the paper's fresh-prediction Langevin correctors, select
`sampler=vmf_predictor_corrector`. This is the same `ComposedSampler` loop with
an optional `SphericalLangevinCorrector`, not a separate execution engine.
The corrector projects `epsilon * score + sqrt(2*epsilon) * normal_noise` into
the tangent space and normalizes the result. Its step size is configurable;
the paper's additional task-specific damping/tuning heuristics are not presets here.

Every correction makes a new posterior prediction at the updated state and time.
Prompts stay clamped. NFE is `predictor_steps * (1 + corrector_steps) + 1`,
including terminal decoding: the example's 63 steps plus one corrector give
127 NFE, **not** 63. The tuner asks the sampler for `steps_for_nfe(budget)`
after applying trial overrides, including a tuned corrector count. It allocates
complete predictor/corrector steps without exceeding the budget; a 128-call
budget with one corrector uses 127 calls. Budgets too small for one complete
step plus decoding fail before evaluation.

Correction requires `commitment=none` and at least one spherical view exposing
a score. In a vMF + discrete hybrid, the spherical view is corrected while the
discrete state stays fixed during correction. Each fresh posterior still sees
all views. This hybrid extension is a framework option, not a method evaluated
in the paper. Views declare spherical geometry through `state_geometry="sphere"`
and provide `score`; correction does not depend on a concrete process class.

Other existing choices remain available: sampled/greedy clean-token forward
resampling, or hard-token/position/commitment policies. They are different
samplers, not the paper's posterior-averaged ODE. In particular, greedy clean
estimates replace the full posterior by a one-hot estimate.

## Validation

`core/tests/test_vmf.py` checks forward sample moments against the Bessel-ratio
formula, unit norms, the velocity differential equation and uniform-limit
solution, schedule direction, differentiable training, prompt preservation,
corrector NFE, and self-conditioned hybrid forward resampling. The existing
seeded sampler snapshots check that adding the optional correction hook leaves
previous sampler behavior unchanged.
