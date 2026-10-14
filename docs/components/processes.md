# Processes

Recipe context: [information path](../concepts/information-path.md).

Processes own forward corruption, priors, and reverse-transition or vector-field
mathematics. They receive prepared states, coefficients, predictions, and masks;
they do not call the model, schedule, or objective. Samplers decide when and how
to invoke these operations.

Discrete states have shape `[B,L]`; continuous states normally have `[B,L,D]`.
Prompt/corruption masks protect positions explicitly. A time derivative must be
with respect to the same coordinate used by its schedule and sampler.

## Available processes

| Config | State / reverse contract |
| --- | --- |
| `absorbing` | Tokens plus mask; clean-token posterior for reveal transitions |
| `uniform` | Tokens replaced uniformly; normalized plug-in estimate |
| `uniform_posterior` | Same forward corruption, posterior mixture of exact bridges |
| `gaussian` | Affine continuous state; clean-state mean for velocity or reverse moments |
| `spherical` | Unit-sphere corruption and log/exp-map geometry |
| `dirichlet` | Simplex-valued state with Dirichlet bridge mathematics |
| `blurring` | Spectral sequence/feature corruption respecting sequence boundaries |

See [implementations](../../core/processes/) and [configs](../../configs/process/).
Hybrid presets additionally compose `CANDIMaskingProcess` and
`CANDIGaussianProcess`; they retain CANDI-specific transition conventions while
standard masking replaces the old accidental token-retention behavior.

## Discrete examples

For absorbing corruption, an eligible clean token survives with probability
`alpha_t`, otherwise becoming the mask. Reverse generation from `t` to cleaner
`s` reveals a masked token with probability
`(alpha_s-alpha_t)/(1-alpha_t)` and draws it from the supplied clean posterior.
Visible tokens stay visible. The process handles commitment; EB decides positions.

Uniform replacement has marginal
`q(x_t=j | x_0=i) = alpha_t * 1[i=j] + (1-alpha_t)/V`.
[UniformProcess](../../core/processes/uniform.py) consumes plug-in estimates.
[PosteriorUniformProcess](../../core/processes/uniform_posterior.py) instead mixes
`q(x_s | x_t, x_0)` over an inferred clean-token posterior. Its likelihood
normalization depends on the observed token. Replacing one predictor with the
other while keeping the same reverse formula is incorrect. The posterior
implementation handles equal keep rates as an identity step.

## Gaussian example

For `z_t = alpha_t * e(x_0) + sigma_t * epsilon`,
[GaussianProcess](../../core/processes/gaussian.py) evaluates the posterior-derived
velocity

```text
v(z_t,t) = alpha'_t * mean + (sigma'_t / sigma_t) * (z_t-alpha_t*mean)
```

The sampler obtains `mean` from posterior probabilities through the representation
and integrates this field toward smaller times. Positive sigma is required; the
solver stops before its singular clean endpoint and performs explicit decoding.
`reverse_posterior` supplies stochastic Gaussian reverse steps for monotone-SNR
paths. These operations are independent of whether `e` is one-hot or embedding.

## Sequence-aware and new processes

Blurring cannot mix spectral information across independent blocks. Its process
uses [SequenceLayout](../../core/sequences.py) to treat each block as a separate
sequence; the block modifier supplies that layout. This keeps block-specific
policy out of the algorithm's corruption mathematics.

A new process should document its state space, endpoints, coefficient ordering,
and required prediction meaning. Provide deterministic probability/moment helpers
where useful for testing, then execution helpers for random draws. Test forward
marginals, reverse normalization, identity steps, endpoint guards, and prompt
preservation independently of model quality.
