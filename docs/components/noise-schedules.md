# Noise schedules

Schedules map global time to corruption coefficients; they do not draw training
times or corrupt states. In the usual convention, zero is clean and increasing
time means more noise. Sampling proceeds toward smaller times. Endpoint choices
are method-specific and are not automatically exact clean/pure-noise marginals.

## Output contracts

Keep-rate schedules return `(alpha, dalpha_dt)` with shape `[B]` and advertise
`output_kind="keep_rate"`. The default `log_linear` gives
`alpha=1-(1-epsilon)*t`, `dalpha_dt=-(1-epsilon)`. A nonzero epsilon leaves terminal
clean mass; an all-mask prior is then a limiting-prior approximation.

Gaussian schedules return [GaussianPath](../../core/noise_schedules/gaussian_path.py):
`alpha`, `sigma`, `dalpha_dt`, `dsigma_dt` for
`z_t=alpha_t*clean+sigma_t*noise`. Derivatives are essential for general continuous
sampling: sigma alone only describes special paths. Optional `training_time`
implements method-specific time adjustment; callers must apply it once.

## Fixed Gaussian choices

| `noise_schedule` | Calibration and options |
| --- | --- |
| `one_hot_rank`, `gaussian_rank` | Same OneHotRankSchedule class; normalized expected competing-coordinate rank |
| `embedding_sphere_rank` | Approximate rank for iid unit-sphere embedding geometry |
| `argmax_error` | Probability that Gaussian-corrupted one-hot argmax is incorrect |
| `exponential_ve` | `alpha=1`, `sigma=sigma_min*(sigma_max/sigma_min)**t` |
| `linear_interpolation` | `alpha=1-t`, `sigma=t` |
| `rescaled_gaussian` | Common time-dependent rescaling of a nested Gaussian base |

[OneHotRankSchedule](../../core/noise_schedules/gaussian_rank.py) supports
`inversion=lookup` or `analytic`, `coupling=linear` or `geometric`, and
`normalization=vocabulary` or `competitors`. The expected fraction of competing
coordinates exceeding the clean coordinate is
`Phi(-1/(sqrt(2)*sigma))`; vocabulary normalization multiplies by `(V-1)/V`.
The lookup interpolates the computed rank table; analytic inversion uses the
normal inverse CDF. Both honor `sigma_min=0.2`, `sigma_max=4.0` defaults and clamp
outside their calibrated interval, with zero slope in saturated regions.
This rank is not the probability of **any** competitor winning.

[OneHotArgmaxErrorSchedule](../../core/noise_schedules/argmax_error.py) calibrates
that different event using quadrature and inversion. It supports linear or
geometric error progression. [EmbeddingSphereRankSchedule](../../core/noise_schedules/embedding_rank.py)
also supports linear or geometric rank progression, but its sphere approximation
is a separate schedule, not the exact one-hot formula applied to arbitrary tables.
Geometry requirements are checked against the [representation](representations.md).

## Rescaling and SNR

[RescaledGaussianSchedule](../../core/noise_schedules/rescaled_gaussian.py) divides
both coefficients by a positive denominator:

| `path_type` | Denominator | Coefficient convention |
| --- | --- | --- |
| `ve` | `alpha` | Signal coefficient one |
| `vp` | `hypot(alpha,sigma)` | Squared coefficients sum to one |
| `linear` | `alpha+sigma` | Coefficients sum to one |

Effective noise `sigma/alpha` and coefficient SNR `alpha**2/sigma**2` are preserved.
Both derivatives include the quotient rule. VP is a coefficient normalization,
not a statement that arbitrary embeddings have unit data variance. Equal SNR
preserves corruption marginals up to scale, not identical neural inputs, training,
or finite-step trajectories. VE is undefined at `alpha=0` and raises rather than
silently substituting epsilon. Linear normalization need not be linear in time.

```bash
uv run python main.py +experiment=gaussian_diffusion_sudoku \
  noise_schedule=rescaled_gaussian \
  noise_schedule@noise_schedule.base=exponential_ve \
  noise_schedule.path_type=vp
```

## Fitted adaptive clocks

`sflm_adaptive` and `cdcd_adaptive` wrap keep-rate paths. `sflm_gaussian_adaptive` and
`cdcd_gaussian_adaptive` wrap Gaussian paths. Their
[shared adaptive implementation](../../core/noise_schedules/adaptive/) owns ring
buffers, finite-observation filtering, distributed gathering, warmup, refit cadence,
EMA, and checkpoint state. Algorithms submit detached masked observations during
training only. `loss_signal=cross_entropy` requires normalized predictions;
`objective` uses the optimization loss.

- S-FLM defaults to `coordinate=legacy_time`: spline/ridge expected-loss fit,
  nonnegative slope, uniform coverage, inverse-CDF PCHIP, and bias-corrected EMA.
  `physical` explicitly fits normalized corruption instead.
- CDCD defaults to physical coordinates and CE. It uses inverse-density-weighted
  isotonic regression and a fixed bin grid (100 bins by default). This is the
  repository's CDCD-style approximation, not trainable original-CDCD bin logits.

Defaults include buffer size 12800, warmup 1000, refit every 50, grid size 1000,
and EMA 0.9. Legacy-time CDF extrapolation can cross an endpoint: public evaluation
bounds keep rates and zeros clipped derivatives; positive times stay below keep
one to avoid singular losses. Raw fitted curves remain available for comparison.
Gaussian wrappers evaluate the base at adapted time and chain-rule both derivatives.
They preserve geometry requirements. Rescaling can wrap an adaptive base or sit
inside one; delegated observations must still be applied once.

## Other schedule families

`spherical` and `spherical_adaptive` provide spherical progress; `simplex`
provides Dirichlet parameters, while its optional adaptive **time sampler** is a
separate component. `blurring` supplies spectral decay coefficients.
`learned_log_snr` and `plaid_log_snr` are differentiable learned Gaussian schedules
used by RePlaid-related constructions, with trainable monotone shape and endpoint
parameters. They differ from loss-buffer clock fitting and rely on the algorithm's
specialized gradient routes. Hybrid branches may each own a schedule and explicit
mapping from global time; there is no requirement to map their noisy states into
each other.

For extension, supply coefficients and derivatives in the stated coordinate,
declare geometry requirements, register persistent state, and test finite-difference
derivatives and endpoint behavior. See [all configs](../../configs/noise_schedule/)
and [implementation helpers](../../core/noise_schedules/utils.py).
