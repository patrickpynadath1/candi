# Adaptive noise schedules

`SFLMAdaptiveSchedule` and `CDCDAdaptiveSchedule` wrap keep-rate schedules.
`SFLMGaussianSchedule` and `CDCDGaussianSchedule` wrap Gaussian paths. All
components live in `core`; no legacy modules are imported.

S-FLM defaults to `coordinate: legacy_time`: record the sampled times unchanged,
fit expected per-example loss with a spline and ridge regression, retain the
nonnegative loss slope, mix uniform coverage, and invert its cumulative sum
using PCHIP. The fitted keep-rate curve uses bias-corrected EMA. This matches the
reference S-FLM fitting path, including the original pointwise-CDF extrapolation.
`coordinate: physical` explicitly selects the local alternative: record normalized
corruption instead, integrate intervals with exact endpoints, and fit on that
stable coordinate.

CDCD defaults to physical coordinates and raw `cross_entropy`. It fits expected
loss with inverse-density-weighted isotonic regression, samples a monotone
piecewise-linear loss CDF on a fixed bin grid, adds uniform coverage, and inverts
it. This ports the repository's CDCD-style approximation, rather than the original
CDCD algorithm's trainable bin logits. Its flat-loss fallback is uniform progress.

The shared clock owns warmup, finite observation filtering, distributed observation
gathering, ring buffers, refit cadence, EMA and checkpoint restoration. Algorithms
supply detached, masked per-example observations through `observe`; evaluation
and sampling do not update the clock. `loss_signal: objective` adapts to the
optimization loss; `cross_entropy` requires a normalized posterior or plug-in
predictor. Unnormalized score outputs cannot silently be treated as token CE.

For Gaussian schedules, physical corruption means progress along the fixed base
path. The wrapper evaluates the base at the adapted progress and multiplies both
alpha and sigma derivatives by the clock derivative. Representation requirements
and `training_time` come from the base; training-time scaling is applied once.
These Gaussian wrappers extend the ported scalar adaptive clock to general paths;
they are not a legacy Gaussian implementation copied verbatim.

The reference legacy CDF can extrapolate slightly past the clean endpoint. Raw
`alpha_t` and its derivative retain that fitted curve for reference comparison.
Public schedule evaluation bounds keep rates to the base endpoints and zeros the
derivative where clipped. Positive times remain strictly below keep=1 using the
next representable value toward noise, avoiding singular training objectives;
time zero retains the exact clean endpoint. Gaussian clocks inherit these bounds.
Physical-coordinate fits preserve exact endpoints without extrapolation. This
boundary stabilization is intentional and is distinct from raw reference parity.

Select `noise_schedule=sflm_adaptive` or `noise_schedule=cdcd_adaptive` for discrete
algorithms. For the Gaussian experiment use
`+experiment=gaussian_diffusion_sudoku noise_schedule=sflm_gaussian_adaptive`
(or `cdcd_gaussian_adaptive`). The nested `noise_schedule.base` is independently
configurable; the defaults use log-linear keep rates and one-hot rank respectively.
The Gaussian base can be selected through Hydra, for example
`noise_schedule@noise_schedule.base=linear_interpolation`.

Defaults retain the reference buffer size 12800, refit interval 50, grid size
1000, warmup 1000 and EMA 0.9. S-FLM uses 20 cubic spline knots, ridge 0.001 and
uniform mix 0.001. CDCD uses 100 bins, maximum importance 100 and uniform mix 0.01.

Validation includes independent S-FLM fit reconstruction, physical endpoints,
finite-difference derivatives, Gaussian chain rule, nested checkpoint/resume,
training-only updates, buffer overflow/filtering, flat-loss behavior and Hydra
composition. A direct comparison against the legacy source also matched raw
S-FLM (both coordinates) and CDCD keep curves and derivatives across three EMA
refits to tolerance 1e-12. No training-quality claim follows from these tests.
