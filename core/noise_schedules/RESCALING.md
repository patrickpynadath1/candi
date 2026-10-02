# Gaussian path rescaling

`RescaledGaussianSchedule` separates effective noise calibration from Gaussian
state scale. A base schedule returns `alpha`, `sigma`, and their time derivatives.
The wrapper divides both coefficients by a positive, time-dependent denominator:

| `path_type` | Denominator | Result |
| --- | --- | --- |
| `ve` | `alpha` | Signal coefficient is one |
| `vp` | `hypot(alpha, sigma)` | Squared coefficients sum to one |
| `linear` | `alpha + sigma` | Coefficients sum to one |

All preserve effective noise `sigma / alpha` and coefficient SNR
`alpha**2 / sigma**2`. Thus an argmax-error or rank calibration remains valid
under the common positive rescaling. For embeddings, its original geometry
requirements still apply. VP here describes coefficient normalization; it does
not assert that arbitrary data embeddings have unit marginal variance.

Both derivatives include the quotient rule. These schedules work with the
existing Gaussian process and posterior-derived continuous samplers. Equal SNR
means equivalent corruption marginals up to state scaling, not identical network
inputs, training behavior, or finite-step numerical trajectories. Model
preconditioning and time conditioning remain independent architecture choices.

Choose the wrapper and its base through Hydra, for example:

```sh
python main.py +experiment=gaussian_diffusion_sudoku \
  noise_schedule=rescaled_gaussian \
  noise_schedule@noise_schedule.base=exponential_ve \
  noise_schedule.path_type=vp
```

The default base is `one_hot_rank`; `path_type` defaults to `vp`. Other base
choices include `argmax_error`, `embedding_sphere_rank`, and
`linear_interpolation`. `linear` means coefficient normalization, not necessarily
linear progression through global time.

An adaptive Gaussian schedule can wrap a rescaled schedule, or be its base.
For example, `noise_schedule@noise_schedule.base=cdcd_gaussian_adaptive` places
CDCD inside the rescaling wrapper. Its observation method, adaptation signal,
and snapshot delegate unchanged to the adaptive base. Training-time adjustment
is applied once; evaluation freezes updates; nested module state persists the
fit and observation buffers in checkpoints. Fixed bases do not expose adaptive
feedback methods through this wrapper.

VP and linear normalization preserve both clean and pure-noise endpoints.
VE coordinates are undefined where the base has `alpha=0`; evaluating that
endpoint raises an explicit error. No epsilon silently substitutes a different
path. Negative, nonfinite, and jointly zero coefficients are invalid.
