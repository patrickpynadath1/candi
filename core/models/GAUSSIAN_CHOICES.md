# Gaussian model architecture choices

`ContinuousDiT.forward(state, time, sigma, alpha)` retains the learned state
embedding transform and the subsequent `input_projection` for both one-hot and
embedding representations. The representation's token embedding table and these
model projections are separate parameters with separate roles.

`model.input_preconditioning` selects:

- `gaussian` (default): multiply projected state features by
  `1 / sqrt(alpha**2 + sigma**2)`, before the affine input projection.
- `none`: omit that scaling, retaining both learned input transforms.

`model.time_conditioning` selects the scalar sent through the existing sinusoidal
and learned timestep embedder:

| Choice | Coordinate |
| --- | --- |
| `log_one_minus_time` (default) | `-log(1 - time)` |
| `time` | The supplied global time |
| `log_sigma` | `log(sigma)` |
| `log_effective_noise` | `log(sigma) - log(alpha)` |
| `log_snr` | `2 * (log(alpha) - log(sigma))` |

The existing default clamps time below one. Logarithmic noise coordinates clamp
individual coefficients to `epsilon` (default `1e-6`) to remain finite at clean
or pure-noise endpoints. This numerical saturation means exact scale invariance
only applies away from that clamp. Gaussian preconditioning requires a nonzero
path scale; alpha and sigma must not both vanish.

With Gaussian preconditioning, `log_effective_noise` and `log_snr` give the same
model input under a common positive rescaling of state, alpha, and sigma.
`log_sigma` intentionally changes under that rescaling. These are model
conditioning choices, not changes to the corruption path or its derivatives.

No parameter names or tensor shapes change. Existing checkpoints load strictly,
and omitting the new options preserves the previous forward calculation.
