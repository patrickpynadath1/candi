# Token CE objective choices

`CrossEntropy` consumes normalized token log probabilities and returns one loss per
token. Defaults reproduce the previous gather-only CE, with no smoothing or
weighting and no new state-dict entries.

- `objective.label_smoothing=0.1` uses uniform full-vocabulary label smoothing,
  matching PyTorch CE. Smoothing rejects nonfinite log probabilities, including
  excluded (`-inf`) classes: an absorbing constrained posterior cannot silently
  smooth over forbidden tokens.
- `objective=cross_entropy_power` multiplies CE by
  `scale * max(coordinate, epsilon)**exponent`. The default exponent zero and scale
  one are neutral. For example, set `objective.weighting.coordinate=sigma` and
  `objective.weighting.exponent=-2` for inverse noise-variance weighting.
- `objective=cross_entropy_learned` uses `exp(-w)*CE+w`. The scalar `w` comes from
  a timestep embedding and MLP, whose final layer is initialized to zero. This
  preserves the original FLM weighting formula while putting the trainable head
  in the objective instead of the backbone. The default coordinate is time.

The algorithm supplies `ObjectiveContext(time, alpha, sigma)` using one tensor of
shape `(batch,)` per field whenever `objective.requires_context` is true. The
objective does not query the model, process, or schedule. Available coordinates
are `time`, `alpha`, `sigma`, `effective_noise` (sigma/alpha), `snr` (alpha²/sigma²),
and `log_snr`. Ratios and logarithms floor coefficients with explicit `epsilon`;
power weighting additionally floors its nonnegative coordinate. These are
coefficient SNRs, without a data-variance correction. Learned weighting is not
clipped, preserving FLM's formula.

Context is not implicitly detached. The algorithm decides whether a learned
schedule should receive gradients; ordinary representation/model loss gradients
remain intact. Adaptive schedule fitting separately uses detached observations.

All weighting parameters are child parameters of the objective, so assigning the
objective to the algorithm registers them for optimizer, EMA, and checkpoint
handling. Loss-mask reduction happens in the algorithm after weighting; it must
mask the learned additive term as well as CE.

Legacy FLM feeds its pre-interpolation schedule alpha into its weighting head.
That quantity need not equal a new Gaussian path's signal coefficient. Selecting
`coordinate=alpha` is only an exact legacy match when those coordinates coincide.
These options apply to clean-token CE; the dedicated uniform, absorbing score,
and MDLM mathematical objectives are unchanged.
