# Objectives

Recipe context: [denoising task](../concepts/denoising-task.md).

Objectives own loss mathematics. They receive predictions, targets, and explicit
coefficient/context tensors rather than reaching into a model or schedule. Most
return `[batch,length]` losses; algorithms apply masks and reduction. Learned
weighting parameters live in the objective so optimizer, EMA, and checkpoints can
track them.

## Objective families

| Selector | Prediction and mathematical role |
| --- | --- |
| `weighted_ce` | MDLM clean log posterior weighted by `-alpha'/(1-alpha)` |
| `cross_entropy` | Clean-token negative log probability |
| `cross_entropy_power` | CE times a configured power of a supplied coordinate |
| `cross_entropy_learned` | `exp(-w)*CE+w`, with learned context-dependent scalar `w` |
| `forward_kl` | KL between target and predicted forward marginals |
| `observed_token_is` | Pointwise Itakura–Saito penalty at the observed noisy token |
| `forward_kl_is` | Configurable sum of those two terms |
| `uniform` | Uniform/DUO plug-in diffusion loss |
| `score_entropy` | Absorbing score entropy, including reference constant |
| `d3pm` | Absorbing discrete variational term plus clean reconstruction CE |
| `replaid` | Prior, reconstruction, and continuous diffusion bound terms |

[Implementations](../../core/objectives/) and [configs](../../configs/objective/)
define the call signatures. A normalized plug-in estimate is not a posterior
merely because CE could be computed on it. The chosen process/objective pair must
share the predictor's interpretation.

## Clean-token CE options

[CrossEntropy](../../core/objectives/cross_entropy.py) gathers target log
probabilities, with optional `label_smoothing`. Smoothing follows uniform
full-vocabulary PyTorch CE; it rejects nonfinite log probabilities rather than
smoothing over excluded `-inf` classes. It therefore cannot silently apply to an
absorbing posterior with forbidden outputs.

When required, the algorithm passes
[ObjectiveContext](../../core/objectives/weighting.py) with `[B]` fields `time`,
`alpha`, and `sigma`. Supported weighting coordinates are `time`, `alpha`, `sigma`,
`effective_noise`, `snr`, and `log_snr`. Ratios/logarithms use explicit epsilon
floors. These are coefficient SNRs, without data-variance correction.

Power weighting computes `scale * max(coordinate,epsilon)**exponent`; exponent
zero and scale one are neutral. Learned weighting uses a timestep embedding and
MLP whose final layer starts at zero, so the initial loss is ordinary CE. It is
not clipped. Masking must remove its additive `w` term as well as the CE term.
The context is not implicitly detached: the algorithm determines whether schedule
parameters should receive gradients. Detached adaptive fitting is a separate path.

```bash
uv run python main.py +experiment=gaussian_diffusion_sudoku \
  objective=cross_entropy_power \
  objective.weighting.coordinate=sigma objective.weighting.exponent=-2
```

A legacy method's conditioning scalar may differ from the current Gaussian path's
alpha. Matching a coordinate's name alone does not establish exact weighting parity.

## Specialized losses

### Forward-distribution terms

[ForwardDistributionObjective](../../core/objectives/forward_distribution.py)
applies the selected view's forward marginal to the one-hot target and predicted
clean-token probabilities. It computes these distributions once, then passes
them to independent `ForwardKL` and/or `ObservedTokenIS` modules. The former is
`KL(q || p)`; the latter is `r - log(r) - 1`, where `r=q(z_t)/p(z_t)`.
There is no sampled or argmax prediction on this loss path.

```bash
uv run --locked python main.py +experiment=gidd_corruption objective=forward_kl
uv run --locked python main.py +experiment=gidd_corruption objective=observed_token_is
uv run --locked python main.py +experiment=gidd_corruption objective=forward_kl_is \
  'objective.weights=[1.0,0.5]'
```

The combined preset lists the KL term first and the observed-token term second.
`terms` is a list of modules; `weights` supplies nonnegative ablation coefficients.
An optional `weighting` module can transform the combined token loss using the
existing weighting interface. Equal coefficients alone do **not** implement the
GIDD ELBO: its schedule-derived token weights are not included.

The hybrid algorithm supplies `processes`, `coefficients`, and `states` tuples in
`ObjectiveContext`. `view_index` selects one view explicitly (default zero), even
when the prediction is conditioned on several views. The selected process must
provide `marginal(clean_probabilities, coefficients)` and a discrete token state;
GIDD is the currently tested implementation. Other process families are not
automatically compatible. Algorithms retain ownership of loss masking/reduction.

### Method-specific objectives

[WeightedCrossEntropy](../../core/objectives/weighted_ce.py) excludes the exact
clean endpoint because its denominator vanishes. Visible absorbing tokens have
zero CE due to mandatory posterior constraints.

[RePlaidObjective](../../core/objectives/replaid.py) returns a `RePlaidTerms`
dataclass with prior, reconstruction, diffusion, and total tokenwise FP64 terms.
Its diffusion term is a squared clean-embedding prediction error weighted by the
SNR derivative. Reconstruction and diffusion subsets are rescaled by their batch
fractions. The algorithm owns the multiple predictions and detached gradient routes;
the objective owns these terms, not the generic self-conditioning modifier.

The block modifier's optional clean autoregressive CE is computed from its clean
stream and combined with the existing diffusion loss. It does not replace the
base objective; see [block diffusion](../modifiers/block-diffusion.md).

For a new objective, define required domains and shapes, expose unreduced losses,
and test masking, gradients, endpoint limits, and any additive weighting terms.
Do not move sampler policy or raw batch handling into the objective.
