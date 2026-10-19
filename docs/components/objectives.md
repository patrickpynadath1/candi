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

### Configured composition

`ComposedObjective` binds named inputs to standalone components through
`Component`, then combines them with `SumTerms`. The same `SumTerms` is used for
insertion losses. Input names and component argument signatures are checked at
construction; incompatible distribution kernels fail when preparation runs.
There is no ELBO class or method-name branch.

`ForwardDistributions` prepares target and predicted distribution objects once.
`TokenProbabilities` supplies categorical clean distributions;
`CleanVector` supplies clean vectors and predicted mean vectors. The latter does
not marginalize a Gaussian mixture over possible tokens. GIDD and Gaussian
processes currently expose `forward_distribution`; other families must implement
that contract before using these recipes.

`ForwardKL(target, predicted)` calls PyTorch's analytic `kl_divergence`.
Unsupported distribution pairs raise rather than silently using a Monte Carlo
estimate. `ObservedRatioPenalty(target, predicted, observed)` computes
`expm1(log_ratio)-log_ratio` from the two log probabilities/densities. Gaussian
vector dimensions form one event, leaving a loss per token. Gaussian kernels
require strictly positive variance.

```yaml
# Excerpt: each component declares its actual arguments via configuration.
terms:
- _target_: core.objectives.composition.Component
  component:
    _target_: core.objectives.forward_distribution.ForwardKL
  inputs:
    target: target_distribution
    predicted: predicted_distribution
  coefficient: 1.0
```

Available recipes:

- `composed_ce`: ordinary CE through the same composition, without preparation.
- `forward_kl`, `observed_token_is`, `forward_kl_is`: unweighted categorical terms.
- `gaussian_forward_kl`: Gaussian KL using predicted clean mean vectors.
- `gidd_elbo`: KL + observed-ratio penalty + interpolating discrete weight.
- `gidd_clipped`: the same composition with a clipped weight (maximum 1).

For `q_t=alpha_t*x+b_t`, `InterpolatingDiscreteWeight` computes
`(b'_t - alpha'_t/alpha_t*b_t)[observed] / q_t(observed|x)`.
It consumes the actual path coefficients and process mixing derivatives, with
no schedule reconstructed inside the loss. It requires interior times.
`ClippedWeight` wraps this weight independently of the loss terms.
The ELBO interpretation assumes the matching GIDD kernel and time integration;
changing training time sampling or time warping requires the corresponding
importance/Jacobian correction. The dynamic GIDD+ weighting is not included.
Using these divergence terms with another kernel does not establish an ELBO.

```bash
uv run --locked python main.py +experiment=gidd_corruption objective=gidd_elbo
uv run --locked python main.py +experiment=gidd_corruption objective=forward_kl_is \
  objective.terms.1.coefficient=0.5
```

The hybrid algorithm supplies processes, representations, coefficients and states
in the context. `view_index` selects the forward kernel. Only preparation reads
this context; individual terms receive their configured explicit arguments.
Algorithms retain ownership of the task mask and final reduction.

### Insertion terms

`GapCountLoss` computes the Poisson count loss without the target-only constant.
`GapTokenLoss` computes clean-token CE at the gap assigned to each deleted token.
Both accept explicit arguments and return per-example losses after applying their
respective gap/token masks and insertion rate. `InsertionObjective` only adapts
aligned insertion inputs and preserves the existing reporting/reduction API;
it combines configured components using the shared `SumTerms`.

`variable_length/insertion.yaml` specifies both components, their input bindings,
and coefficients. The existing `count_weight` and `content_weight` config fields
feed those coefficients so saved experiments and head-off ablations still work.

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
