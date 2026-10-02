# Algorithms

Algorithms connect the components for training and expose prediction endpoints for
sampling. They own batch preparation, calling time samplers and schedules,
corruption, model evaluation, prediction interpretation, objective evaluation, and
masked loss reduction. The trainer owns optimizer steps, EMA, and checkpointing;
the sampler owns the generation trajectory.

An algorithm's `loss(tokens, loss_mask, corruption_mask, ...)` returns a dictionary
containing `loss`, detached `loss_sum`, and `num_tokens`. Masks have separate
meanings: corruption permission does not imply supervision, and a fixed generation
prompt is not a loss mask. Tokens and masks are normally `[batch, length]`;
normalized token predictions are `[batch, length, vocabulary]`.

`generate(sampler, prompt_tokens, prompt_mask)` enters evaluation mode, delegates
to the sampler, and restores the previous mode. Reverse-transition mathematics
belongs in processes, not in this method.

## Implementations and selection

The [algorithm configs](../../configs/algorithm/) declare the components to
instantiate recursively. Prefer a matching [experiment](../../configs/experiment/)
when changing method families: changing `algorithm` alone does not change the
process, predictor, model, or objective.

| Selector | Training construction |
| --- | --- |
| `mdlm` | Absorbing corruption, constrained clean posterior, weighted CE |
| `d3pm_absorb` | Discrete-time absorbing bound plus reconstruction |
| `sedd_absorb` | Absorbing log scores and score entropy |
| `uniform_diffusion`, `duo` | Normalized plug-in estimates and uniform diffusion objective |
| `uniform_posterior` | Clean-token posterior, uniform reverse bridge, token CE |
| `gaussian_diffusion` | Representation encoding, affine Gaussian path, clean-token CE |
| `spherical_flow_matching` | Unit embeddings and spherical corruption |
| `simplex_diffusion` | Dirichlet corruption and clean-token CE |
| `blurring_diffusion` | Sequence-aware spectral corruption over learned embeddings |
| `replaid` | Learned embedding likelihood bound and specialized gradient routes |
| `hybrid_diffusion` | Multiple noisy views fused before one backbone evaluation |
| `autoregressive`, `permuted_autoregressive` | Causal token prediction in fixed or sampled order |

## Prediction and feedback boundaries

[PosteriorDiffusion](../../core/algorithms/posterior.py) shares
`training_prediction`, `sampling_prediction`, and `greedy_prediction` routing.
It supplies model inputs and expected-embedding conversion to the shared prediction
coordinator. It does not embed self-conditioning decisions in MDLM's loss.
[DiscreteDiffusion](../../core/algorithms/discrete.py) shares the separate
plug-in/score training lifecycle. These bases represent different mathematical
contracts rather than a universal interface every method must inherit.

Adaptive schedule observations are routed through
[adaptive.py](../../core/algorithms/adaptive.py); validation must not update them.
The algorithm chooses whether feedback is raw CE or the objective and supplies
masked, detached per-example observations. RePlaid's differentiable learned
schedule has additional gradient semantics; it is not interchangeable with a
buffer-fitted adaptive clock.

## Extending

Add an implementation under [core/algorithms](../../core/algorithms/) and its
config. Reuse a base only when its prediction and loss meanings match. Validate
component compatibility at construction, including predictor domain, schedule
output, representation geometry, and model dimensions. There is no static typing
system proving combinations valid; constructor and operation checks remain
necessary. Test gradients, prompt preservation, checkpoint state, and sampler
integration, not only config construction. See [adding a method](../extending/adding-a-method.md).
