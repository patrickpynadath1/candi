# Inference settings

Inference specifies how a trained denoiser generates samples along its information
path. The sampler coordinates these choices:

| Choice | Meaning |
| --- | --- |
| Initialization | Initial noise/prior state and treatment of fixed prompt tokens |
| Inference time/noise grid | Which points on the path are visited, in what order, and with what evaluation budget |
| Update rule | How a prediction changes the state: stochastic reverse transition, flow integration, or commitment policy |
| Terminal decoding | How the remaining state or final prediction becomes discrete tokens |

An inference grid is not necessarily a separate learned noise schedule. Usually
it queries the trained path, including any fitted adaptive clock, at a selected
set of times. Some samplers expose custom grids or adaptive stepping; support is
sampler-specific. This is why the documentation uses **inference time/noise grid**
when distinguishing it from the information path's noise schedule.

[Samplers](../components/samplers.md) own the generation loop; processes supply
compatible transition mathematics. [Predictors](../components/predictors.md) own
temperature and filtering. Not every setting applies to every update rule.

## Greedy has more than one meaning

A top-1 clean-token estimate projects the predicted posterior before an update.
Greedy terminal decoding takes the final most likely tokens. Either can be used
inside an otherwise stochastic sampler. In particular, Gaussian top-1 ancestral
sampling can still add transition noise; it is not automatically a deterministic
ODE solver. Record both choices rather than labeling everything simply “greedy.”

NFE counts model evaluations, including terminal calls. A nominal step count is
not interchangeable with NFE: multi-stage solvers and early-stopping policies
have different costs. Our fixed-budget tuning reserves the terminal call and
checks the evaluator's reported maximum NFE.

## Validation selection

[Nested tuning](../guides/nested-tuning.md) searches each configured sampler on
validation examples. It selects the sampler and settings on validation, then
reports only that winner on reserved final examples. Training hyperparameters
are selected using the same validation criterion. With an existing checkpoint,
the outer training search has one fixed candidate.

See [sampling and evaluation](../guides/sampling-and-evaluation.md) for entrypoints,
[experiment terminology](../guides/experiments.md) for the hierarchy, and
[the recipe overview](recipe.md) for the other two parts.
