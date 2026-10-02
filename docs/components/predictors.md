# Predictors

A predictor defines the mathematical meaning of raw network output. Normalized
arrays are not automatically interchangeable: a clean-token posterior and a
uniform plug-in estimate have different roles in reverse transitions.

[Parameterizations](../../core/parameterizations/) own output interpretation and
all optional logit shaping. Algorithms choose explicit training, sampling, or
greedy pathways. The choice is not inferred from `model.training`, because
validation loss must also use unshaped predictions.

## Prediction domains

| Config selector | Meaning |
| --- | --- |
| `posterior` | Clean-token log posterior; excludes absorbing mask and fixes visible tokens |
| `unconstrained_posterior`, `gaussian_posterior` | Clean-token log posterior without absorbing visible-token constraints |
| `autoregressive` | Posterior with reserved-token exclusion |
| `d3pm` | Unconstrained normalized output for D3PM's objective |
| `plug_in` | Softmax-normalized clean-token estimate for uniform/DUO plug-in mathematics |
| `score` | Absorbing log scores, **not** a normalized token distribution |
| `gaussian_prior` | Posterior with an explicit Gaussian output-prior correction |

The [configs](../../configs/parameterization/) specify the actual constraint
objects. [constraints.py](../../core/parameterizations/constraints.py) separates
support exclusion from observed-token conditioning. A Gaussian noisy state is
not an observed discrete token and must not receive absorbing constraints.

## Shaping order and endpoints

[PosteriorParameterization](../../core/parameterizations/posterior.py) exposes
`training_posterior`, `sampling_posterior`, and `greedy_posterior`;
[PlugInParameterization](../../core/parameterizations/plug_in.py) has corresponding
`*_plugin` endpoints. Both expose `probabilities` and `entropy` for consumers of
their respective domains.

1. Optional softcap applies `c * tanh(logits / c)`, with positive finite `c`.
2. Mandatory exclusions apply before normalization.
3. Sampling may apply temperature, top-k, then top-p.
4. Normalization and mandatory observed-token constraints produce the prediction.
5. Explicit greedy prediction or commitment can project it to a point mass.

Softcapping is part of the model-output definition during training **and**
inference. Temperature and filters are inference-only. Defaults are
`softcap=null`, `temperature=1`, `top_k=0`, `top_p=1`.
Utilities in [utils.py](../../core/parameterizations/utils.py) implement the
individual transforms. Exclusion follows softcapping so a forbidden `-inf` logit
cannot become a finite capped value.

`probabilities(prediction, apply_commitment=False)` converts log probabilities;
it does not compute a reverse transition. EB measures entropy before commitment
projection. Consumers promote to FP32 where their numerical operations need it;
conversion itself need not silently change the input dtype.

## Output-prior correction

[GaussianPriorPosterior](../../core/parameterizations/gaussian_prior.py) adds
`alpha / sigma**2 * (state @ E.T)` to network logits before posterior
normalization. The algorithm supplies state, path coefficients, and the embedding
table explicitly. The predictor does not reach into the representation. This
unit-embedding correction is used by the RePlaid recipe, not automatically by
ordinary Gaussian one-hot diffusion.

## Extending safely

Add a parameterization with named endpoints expressing its prediction domain and
a config under `parameterization/`. Wire it only to processes/objectives that
consume that domain. In particular, hybrid posterior mixing rejects plug-in
predictors, and posterior-mean self-conditioning requires actual token
probabilities. Add tests showing inference shaping does not alter training loss,
constraints survive filtering, and entropy is measured before commitment.
