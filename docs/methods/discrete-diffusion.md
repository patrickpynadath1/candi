# Discrete diffusion

Discrete methods corrupt token IDs directly. The algorithm assembles training inputs and predictions; the [process](../components/processes.md) owns transitions and the [sampler](../components/samplers.md) owns generation. Prediction semantics matter: a posterior, plug-in estimate, and score are different mathematical objects.

| Method | Experiment | Prediction | Objective | Default sampler |
|---|---|---|---|---|
| MDLM | `sudoku_hard` (or main defaults) | Absorbing clean-token posterior | Weighted CE | Entropy bounded |
| UniformDiffusion | `uniform_diffusion` | Normalized plug-in weights | Uniform diffusion objective | Uniform ancestral |
| Posterior uniform | `uniform_posterior` | Unconstrained clean-token posterior | CE | Uniform ancestral |
| SeddAbsorb | `sedd_absorb` | Absorbing score | Score entropy | Analytic |
| Duo | `duo` | Normalized plug-in weights | Uniform diffusion objective | Uniform ancestral |
| D3PMAbsorb | `d3pm_absorb` | Token posterior without observed-token clamping | Discrete variational bound plus reconstruction | Ancestral |

## MDLM

An absorbing process keeps a clean token with probability $a(t)$ and otherwise replaces it with the mask token. The log-linear schedule supplies the keep rate and its derivative. MDLM draws antithetic training times and uses weighted token CE. Its posterior excludes the mask from clean predictions and fixes already revealed tokens to their observed values.

```bash
uv run python main.py +experiment=sudoku_hard
uv run python main.py +experiment=sudoku_hard sampler=ancestral
```

Sampling temperature, top-k, top-p, and commitment belong to the [predictor](../components/predictors.md); they are excluded from the training pathway. MDLM defaults to no model time conditioning. The entropy-bounded sampler evaluates entropy before greedy commitment.

## Two uniform constructions

Uniform corruption has marginal $q(x_t=j\mid x_0=i)=a_t\mathbf{1}_{j=i}+(1-a_t)/V$. Replacement can produce the original token, so a visible token does **not** imply that its clean value is known.

The plug-in method normalizes network logits and inserts these weights into the uniform reverse-transition expression. Posterior uniform instead predicts $p(x_0\mid x_t)$ and averages individually normalized bridges:

$$
p(x_s=j\mid x_t=k)=\sum_i p(x_0=i\mid x_t=k)
\frac{[a_t/a_s\,\mathbf{1}_{j=k}+(1-a_t/a_s)/V]
[a_s\mathbf{1}_{j=i}+(1-a_s)/V]}
{a_t\mathbf{1}_{i=k}+(1-a_t)/V}.
$$

The posterior implementation evaluates this in O(V), handles the uniform-noise endpoint, and returns the observed token at identical times. Plugging an already conditioned posterior into the original plug-in formula would condition on the observation again.

```bash
uv run python main.py +experiment=uniform_posterior
uv run python main.py +experiment=uniform_posterior sampler=uniform_entropy_bounded
uv run python main.py +experiment=uniform_diffusion
```

Posterior uniform requires an unconstrained posterior, time-conditioned model, keep-rate schedule, and ordinary CE. Both uniform samplers accept matched posterior/process or plug-in/process pairs. [Hybrids](hybrids.md) require the posterior variant.

## SEDD, DUO, and D3PM

SEDD uses an absorbing score parameterization and score-entropy objective. Its analytic sampler consumes scores; changing only the predictor to a posterior is not a valid assembly.

DUO uses a curriculum that switches between discrete uniform corruption and its calibrated one-hot Gaussian relaxation. The relaxed input is scaled by curriculum inverse temperature, then the model's soft-input embedding applies softmax. Separately, the output plug-in predictor normalizes network logits. These are two different normalization operations. Inference retains uniform transitions; DUO is not assembled as a parallel-view hybrid.

D3PM samples a discrete time grid matched to the objective step count. It evaluates both corrupted-input prediction and clean-input reconstruction. Its supported schedule is log-linear; its predictor must not clamp visible tokens as MDLM does.

```bash
uv run python main.py +experiment=sedd_absorb
uv run python main.py +experiment=duo
uv run python main.py +experiment=d3pm_absorb
```

Uniform, SEDD, and DUO presets retain long reference training budgets (one million steps). Override `trainer.max_steps` explicitly for a short check. Named presets also set their own batch and warmup values; inspect the composed config before launching.

## Evidence and limits

MDLM has historical Sudoku reproduction checks and pretrained OpenWebText execution checks. The other discrete ports have mathematical, configuration, gradient, and sampling regression coverage; this does not establish matched full-budget task quality. See [coverage](../validation/coverage.md) and [recorded results](../validation/reproduction-results.md).

Implementation entrypoints: [algorithms](../../core/algorithms/), [processes](../../core/processes/), and [experiments](../../configs/experiment/).

Uniform EB uses **step-local acceptance**: selected positions take the clean-token
prediction for that step, while unselected positions take the uniform reverse
transition. All generated positions are reconsidered on the next step, including
previously selected positions. Only prompt tokens remain fixed within an active sequence. A sequence stops when
all generated positions are selected in the **same step**; acceptance accumulated
over earlier steps does not count. Unfinished sequences continue up to
`sampler.steps`; endpoint outputs are returned with `completed_per_sample=false`
if they have not met the simultaneous-acceptance criterion.
