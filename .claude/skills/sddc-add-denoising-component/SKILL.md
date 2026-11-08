---
name: sddc-add-denoising-component
description: Add a denoising-task component to SDDC - a parameterization (what network outputs mean - posterior, plug-in, score, Gaussian prior), an objective or loss weighting, a training time sampler (including adaptive or learned ones like CDCD), or, rarely, a new algorithm class. Use when changing what the model predicts or how it is trained.
---

# Add a denoising-task component

The denoising task fixes what the network predicts and how it is trained. Read
`docs/concepts/denoising-task.md`, the "Prediction meaning" table in
`docs/concepts/component-contracts.md`, and the page you extend:
`docs/components/predictors.md`, `docs/components/objectives.md`,
`docs/components/time-sampling.md`, `docs/components/algorithms.md`.

## 1. Pick the boundary

| You change | Code | Config group |
| --- | --- | --- |
| meaning of network outputs, constraints, inference shaping | `core/parameterizations/` | `denoising_task/parameterization` |
| loss math or weighting | `core/objectives/` (compose with `core/objectives/composition.py`) | `denoising_task/objective` |
| distribution of training times | `core/time_sampling/` | `denoising_task/time_sampler` |
| how predictions are formed (shared) | `core/algorithms/posterior.py`, `core/modifiers/self_conditioning/prediction.py` | — |
| orchestration that genuinely differs | a new class in `core/algorithms/` | `denoising_task/algorithm` |

Same `[B, L, V]` shape does not mean interchangeable: a posterior, a plug-in estimate and a
score feed different reverse equations. An objective must say which of them it consumes.

## 2. Rules

- **Training and sampling predictions stay separate.** Temperature, top-k/top-p and
  commitment act only on the sampling path; they must never change the training objective.
- **Algorithms assemble losses; they do not loop over time.** `loss(tokens, loss_mask,
  corruption_mask=...)` returns `{loss, loss_sum, num_tokens}`; generation delegates to a
  sampler. Reuse the posterior coordinator instead of adding a new forward path.
- **A new algorithm class is rare.** Most methods are recipes of existing parts
  (`docs/extending/adding-a-method.md`). Add a class only when the orchestration differs, and
  override only the hooks that differ (every family runs on the view path,
  `core/modifiers/hybrid/`).
- **Learned or adaptive state** (an adaptive time sampler, a learned schedule) is an
  `nn.Module` owned by the algorithm, so optimizers, EMA, checkpoints and device moves include
  it. It observes training losses only, never validation, and synchronizes across ranks.
  `post_optimizer_step` hooks run after EMA updates.
- Validation loss is logged as `val/nll` even when it is not a likelihood; say what yours is.

## 3. Configure and test

- YAML in the group (copy a neighbour's `# @package` header).
- Tests of independent properties:
  - objectives: equals a reference value on a hand-computed case, has the right gradient
    sign, and its weighting integrates as documented (`core/tests/test_forward_distribution_objective.py`);
  - time samplers: sample distribution, warmup and resume behaviour, observe-train-not-val,
    distributed sync (`core/tests/test_cdcd_time_sampler.py`, `core/tests/test_continuous_adaptive_feedback.py`);
  - parameterizations: outputs satisfy their constraint (normalized, masked support) and
    sampling shaping leaves the training loss unchanged (`core/tests/test_uniform_sedd_duo.py`);
  - a checkpoint round trip for any new learned state (`core/tests/test_checkpoint_recipe.py`);
  - one tiny train/generate through a recipe.
- Document on the component page and update `docs/validation/coverage.md`.

## Done when

The tests pass, a recipe using the component trains and samples on CPU, and if it changes what
a family accepts, `scripts/docs/compatibility_table.py` has been rerun and
`core/tests/test_compatibility_table.py` passes.
