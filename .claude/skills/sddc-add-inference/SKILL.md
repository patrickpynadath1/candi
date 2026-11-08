---
name: sddc-add-inference
description: Add an SDDC inference setting - a sampler, an update rule, a commitment or token-selection policy, a corrector, a time/noise grid, terminal decoding - with correct NFE accounting, prompt preservation and capability checks, and expose it as a tunable policy in sweeps. Use when changing how a trained model generates.
---

# Add an inference setting

Samplers own state initialization, time progression, stopping and NFE accounting; algorithms
only provide predictions. Read `docs/concepts/inference-settings.md` and
`docs/components/samplers.md`.

## 1. Compose before you write a sampler

Most new inference settings are a composition, not a new loop:

| You change | Where |
| --- | --- |
| clean-token choice (sample, greedy, posterior) | `TokenSelection` in `core/samplers/policies.py` |
| which positions commit per step | position policies in `core/samplers/policies.py` |
| a corrector step | `core/samplers/correctors.py` |
| one update rule for all fixed-length recipes | `ComposedSampler` in `core/samplers/composed.py` |
| the fixed time grid and loop | `core/samplers/loop.py` |

Configure a composition as YAML in `configs/inference/sampler/` (see `composed.yaml`,
`dfm.yaml`, `random_commitment.yaml`). Write a new sampler class only when control flow
differs (adaptive step acceptance, entropy-bounded completion, autoregressive caches).

## 2. Contract

- `sample(algorithm, tokens, fixed)` returns the generated tokens and metadata with
  `nfe_per_sample` (and usually `nfe`). **NFE counts every network call**, including terminal
  predictions and self-conditioning bootstraps; it is not the step count.
- `steps_for_nfe(nfe)` maps a budget to steps, so sweeps compare samplers at equal NFE
  (`core/sweeps/budget.py`). A sampler that makes two calls per step must say so; if a mode
  only applies with self-conditioning, fall back when the model has none (see
  `GaussianDDPMSampler`'s `self_conditioning` argument).
- Positions with `fixed` True never change.
- Check capabilities on attach with `core/samplers/requirements.py` (`views`, `process_method`,
  ...), so an incompatible recipe is refused with a reason (a `TypeError`) instead of
  failing inside the loop.
- Sampling-only shaping (temperature, top-k/top-p) goes through the sampling prediction
  path; it must not touch the training objective.

## 3. Expose it to sweeps

A sweep tunes inference per policy (skill `sddc-sweep`). Add a policy to a set in
`configs/inference/inference_policies/` (or a sweep's `preparation.inference`): `name`,
`sampler` (or `sampler_overrides`), a search `space` over `sampler.*` keys (not
`sampler.steps`, which the NFE budget sets) or `algorithm.predictor.temperature`/`commitment`,
`trials`, `budget_parameter: steps`.

## 4. Test

Patterns: `core/tests/test_composed_sampler.py`, `core/tests/test_gaussian_ddpm.py`,
`core/tests/test_sampler_budget.py`, `core/tests/test_forward_resampling.py`.

- prompts are preserved exactly;
- the measured NFE equals the declared NFE, and `steps_for_nfe` inverts it;
- a degenerate case with a known answer (a perfect predictor returns the clean sequence; one
  step reduces to the terminal decode);
- refusal of an incompatible family with a clear message;
- regenerate the compatibility table:
  `uv run --locked python scripts/docs/compatibility_table.py`.

## Done when

The tests pass, the sampler runs through one tiny recipe on CPU at a fixed NFE, its row in
`docs/validation/compatibility.md` is current (`core/tests/test_compatibility_table.py`), and
the samplers page documents it.
