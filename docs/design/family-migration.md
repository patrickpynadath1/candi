# Moving a standalone family onto the view path

This is the working guide for the parallel unification workers (one family per
worker, each in its own branch). Design and status: [modular-dit.md](modular-dit.md).
The finished examples are `core/algorithms/mdlm.py` (token network, views in the
network) and `core/algorithms/uniform_posterior.py`.

## Goal

The family's class keeps its name, constructor arguments, checks and every preset
config, but becomes a one-view `HybridDiffusion` recipe on an adopted `ModularDiT`.
It must be **behaviorally equivalent** to the former class. Bit-for-bit equality is
not the goal, and clean code is never compromised to get it.

- **May differ:** rounding, summation order, parameter registration or initialization
  order, and the order random numbers are drawn in.
- **Must match:** the math; which time, keep or coefficient feeds the network,
  objective and observers; clamp and epsilon values; exposed precision options;
  masked and counted positions; self-conditioning source and space; prompt handling;
  terminal steps; NFE; schedule and time-sampler feedback; constructor checks.
- **Tests:** compare loss and gradients with `torch.testing.assert_close` at tight
  float32 tolerances. Compare samples exactly while the draw order is unchanged;
  otherwise compare per-step outputs under shared noise, plus NFE. Keep exact checks
  where they come for free. Report every intentional behavior-neutral difference.

## Building blocks (already in place, tag `unify-4-foundation`)

- `core/models/legacy.py:adopt(legacy_network, time_coordinate=...)` turns DiT,
  SimplexDiT, SphereDiT and ContinuousDiT into a `ModularDiT`. It reuses the same
  tensors and draws no random numbers (`core/tests/test_adopt.py`).
  - A token DiT can take `views=` (MDLM).
  - The other kinds keep their single-state input slot. Pass the view to
    `HybridDiffusion(..., views=[view])`; it is registered after the network and
    predictor, where the standalone classes held process, schedule and
    representation.
- `core/modifiers/hybrid/one_view.py:OneViewAlgorithm`. Mix it in first, before
  `HybridDiffusion`. It forwards `process`, `schedule` and `representation` to the
  view, including assignment.
- `HybridDiffusion` extension points (`core/modifiers/hybrid/training.py`):
  - **Class attributes:**
    - `prediction_pathway`: endpoint prefix, `posterior` or `plugin`.
    - `prediction_kind`: the process kind accepted.
    - `clamp_observed`.
  - **Overridable hooks:**
    - `_corrupt(tokens, points, corruption_mask)`
    - `_objective(objective, prediction, tokens, time, points, states)`
    - `_observe(...)`
    - `model_inputs(states, points, global_time)`
    - `prediction_mean`
    - `loss` itself, if the family's loss is bespoke.

  The objective context carries a single view's path `alpha`/`sigma`.
- `ModularDiT` time options:
  - `TimeConditioning(time_source="point")` reads the view's local, training-mapped time.
  - Coordinates `negative_log_keep_exact` (MDLM) and `negative_log_keep_floor` (1e-12, spherical).
- `ModularDiT.clean_token_hidden` gives block diffusion the legacy clean prefix
  for single-state inputs.
- `core/samplers/view_calls.py:predict(algo, state, time, endpoint)`. A family
  sampler keeps its own loop and calls a view algorithm through this helper.
- Tests: `core/tests/view_equivalence.py` provides `build`, `batch`,
  `assert_training_identical(..., ordered=)`, `assert_sampling_identical`, and
  `core/tests/view_inputs.py` (`token_call`, `view_prediction`). ContinuousDiT
  held bare root parameters, so its parameter order differs; pass `ordered=False`
  and say so in the test.

## Steps

1. **Freeze the reference.** Copy the current class, unchanged, to
   `core/tests/reference_algorithms/<family>.py` with absolute imports. Do the
   same for a family sampler whose call form you change, under
   `core/tests/reference_samplers/`.
2. **Rebase the class** as `class X(OneViewAlgorithm, HybridDiffusion)`:
   - Build the view: `DiffusionView(process, schedule, representation, update)`.
     Use the update rule from `core/samplers/hybrid_updates.py` that matches the
     family.
   - `adopt` the network.
   - Keep every constructor check.
   - Put the family-specific behavior in the hooks or overrides, in the family's
     own module.
3. **Keep configs unchanged.** `python main.py +experiment=<preset> --cfg job`
   must give identical output before and after, for every preset of the family.
4. **Write `core/tests/test_view_equivalence_<family>.py`.** Cover every option
   its presets use: self-conditioning on and off, each schedule, objective, input
   or decoding mode, and every sampler setting in its presets. Each case runs
   `assert_training_identical` and `assert_sampling_identical`, plus a strict
   load of the converted old state dict (`convert_state_dict` names under
   `model.`).
5. **Port the family's tests** to the view call form, then run the full local
   suite and `ruff==0.15.13`.

## Ownership rules for parallel workers

- **Edit only the files your brief lists as owned.** Do not edit:
  - `core/modifiers/hybrid/training.py`, `core/samplers/composed.py`
  - `core/models/legacy.py`, `core/models/modular.py`
  - `core/representations/noisy_inputs.py`
  - `core/modifiers/block_diffusion/*`, `core/training/performance.py`
  - `core/tests/view_equivalence.py`, `docs/design/modular-dit.md`

  If you need a shared change, override in your class or add a new file you own.
  If that is impossible, stop and report the exact change you need.
- **Shared test files** (`test_self_conditioning.py`, `test_uniform_sedd_duo.py`,
  `test_adaptive_feedback.py`, `test_sampler_composition.py`, `test_sweep_budget.py`):
  edit only your family's cases, in small hunks.
- **Commits:** one per verified step on your branch, ending with the
  Co-Authored-By line. No server, no jobs. Never touch untracked files or
  datasets.

## Running the tests (CPU, no project env needed)

```bash
DEPS=(--with torch==2.5.1 --with lightning==2.6.1 --with hydra-core==1.3.2 --with omegaconf==2.3.0
      --with optuna==4.9.0 --with 'numpy<2' --with einops --with scipy --with scikit-learn
      --with torchdiffeq --with datasets==4.8.5 --with transformers==5.8.1
      --with 'ray[default,tune]==2.59.0' --with sacrebleu --with pytest)
uv run -q --no-project --python 3.10 "${DEPS[@]}" python -m pytest -q -p no:cacheprovider core/tests
uv run -q --no-project --python 3.10 --with ruff==0.15.13 ruff check core main.py frontier.py scripts/cluster
```

Byte-level sampler snapshots (`test_sampler_equivalence`) run only on Linux
x86_64. They are rechecked on the GPU server at merge time.
