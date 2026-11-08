---
name: sddc-add-information-path
description: Add an information-path component to SDDC - a representation (token to continuous state and back), a corruption process (forward corruption and reverse transitions), a noise schedule (time to corruption coefficients), a token space, or a hybrid view that combines them. Use when a new way of representing or corrupting tokens is needed, or a schedule (rank-calibrated, learned, adaptive) is added.
---

# Add an information-path component

The information path decides what the model sees: how tokens are represented, corrupted, and
on what clock. Read `docs/concepts/information-path.md`, `docs/extending/adding-a-component.md`
and the component page you are extending: `docs/components/representations.md`,
`docs/components/processes.md`, `docs/components/noise-schedules.md`.

## 1. Pick the boundary

| You change | Code | Config group | Contract |
| --- | --- | --- | --- |
| token ↔ continuous encoding | `core/representations/` | `information_path/representation` | encode tokens to `[B, L, D]`, decode or score back; learned tables are module parameters |
| forward corruption and reverse step | `core/processes/` | `information_path/process` | `corrupt(clean, coefficients, noise=...)` and the reverse transition the samplers call; preserve fixed prompts |
| time → coefficients | `core/noise_schedules/` | `information_path/noise_schedule` | discrete: keep rate and its derivative; Gaussian: the full path coefficients and derivatives (`core/noise_schedules/gaussian_path.py`), not a lone sigma |
| vocabulary layout | — | `information_path/token_space` | whether an absorbing mask token is appended |
| a branch of a hybrid | `core/modifiers/hybrid/views.py` | `information_path/hybrid_view` | one `DiffusionView` with its own process, schedule, representation and clock |
| a learned or adaptive clock | `core/time_sampling/`, `core/modifiers/hybrid/time_maps.py` | `information_path/time_map` | see skill `sddc-add-denoising-component` for time samplers |

A schedule and a process must agree on what the coefficients mean; a Gaussian schedule's
rank calibration is tied to a representation's geometry. Check `docs/validation/compatibility.md`
for what the neighbouring components accept.

## 2. Implement

- One responsibility per class; dependencies as explicit constructor arguments (Hydra fills
  them from interpolations such as `${task.vocab_size}`), never the whole config.
- Reject unsupported inputs at construction with an actionable error.
- Keep precision explicit: promote to float32/float64 around transition math rather than
  relying on autocast (`core/processes/blurring.py` keeps fp32 under bf16).
- For variable length, the birth state of inserted tokens must match the process
  (`docs/modifiers/variable-length.md`).

## 3. Configure

A YAML file in the group, for example `configs/information_path/noise_schedule/<name>.yaml`
with `_target_` and arguments. Group files keep their package via a `# @package` header
(copy one from the same folder). A hybrid view goes in `configs/information_path/hybrid_view/`
and is selected with `information_path/hybrid_view@model.views.content=<name>`.

## 4. Test an independent property

Not a restatement of the formula. Patterns:

- schedules: analytic derivatives against autograd or finite differences, endpoints and
  monotonicity (`core/tests/test_gumbel_schedule.py`, `core/tests/test_rank_linear_paths.py`);
- processes: reverse transitions normalize, stay on the support, preserve prompts, and reach
  the clean state at t→0 (`core/tests/test_spherical.py`, `core/tests/test_blurring_diffusion.py`);
- representations: encode/decode round trip and gradients reach learned tables
  (`core/tests/test_representations.py`);
- one tiny train/backward/generate through a recipe that uses it
  (`core/tests/test_hybrid_composition.py`).

## 5. Wire it into recipes and docs

- A preset in `configs/experiment/` if it defines a new family (skill `sddc-new-recipe`).
- The component page in `docs/components/`, and `docs/validation/coverage.md` with the actual
  evidence level.
- `uv run --locked python scripts/docs/compatibility_table.py` when a family's accepted
  samplers or modifiers change.

## Done when

The new tests pass, a recipe using the component trains and samples on CPU, and
`core/tests/test_compatibility_table.py` passes (the generated table is current).
