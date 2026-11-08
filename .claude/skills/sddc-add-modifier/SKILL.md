---
name: sddc-add-modifier
description: Add or change an SDDC modifier that cuts across the recipe parts - self-conditioning, block diffusion (and block-size mixtures), encoder-decoder (E2D2), multi-view hybrids, token feedback, variable length (insertion/deletion) - through the shared prediction and view machinery instead of a new forward path. Use when a technique should apply to several families at once.
---

# Add or change a modifier

Modifiers change how several recipe parts work together, so they apply across families. They
live in `core/modifiers/` and are switched by top-level blocks in `configs/config.yaml`
(`self_conditioning`, `block_diffusion`, `token_feedback`, `variable_length`) or by the
`configs/modifiers/` groups. Read `docs/concepts/recipe.md` (modifiers across the three
parts) and the pages in `docs/modifiers/`.

## 1. Find the hook

| Modifier | Code | Hooks it uses |
| --- | --- | --- |
| self-conditioning | `core/modifiers/self_conditioning/` | the prediction coordinator (`prediction.py`) stages one detached feedback tensor; the model's self-conditioning slot injects it |
| block diffusion, block mixtures | `core/modifiers/block_diffusion/` | sequence layout, attention masks, the clean-prefix cache |
| encoder-decoder (E2D2) | `core/modifiers/encoder_decoder/` | separate clean and noisy streams |
| hybrid views | `core/modifiers/hybrid/` | `DiffusionView` branches with their own process, schedule and clock, fused in `fusion.py` |
| token feedback | `core/modifiers/token_feedback/` | an auxiliary prediction fed back as input |
| variable length | `core/modifiers/variable_length/` | insertion during generation, deletion during forward corruption |

Reuse these hooks. A modifier that needs its own forward pass, its own loss loop, or reads the
root config is usually at the wrong boundary.

## 2. Rules

- **Opt-in and off by default**: a modifier's switch defaults to the unmodified recipe, and
  with the switch off the composed config and outputs match the recipe without it.
- **Declare compatibility**: say which families, samplers and layouts it supports, and refuse
  the rest with a clear error at construction (`core/samplers/requirements.py`,
  `core/modifiers/hybrid/training.py`'s validation).
- **Feedback needs a meaning**: expected-embedding self-conditioning requires probabilities;
  hidden-state feedback requires compatible hidden outputs. Never reinterpret scores as
  probabilities.
- **NFE**: any extra network call at inference counts toward NFE.
- **Variable length**: the birth state of inserted tokens must match the process; deletion
  happens in forward corruption only.
- **Ablations** of a modifier go in `configs/modifiers/` as `@package _global_` files that a
  sweep axis can select (see `configs/modifiers/feedback/`), each setting a
  `preparation.*_name` used in case names.

## 3. Test

Patterns: `core/tests/test_self_conditioning.py`, `core/tests/test_block_diffusion.py`,
`core/tests/test_encoder_decoder.py`, `core/tests/test_hybrid_composition.py`,
`core/tests/test_token_feedback.py`, `core/tests/test_variable_length.py`.

- off-switch equivalence with the plain recipe;
- the property the modifier promises (feedback is detached and staged once per step; cached
  equals uncached; inserted tokens are born in the process's state; prompts are preserved);
- one tiny train/generate for each family it claims to support;
- regenerate `docs/validation/compatibility.md` with
  `uv run --locked python scripts/docs/compatibility_table.py`.

## Done when

The tests pass, the modifier page in `docs/modifiers/` documents the switch, supported
families and NFE cost, and `core/tests/test_compatibility_table.py` passes.
