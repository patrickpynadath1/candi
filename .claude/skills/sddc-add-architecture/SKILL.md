---
name: sddc-add-architecture
description: Change the SDDC network - add or swap ModularDiT slots (input, time conditioning, self-conditioning, output head), block sublayers (MLP replacements), depth-mixing connections, encoder/decoder block sizes, or the causal model - while keeping checkpoints loadable and caches correct. Use for any architectural modification or ablation.
---

# Change the architecture

Every diffusion recipe uses one network, `ModularDiT` (`core/models/modular.py`); the
autoregressive recipes use `CausalDiT` (`core/models/causal.py`). Both share the transformer in
`core/models/text_dit.py` and the layers in `core/models/layers.py`. Read
`docs/components/models.md` first.

## 1. Pick the extension point

| Change | Code | Config | Select with |
| --- | --- | --- | --- |
| how a state enters the network | an input slot class in `core/models/modular.py` | `configs/denoising_task/model/input/` | `- override /denoising_task/model/input@model.input: <name>` |
| output head | output slot class | `configs/denoising_task/model/output/` | `...model/output@model.output: <name>` |
| time conditioning | `TimeConditioning` options (`coordinate`, `adaLN`, ...) | `model.time.*` fields | `model.time.coordinate=log_snr` |
| self-conditioning injection | `SelfConditioning` options | `model.self_conditioning.*` | fields |
| the MLP sublayer of every block | `core/models/sublayers.py` | `configs/denoising_task/block_sublayer/` | `+denoising_task/block_sublayer=<name>` |
| how blocks read earlier blocks | `core/models/depth_mixing/` | `configs/denoising_task/depth_mixing/` | `+denoising_task/depth_mixing=<name>` |
| encoder/decoder sizes (E2D2) | `core/modifiers/encoder_decoder/` | `configs/denoising_task/block/` | `denoising_task/block=<name>` in its sweep |
| a new model preset | combine slots | `configs/denoising_task/model/<name>.yaml` | `denoising_task/model=<name>` |

Most new architectures are a new slot: a module that accepts the dimensions it needs
(`vocab_size`, `hidden_size`, `cond_dim`, `state_dim`, which the network fills in) and maps a
state to hidden vectors (input) or hidden vectors to logits (output). Slots are Hydra
partials (`_partial_: true`).

## 2. Rules

- **Checkpoints**: changing a slot, sublayer or depth mixing changes parameter names and
  shapes. Keep existing classes' parameter paths stable (see `PositionwiseMLP`, which keeps
  numbered names); put the new behaviour in a new option, defaulting to the old one.
- **Parameter budget**: an ablation that adds parameters should say so, or shrink something
  to match (`depth_mixing` options shrink `mlp_hidden_size`); report parameter counts.
- **Initialization that starts at the baseline** (a zero gate, as in `centered_mlp`) makes
  the ablation's starting point the plain model; test that equality.
- **Caches**: `CausalDiT` owns its KV cache; block diffusion has a clean-prefix cache. Any
  change there needs a cached-versus-uncached equality test.
- **Features for modifiers**: return `ModelPrediction` (logits and final hidden state) when a
  modifier asks for features; do not run an extra forward pass.
- **Masks and layout**: respect the sequence layout (padding, blocks); some sublayers only
  support full unmasked sequences and must reject others.

## 3. Test

Patterns: `core/tests/test_block_sublayers.py` (a sublayer through a real sweep recipe),
`core/tests/test_depth_mixing_execution.py`, `core/tests/test_model_composition.py`,
`core/tests/networks.py` (small reference networks), `core/tests/test_block_diffusion.py`
(cache equality).

- forward shapes and dtype under bf16 autocast;
- the baseline-equivalence case (gate zero, identity mixing) matches the plain model;
- the parameter count you report;
- a state-dict round trip, and that an old checkpoint of the unchanged default still loads;
- one tiny train/generate through a recipe.

## 4. Document

Add the option to `docs/components/models.md` (presets table or the sublayer/depth-mixing
section) with its parameter cost.

## Done when

The tests pass, `--cfg job` shows the new slot or option, a tiny CPU run trains and samples,
and the default recipes' checkpoints still load (`core/tests/test_checkpoint_recipe.py`).
