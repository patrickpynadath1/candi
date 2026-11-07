# Modular DiT and one algorithm path (in progress)

Goal: one network, one algorithm wrapper and one sampler loop, all configured
from YAML, with declared compatibility contracts. Status is tracked at the end.

## Network: one DiT assembled from slots

Every model already shares `TextDiT` execution (blocks, rotary positions,
swappable block execution). What differed between `DiT`, `SimplexDiT`,
`SphereDiT`, `ContinuousDiT` and `HybridModel` falls into five slots. Each slot
is a module that owns its learned weights and is built by the network with the
dimensions it needs (Hydra `_partial_: true`), so a config only names choices.

| Slot | Choices |
| --- | --- |
| `input`: noisy state -> hidden | `TokenInput` (embedding lookup), `SimplexInput` (expectation / argmax), `LinearInput` (projection, optional preconditioning), `EmbeddedProjectionInput` (ContinuousDiT embedding then projection), later `ViewsInput` (hybrid sum over views) |
| `time`: conditioning vector | coordinate (`time`, `zero`, `negative_log_keep`, `log1p_negative_time`, ContinuousDiT's `log_one_minus_time`, `log_sigma`, `log_snr`, ...), AdaLN on/off, optional time vector added to the input |
| `self_conditioning` | source (`posterior_mean`, `hidden`), feedback space (`hidden` or `state` dimension), scale |
| `output`: hidden -> logits | `AdaLNHead` (incl. temperature column, tied embedding), `EmbeddingHead` (projection, embedding, bias) |
| blocks | unchanged: `block_execution`, sublayers, MLP width, AdaLN in blocks |

Interface: `forward(states, points, global_time)`, the hybrid form, so one
network serves single- and multi-view recipes. Construction order follows the
legacy classes (input, time embedder, blocks, output, time input projection,
self-conditioning) so a fresh run with the same seed initializes identically.

## Algorithms and samplers

Every diffusion family runs through `HybridDiffusion` with one or more views and
the composed samplers. Standalone class names (`MDLM`, `GaussianDiffusion`, ...)
remain as constructors for their recipes. Families that need a different
prediction kind (SEDD scores, D3PM discrete time, DUO / uniform plug-in, RePlaid)
follow afterwards; autoregressive and the latent autoencoder stay separate.

## Contracts

Components declare what they produce and need (process: state kind and
prediction kind; schedule: output kind and representation; network slots: state
kind, view count, self-conditioning space, layout and cache support; update
rules; samplers and policies; modifiers). One check runs when a recipe is built
and when a sampler is attached, and a compatibility table is generated from the
declarations and tested by building every combination it lists.

## Old checkpoints and results

No checkpoints are released, so core code carries no per-layout load hooks.
`core/models/legacy.py` converts old model configs and state dicts; it is used
by the equivalence tests during the refactor and for a final check that
regenerates the blog's ruqi6000 results exactly (`scripts/analysis/
regenerate_final.py`). The reference state is tag `v2.1-blog-reference`.

## Status

- [x] Posterior uniform on the hybrid path (bit-identical).
- [x] Modular DiT: token / simplex / sphere / continuous presets bit-identical (core/models/modular.py, legacy.py, test_modular_dit.py).
- [ ] HybridModel as a DiT with `ViewsInput`.
- [ ] MDLM, block diffusion, E2D2 on the modular path.
- [ ] Gaussian, simplex, spherical, blurring on the modular path.
- [ ] Contracts and generated compatibility table.
- [ ] Remove legacy classes; regenerate blog results through the converter.
