# A diffusion language model recipe

SDDC is a framework for testing claims about diffusion language models through
composable recipes. Its scope is controlled comparisons across affordable tasks,
not large-scale pretraining or a claim that one particular diffusion algorithm is
best. The CANDI hybrid method is one recipe among many.

A recipe has three parts. This is a conceptual organization of the existing
components, not a new wrapper class.

| Part | Question | Components |
| --- | --- | --- |
| [Information path](information-path.md) | What information does the model observe at each time? | Representation, corruption process, noise schedule |
| [Denoising task](denoising-task.md) | What must the model predict, and how is it trained? | Backbone, output parameterization, objective, training time sampler |
| [Inference settings](inference-settings.md) | How are predictions turned into generated sequences? | Initialization, inference time/noise grid, update rule, terminal decoding |

## Noise schedule and training time sampler

Two components in different parts both concern time, and the recipe keeps them
apart:

- The **noise schedule** (information path, config group
  `information_path/noise_schedule`) fixes what each time `t` means: the corruption
  level at `t`. Training and generation share it. An adaptive schedule such as
  CDCD's learns it from training losses so that the loss rises linearly in `t`.
- The **training time sampler** (denoising task, config group
  `denoising_task/time_sampler`) decides how often training visits each `t`. It
  is used only in training and has no role at inference: generation never
  consults it, in any recipe.

A learned map that generation also reads belongs with the schedule: a **learned
time map** `u = Q(t)` (information path, config group `information_path/time_map`,
placed at `algorithm.time_map`). CDCD's learned loss CDF for adaptive views
(`path_schedule=cdcd`), the REPLAiD hybrid's learned importance density and the
simplex loss-derivative CDF (SDM's warp toward the `t` where the loss changes
fastest, read by `time_grid=adaptive`) are such maps. Their
recipes draw uniform `t` with the training time sampler; the views read `Q(t)` in
training and generation, and the density of `Q(t)` is the map's
([learned time maps](../components/noise-schedules.md#learned-time-maps)).

In training only their combination, together with the loss weighting, matters: a
schedule that spends more of `[0, 1]` on mid-range corruption looks the same to the
optimizer as a sampler that visits those times more often. They are separate
components for two reasons. Generation reuses the schedule but not the sampler,
so moving a choice from one to the other changes the inference grid. And an
adaptive schedule's linearity constraint pins the schedule's shape, so any further
emphasis on some times has to come from the sampler.

Every training time sampler reports the density of its draws relative to uniform.
Adaptive noise schedules divide their fitting weights by that density, so the
schedule they learn does not depend on the sampler. Each sampler also declares an
`objective_weighting`: `importance_corrected` divides the loss by the density, so
the objective is unchanged and only its variance differs, while `uncorrected`
lets oversampling also re-weight the objective. See
[training time samplers](../components/time-sampling.md) for the contract and each
sampler's default, and `+experiment=cdcd_mid_range_times` for a recipe that
combines CDCD's adaptive schedule with a non-uniform sampler.

Tasks supply data, conditioning masks, and metrics independently of these choices.
A Sudoku task does not select a corruption process. An experiment combines recipes
with a set of tasks ([preparing sweeps](../guides/programmatic-sweeps.md)), budgets, seeds, and an evaluation
protocol.

## Components and compatibility

The goal is to make compatible changes easy to express and compare. It is not to
allow every Cartesian product of components: a score predictor cannot silently
stand in for a clean-token posterior, and a Gaussian reverse rule cannot operate
on a Dirichlet state. The [component contracts](component-contracts.md) spell out
those boundaries. Execution support and paper reproduction are different claims;
see [coverage](../validation/coverage.md).

The same information path can support different denoising objectives or samplers.
Likewise, a shared backbone and cross-entropy objective can be paired with several
corruptions. Existing presets are starting recipes, not indivisible methods.
[Configuration](configuration.md) explains their Hydra assembly, while
[architecture](architecture.md) explains ownership in Python.

## Modifiers across the three parts

Modifiers can affect several parts of a recipe. Treating them as a fourth,
independent bucket would hide interactions that experiments should expose.

- [Self-conditioning](../modifiers/self-conditioning.md) changes the denoiser's
  inputs and training feedback policy, plus the feedback carried during inference.
- [Multiple views](../methods/hybrids.md) combine information paths and fuse their
  representations in a shared model. They are not synonymous with self-conditioning.
- [Insertion/deletion](../modifiers/variable-length.md) changes sequence structure
  along the information path, introduces count and optional gap-content objectives,
  and requires compatible generation edits and birth states. Current insertion
  recipes delete during forward corruption and insert during reverse generation;
  that does not imply a general reverse deletion operation.
- [Block diffusion](../modifiers/block-diffusion.md) changes attention, sequence
  layout, conditioning, and the order of generation.

For uniform insertion, newborn tokens are noise even when a gap-content head is
trained. The auxiliary prediction can change learned representations without
changing that birth rule. This distinction is part of the recipe, not a task detail.

## From recipes to claims

An experiment can compare self-conditioning, corruption form, or inference tuning
while keeping other declared choices fixed. [Nested tuning](../guides/nested-tuning.md)
selects training and inference hyperparameters on validation, then reports the
selected configuration on reserved final examples. Inference-only tuning is the
same workflow with one fixed checkpoint.

The framework makes comparisons explicit; it does not automatically establish
causality or significance. Different schedules, objectives, training budgets, or
search budgets can change the interpretation of an ablation. A single training
seed does not measure training variability. Record the complete recipe and
selection protocol with every result.
