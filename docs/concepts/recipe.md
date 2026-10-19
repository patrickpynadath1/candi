# A diffusion language model recipe

CANDI is a framework for testing claims about diffusion language models through
composable recipes. Its scope is controlled comparisons across affordable tasks,
not large-scale pretraining or a claim that one particular diffusion algorithm is
best. The name of the repository does not restrict the recipes to CANDI's original
method.

A recipe has three parts. This is a conceptual organization of the existing
components, not a new wrapper class.

| Part | Question | Components |
| --- | --- | --- |
| [Information path](information-path.md) | What information does the model observe at each time? | Representation, corruption process, noise schedule |
| [Denoising task](denoising-task.md) | What must the model predict, and how is it trained? | Backbone, output parameterization, objective, training time sampler |
| [Inference settings](inference-settings.md) | How are predictions turned into generated sequences? | Initialization, inference time/noise grid, update rule, terminal decoding |

Tasks supply data, conditioning masks, and metrics independently of these choices.
A Sudoku task does not select a corruption process. An experiment combines recipes
with a [task suite](../guides/task-batteries.md), budgets, seeds, and an evaluation
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
