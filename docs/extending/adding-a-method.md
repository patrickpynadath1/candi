# Adding a method

A method is often a configuration of existing components. Add a new algorithm
class only when the training/prediction orchestration genuinely differs.

## Map the mathematics first

Write down the clean representation, corruption path, prediction meaning, training
objective, and reverse update. Then map each item to an existing implementation.
The [Gaussian family](../methods/gaussian-diffusion.md), for example, exposes many
methods through representation, schedule, model-conditioning, and objective choices.
A different sigma progression alone does not require another Gaussian algorithm.

Check what remains different: coupled loss terms, branch state, auxiliary training
passes, conditioning transforms, and checkpoint state may require new orchestration.
RePlaid and hybrids are useful examples of cases that are more than a noise schedule.

## Compose a recipe

An experiment YAML belongs in root `configs/experiment/`. Use `# @package _global_`
and override the relevant existing groups. The following is the structure of a
recipe, not a new algorithm provided by this repository:

```yaml
# @package _global_
defaults:
  - override /algorithm: gaussian_diffusion
  - override /model: continuous_dit
  - override /representation: one_hot
  - override /process: gaussian
  - override /parameterization: gaussian_posterior
  - override /objective: cross_entropy
  - override /noise_schedule: one_hot_rank
  - override /sampler: euler
  - override /token_space@token_space: data_only
```

A complete recipe must also select a compatible schedule and time sampler and
supply any required model/algorithm options. Copy the structure from a verified
existing recipe such as [gaussian_base.yaml](../../configs/experiment/gaussian_base.yaml),
then inspect composition with `uv run python main.py +experiment=<name> --cfg job`.
Do not assume that replacing only `algorithm` automatically selects its companions.

Prefer task-independent method recipes plus explicit task/architecture presets
where possible. A Sudoku-specific validation preset can set sequence/model/data
settings without making the algorithm depend on Sudoku.

## If an algorithm class is necessary

Follow the established family interface:

- Accept model/process/predictor/schedule/objective dependencies explicitly.
- Implement training assembly and method-specific feedback; keep the objective's
  mathematics in the objective component where it can be reused.
- Expose prediction endpoints that prepare model inputs and parameterize outputs.
- Delegate generation to a sampler, which owns the update loop.
- Register learned submodules so optimizer/checkpoint/EMA ownership is explicit.

For posterior methods, inspect [PosteriorDiffusion](../../core/algorithms/posterior.py)
and the [prediction coordinator](../../core/modifiers/self_conditioning/prediction.py)
before adding another forward path. Self-conditioning should reuse those pathways
rather than introduce method-specific bootstrap logic. Sequence-wide operations
must honor the segment metadata needed by block diffusion; see
[blurring diffusion](../methods/blurring-diffusion.md).

For hybrids, compose the process, representation, schedule, time map, and sampler update in a `DiffusionView`; no branch adapter class is needed.
Each branch maintains its own noisy view, clock, and reverse update. Shared model
fusion and posterior interpretation must remain compatible; the current CE hybrid
framework does not implement a RePlaid hybrid objective.

## Validate and document

Start with fixed-input mathematical checks, then a small training/backward/sampling
run. Compare a reference checkpoint only when architectures, parameterization,
corruption, and sampling semantics actually match. If any of those changed, state
that the comparison is approximate or requires fresh training.

Check relevant modifiers and representations deliberately rather than claiming
universal compatibility. Record raw/EMA selection, precision, seed, sample count,
training steps, and actual NFE for quality measurements. Add the method recipe and
limitations to `docs/methods/`, and update [coverage](../validation/coverage.md) and
[reproduction results](../validation/reproduction-results.md) only with observed
results. Configuration availability alone belongs in the implementation column.
