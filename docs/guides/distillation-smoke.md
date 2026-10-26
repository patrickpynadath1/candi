# Composable training procedures and the smoke sweep

A procedure is an optional part of the denoising task. Each `Phase` composes state
construction, prediction, target construction, a numerical objective and a
reduction. `Procedure` owns frozen teachers, auxiliary models and progressive
promotion; the existing Lightning module owns optimizers and EMA. Named methods
are YAML compositions under `configs/procedure/`, not separate trainers.

## Implemented recipes

| Preset | Initialization | Implemented construction |
| --- | --- | --- |
| `alpha` | Task-matched masked checkpoint | Mask-normalized sequence-alpha objective, k=0.5 |
| `sdtt` | Task-matched masked checkpoint, copied frozen teacher | Two-step unmasking targets; reverse KL; optional progressive promotion |
| `dmmd` | Masked student/teacher/auxiliary | Generated soft proposal, reverse-bridged state, CE difference and teacher-regularized auxiliary CE |
| `idlm` | Masked student/teacher/auxiliary | Generated proposal, detached categorical forward corruption, weighted masked teacher-minus-auxiliary CE |
| `dmd` | Gaussian student/teacher/auxiliary | Soft proposal in fixed teacher embedding coordinates, Gaussian noising, reverse-KL score-difference surrogate, auxiliary soft CE |
| `flow_map` / `discrete_flow_map` | Fresh Gaussian–one-hot two-time map | Diagonal CE and linear mean-denoiser semigroup consistency |
| `beckmann` | Fresh autonomous endpoint | Boundary CE and signed directional-JVP squared residual |

The two map names select the same implemented FMLM-direct/DFM-PSD construction;
the sweep runs it once. The Gaussian path is in one-hot ambient space for maps,
which differs from the older learned-embedding Gaussian checkpoints. Loading
those checkpoints into a new one-hot map would not be an exact warm start.

These are core implementations, not reproductions of every paper setting.
The map presets do not implement ESD/LSD or FMLM's frozen-velocity distillation.
Beckmann omits its anchor and refinement training curriculum. DMD currently uses
the Gaussian pathwise estimator, not Dirichlet or REINFORCE variants. D-MMD and
IDLM presets support masking; uniform variants need their own validated losses.
Generated-state recipes draw partially corrupted real-data sources and a lower
matching time, rather than training only at the deterministic all-mask prior.

## Configuration

For a compatible existing hybrid recipe, select a procedure:

```yaml
procedure:
  _target_: core.training.procedures.composition.Procedure
  phases:
  - _target_: core.training.procedures.composition.Phase
    states:
      _target_: core.training.procedures.states.CorruptData
      only_corrupted: true
    prediction:
      _target_: core.training.procedures.states.PosteriorPrediction
    targets:
      _target_: core.training.procedures.states.DataTargets
    objective:
      _target_: core.training.procedures.losses.SequenceAlpha
      k: 0.5
    reduction:
      _target_: core.training.procedures.losses.ExampleMean
```

`training.initialize_from` imports only student weights and student EMA from a
core checkpoint. `checkpoint.resume` restores the complete procedure and optimizer
state. Teacher/auxiliary parameters never enter student EMA. Inactive parameters
are frozen without generally disabling input derivatives. Alternating phases
currently require one microbatch per update; unsupported accumulation is rejected.
Their `global_step` counts both optimizers: 200 student updates plus 200 auxiliary
updates appear as 400 Lightning steps.

Generation still uses the existing task/evaluation pipeline. Masked presets use
composed reverse sampling; Gaussian DMD uses clean-token sampling followed by
forward corruption. Learned maps and autonomous refinement share a map iteration
loop with separate step policies. Map validation reports `val/diagonal_ce`, not a
likelihood claim. Task solve rates, text8 word metrics and TinyStories BLEU are
computed through the existing evaluators.

## Smoke sweep

`examples/distillation_smoke.py --plan` discovers exact no-SC, six-layer task
checkpoints from a specified prior sweep and records a full plan in the cache.
The default is one seed, 200 student updates, batch size 4, BF16, and 16 validation
examples at 4 and 16 NFE. Map consistency starts after 100 warmup updates. The
Beckmann JVP path uses math attention and is explicitly uncompiled; ordinary
recipes retain block compilation. No test-set tuning or new dataset download is
performed.

Seven distinct recipes × three tasks = 21 training jobs and 42 inference jobs.
The shared FMLM-direct/DFM-PSD recipe is recorded as an alias. The fresh maps are
roughly 21.5M parameters; existing masked bases are 20.19M (Sudoku), 20.22M
(text8), and 20.45M (TinyStories). These small runs validate execution, finite
loss/gradients, checkpointing and generation. They do not establish comparative
quality or convergence.

Ray's existing `TrainingInferenceSearch` and durable `SubmissionQueue` schedule
both stages. Each successful training run adds its inference jobs. The live
submission queue stays open for further jobs; close it through `SubmissionQueue`
when finished to let the driver exit and write its final dataframe. Per-job
checkpoints, CSV metrics, generation records and logs are available before then.
