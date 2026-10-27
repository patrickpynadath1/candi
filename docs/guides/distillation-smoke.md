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

## First effectiveness comparison

`examples/distillation_effectiveness.py` submits to an existing live queue using
`--queue PATH --plan PATH --submit`. Defaults are 3,000 student updates, batch size
16, one seed, and 256 validation examples. It keeps the approved smoke W&B project
and uses group `fewstep-3000`.

Alpha, SDTT, dMMD, IDLM and DMD each run on all three tasks. Each masked/Gaussian
base also has an unchanged baseline and an ordinary continued-training control.
The unchanged baseline uses zero optimizer updates and saves the imported EMA
weights for evaluation through the same pipeline. All candidates use matched
samplers at 4 and 16 NFE; unchanged bases additionally use 128 NFE. Baselines are
queued first. There are 21 optimization runs, six baseline snapshots, and 60 final
evaluations. The saved plan records source checkpoints and fully resolved configs.

SDTT promotes its teacher every 500 updates, increasing the target rollout stride
from 1/128 through 1/4 over six training stages. Alternating methods perform 3,000
student and 3,000 auxiliary updates, so their Lightning step count is 6,000.
Controls match student updates and data batch size, not total FLOPs. Fixed final
checkpoints are evaluated; no inference tuning or test-set selection is performed.

Compare each method with its own unchanged base and continued-training control
at the same NFE. Sudoku accuracy, text8 lexical metrics, and TinyStories reference
BLEU are task-specific evidence; one seed and 256 examples are an initial quality
screen, not a robust ranking. The linear flow-map and Beckmann presets are excluded
because their fresh initialization and different information path do not test
few-step improvement of the existing base checkpoints.

## Inherited affine-path map adapters

`AffineMap` wraps a pretrained single-view Gaussian algorithm. It retains the
base representation (including embedding codebooks), input projection,
preconditioning, posterior parameterization, and learned noise schedule. The
flow-map destination adapter starts at zero, preserving base predictions exactly.
The generic checkpoint importer loads base EMA into `initialization_target` and
initializes a new EMA over the complete adapted student. Teachers are frozen
copies of that initialized student. No adaptive schedule fitting is performed
during these distillation procedures.

For an affine path, write a deterministic jump as
`z_s = A_ts z_t + B_ts mean(p)`, where
`A_ts = sigma(s)/sigma(t)` and
`B_ts = alpha(s) - A_ts alpha(t)`.
Composing two jumps gives the PSD target
`p_target = w p_first + (1-w) p_second`, with
`w = A_us B_tu / B_ts`. These are inherited-schedule coefficients, not the linear
one-hot interpolation's time ratios. Teacher predictions produce the target;
the student learns destination-conditioned jumps alongside diagonal CE.

`beckmann_inherited` is explicitly a **time-conditioned characteristic extension**,
not the autonomous Beckmann paper implementation. The endpoint remains a function
of state and source time to preserve the pretrained denoiser. Its target is the
stopped-gradient endpoint plus its total directional derivative along the frozen
teacher's probability-flow field, including the time derivative. Clean-boundary
CE anchors the endpoint. Field coefficients use finite differences of the full
inherited schedule, including clock warps. This JVP path is uncompiled. Sampling
first predicts the endpoint from the prior, then refines in clean representation
space at time zero. Its 4/16-NFE baselines must use this same refinement policy.

`examples/inherited_map_comparison.py` accepts an explicit JSON mapping of task
names to base `config` and `checkpoint` paths. It prepares both distilled and
zero-update adapted baselines at 4/16 NFE and submits to the existing live queue.
The unchanged adapted baselines distinguish changes in inference policy from
improvements due to training. It does not silently convert one-hot checkpoints
into embedding-corruption checkpoints. The older pure Gaussian checkpoints used
in the first effectiveness sweep have one-hot state representations; their input
projections alone do not make their corruption an embedding-space process.

## Sampler tuning after the fixed-setting comparison

`examples/tune_distillation.py` builds manifests for `core.sweeps.nested_tuning`
from the 27 completed original comparison checkpoints. Each checkpoint × NFE
(4 or 16) × sampler is an independent case, so every sampler receives its own
selected parameters and final evaluation. This is the existing nested search
with one fixed checkpoint as its outer candidate, not a separate tuning engine.

Policies are native posterior sampling, native greedy, and entropy-bounded greedy;
Gaussian checkpoints additionally use forward-corruption sampling. Temperature
is searched on [0.5, 1.5]; EB also searches its entropy bound on [0.02, 0.5]
logarithmically. Native greedy compares greedy versus sampled terminal decoding.
Forward corruption searches sampled versus greedy clean-token proposals. Budgets
are 16 Optuna trials per searched policy, or two for native greedy.

The shared task partitions exclude previously generated subsets of sizes 16,
128, and 256. Selection uses 128 validation examples with generation seeds 41/42;
final reporting uses 256 disjoint validation examples with seeds 1041/1042.
These are held-out validation partitions, not an independent official test set.
The score is Sudoku accuracy, text8 mean unique-valid words of length at least
five, or TinyStories corpus BLEU. This creates 180 cases and retains a separate
`selected.json`, Optuna database, and `final/result.json` for each policy/budget.

## Alpha strength follow-up

`examples/alpha_sweep.py` submits k=4,8,16 on each masked task base, retaining the
3,000-update/batch-16/one-seed protocol. These are k values; effective sequence
alpha is k divided by the selected token count. The original k=.5 runs remain
separate comparisons. No additional k=0 control is included in this follow-up.

The SequenceAlpha numerical component consumes clean-token log probabilities,
target IDs and a supervised mask; it contains no corruption-specific branch.
For nonmasked CE paths, configure CorruptData.only_corrupted=false. Uniform and
embedding-Gaussian forward/backward tests cover all three new strengths. This
establishes numerical compatibility, not a theoretical equivalence for arbitrary
weighted CE, ELBO or auxiliary-objective compositions.

The implementation uses expm1 near zero and exp for strongly negative exponents:
FP32 expm1 backward can otherwise round its expm1(x)+1 derivative to zero long
before exp(x) underflows. An analytical saturated-gradient test guards this.

The tuning manifest builder accepts --prefix/--expected and --exclude to select
completed follow-ups. --wait --run-address ADDRESS waits for those checkpoints,
then invokes the existing nested Optuna tuner. The Alpha follow-up uses the same
validation partitions and per-sampler 4/16-NFE searches. Its superseded original
k4 Sudoku run is explicitly excluded in the saved launch configuration.
