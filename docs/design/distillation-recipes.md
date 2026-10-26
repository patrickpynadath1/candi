# Distillation and flow-map recipes: implementation sketch

**Proposal, not implemented.** Based on `candiv2.1` at `f69a124`. Every configuration
below is illustrative and is **not runnable**. Method names identify presets of
shared components; they do not identify separate trainers. This proposal covers
teacher-free methods as well as distillation.

## Keep the existing three-part recipe

| Existing part | Additional choices needed |
| --- | --- |
| Information path | Coupled interpolation samples and their derivatives; forward noising versus a reverse bridge; differentiable or discrete generated states |
| Denoising task | Clean posterior versus two-time map versus autonomous endpoint; construction of training states and targets; loss reduction; model roles and update schedule |
| Inference settings | Existing posterior/update policies, learned finite-time jumps, or repeated endpoint refinement |

The **training procedure belongs to the denoising task**. It says where examples
and targets come from, which model receives gradients, and when models update.
It is not a fourth recipe group or a replacement for the information path.
A tractable marginal kernel alone does not provide an interpolation coupling,
a path derivative, or a differentiable categorical sampler.

## What exists, and the small seams to add

- `core/modifiers/hybrid/views.py::DiffusionView` already groups representation,
  process, schedule and update; `HybridDiffusion` in `training.py` provides the
  shared prediction endpoints. Reuse these for ordinary forward corruptions.
  Add coupling/bridge operations only to processes that actually support them.
- `core/training/module.py::TrainingModule.training_step` calls `algorithm.loss`.
  Extract reusable state construction and prediction from that loss path rather
  than calling a complete loss just to obtain a noisy state. A procedure assembles
  those calls and returns reduced losses and explicit metric statistics.
- `core/objectives/composition.py` already binds named inputs to small components.
  Keep numerical losses here: **no rollout, teacher promotion, optimizer step or
  checkpoint loading inside a loss**. Retain the existing tokenwise objective API.
- Add a small procedure package under `core/training/` with ordinary Python
  callables for state construction, target construction and update phases. A
  standard one-update procedure and an alternating-update procedure are enough
  initially. Do not build a configurable computation-graph language.
- Extend prediction where the method requires it: `predict_map(state, t, s)` needs
  both times in the network; an autonomous endpoint takes a state without time.
  A sampler override cannot turn today's single-time denoiser into either model.

A phase constructs states and targets, predicts, reduces losses and updates its
parameter group. Components receive explicit arguments following `Component.inputs`.
For example a rollout target takes `teacher, states, times, mask, rollout` and
returns target probabilities and scoring positions; it never calls an optimizer.

**Reduction is part of the contract.** Current `ComposedObjective` requires losses
shaped like target tokens, and `Component` forbids negative coefficients. Keep
that contract for token objectives. At the procedure boundary, explicitly reduce
any token terms, sequence terms or transport terms to scalar losses before their
combination. Use a standalone `Difference(first, second)` numerical component for
signed objectives, rather than a negative coefficient hidden in the old API.
Do not broadcast a sequence loss across tokens to satisfy a shape check.

A reduced term must supply metric numerator and denominator appropriate to its
reduction. Alpha's sequence average cannot use the current token-count-weighted
logging in `TrainingModule`. Log each term and its normalization separately.

## Methods as choices

The following table describes the defining construction, not a claim that all
paper variants share identical equations or that their presets exist today.

| Preset | State/trajectory and target | Loss and updates | Generation |
| --- | --- | --- | --- |
| Flow Map Language Models | Gaussian-to-one-hot interpolation; two-time categorical prediction parameterizes a finite map; compare a direct jump with composed jumps | Diagonal denoising CE plus map consistency; frozen/detached targets as required by the selected formulation | Learned jumps and terminal decoding |
| Discrete Flow Maps | Mean-denoiser map with explicit source/destination times; semigroup target (PSD), Eulerian target (ESD), or Lagrangian target (LSD) | Diagonal CE plus the selected consistency loss; derivative targets require JVPs, not a teacher rollout substituted for a derivative | Learned finite-time jumps |
| Discrete Beckmann Transport | Autonomous endpoint on interpolated states; directional target `stopgrad(T(I) + J_T(I) dot_I)` | Transport residual, clean-boundary CE, and scheduled anchor/composition terms; teacher-free | Endpoint prediction, optionally repeated refinement/partial commitment |
| DMD | Student-generated states noised with the chosen kernel; frozen data teacher and trainable student-distribution auxiliary | Noisy-marginal reverse-KL gradient, using teacher/auxiliary score difference; alternate generator and auxiliary updates | Few student predictions with compatible noising/bridge rules |
| D-MMD | Student proposal and a reverse-bridged intermediate state; teacher and auxiliary predictions there | Student CE-to-teacher minus CE-to-auxiliary; auxiliary fits student distribution with teacher regularization | Short stochastic reverse chain |
| Alpha Diffusion Loss | Real data and forward masking; ground-truth sequence target | Sequence-level alpha loss, generally CE warm-start; no teacher or auxiliary | Existing masked decoding |
| SDTT | Forward-mask real examples; teacher unmasking rollout records distributions at newly revealed positions | Student distribution matching (e.g. reverse KL); periodically promote student to teacher and reduce rollout budget | Existing masked sampler with fewer steps |
| Inverse-distilled DLM | Student-generated examples followed by forward corruption; teacher and optimized auxiliary denoising losses | Student **minimizes teacher loss minus auxiliary loss**; auxiliary minimizes its own loss; alternating bilevel approximation | Compatible few-step student decoding |

Important distinctions:

- FMLM's Gaussian interpolation lives in ambient space; its categorical endpoint
  does not imply every intermediate state is on the simplex.
- DMD Gaussian pathwise training and Dirichlet/simplex DMD need different kernel
  gradient estimators. A categorical REINFORCE variant additionally needs its
  density-ratio/discriminator machinery. They are not a kernel-name substitution.
- D-MMD means **Discrete Moment Matching**, not maximum mean discrepancy. Its
  reverse bridge is not interchangeable with IDLM's forward corruption. Uniform
  D-MMD may require hard clean samples for the auxiliary because noises correlate.
- Alpha uses a joint target probability: `(1 - P**alpha) / alpha`, with the CE
  limit at zero. Stable log-domain evaluation, empty-mask behavior, and the
  paper's length/alpha normalization must be explicit. Independent token alpha
  losses would be a different objective.
- A frozen teacher's parameters need no gradients, but derivatives through its
  input may still be required. `requires_grad_(False)` and `no_grad()` are not
  interchangeable. Beckmann's target JVP must be computed before detachment.

## Proposed configuration shape

Use existing Hydra composition and `_target_` bindings in the eventual files.
The short names below denote proposed component presets, not a new expression
language. The three headings describe conceptual ownership; they do not require
moving all existing configuration keys into three new wrapper objects.

```yaml
# Proposed denoising task for SDTT; NOT RUNNABLE.
prediction: clean_posterior
procedure:
  states: corrupt_real_data
  targets:
    component: unmasking_rollout
    record: first_reveal_distribution
    terminal_remaining: teacher_distribution
  roles:
    student: {initialization: pretrained_checkpoint, trainable: true}
    teacher: {initialization: copy_student, trainable: false}
  phases:
    student:
      loss: {component: reverse_kl, inputs: [student_probs, target_probs]}
      reduction: masked_token_mean
      parameters: student
      target_gradients: detached
  schedule:
    component: progressive_teacher_promotion
    interval_steps: 10000
    rollout_budget_factor: 0.5
inference:
  sampler: composed
  transition: reverse
  positions: entropy
  commitment: freeze
# Pin decoding and rollout policies to the selected paper experiment.
```

A map preset changes the prediction and training-target components, not the
trainer class:

```yaml
# Proposed Discrete Flow Maps PSD choices; NOT RUNNABLE.
prediction: two_time_mean_denoiser_map
procedure:
  states: coupled_interpolation
  targets: {component: composed_map, times: ordered_triplet}
  roles: {student: {trainable: true}}
  phases:
    student:
      terms:
        - {loss: diagonal_ce, reduction: masked_token_mean}
        - {loss: map_consistency, reduction: example_mean}
      parameters: student
      target_gradients: detached  # explicitly choose the formulation's routing
inference: {executor: learned_map, times: configured_jump_grid, decoding: greedy}
# ESD/LSD substitute directional-target construction plus the matching loss;
# they are not merely alternate names for composed_map.
```

An alternating preset names the two losses and their input/gradient ownership:

```yaml
# Proposed IDLM choices; NOT RUNNABLE.
prediction: clean_posterior
procedure:
  states: {component: generated_then_forward_corrupted, estimator: soft_relaxation}
  roles:
    student: {trainable: true}
    teacher: {initialization: pretrained_checkpoint, trainable: false}
    auxiliary: {initialization: pretrained_checkpoint, trainable: true}
  phases:
    auxiliary:
      loss: auxiliary_denoising
      generated_gradients: detached
      parameters: auxiliary
    student:
      loss:
        component: difference
        first: teacher_denoising
        second: auxiliary_denoising
      generated_gradients: enabled
      parameters: student
      frozen_parameters: [teacher, auxiliary]
  schedule: {component: alternating, auxiliary_updates: 1, student_updates: 1}
inference: {sampler: composed, transition: forward_corruption}
# Bind denoising losses to process-specific objectives and reductions.
# Soft relaxation needs a compatible differentiable path. Finite auxiliary updates
# approximate, rather than exactly solve, the inner optimization.
```

Alpha uses ordinary corruption with sequence-alpha loss; D-MMD substitutes a
bridge and CE terms; DMD supplies its score-gradient estimator and auxiliary fit.
Beckmann uses directional endpoint targets and scheduled terms. These are
component choices, not subclasses named after papers.

## Optimizer, checkpoint and sampler ownership

`TrainingModule` currently constructs one optimizer, computes EMA over all
`algorithm.parameters()`, and recognizes a fixed list of training metrics.
Adding auxiliary networks to that object without changing these assumptions is
incorrect. Introduce explicit role parameter groups; the usual case is one
student group. Alternating updates require Lightning manual optimization with
correct accumulation, BF16 behavior, clipping and scheduler cadence. Freeze
non-active parameters while retaining input gradients when required. Promotion
is a procedure lifecycle event, never a loss side effect.

Resume checkpoints contain role states, optimizer/scheduler states, EMA,
progressive stage and RNG. Inference exports contain only the selected student
and its recipe; teacher and auxiliary weights must not inflate or alter them.
EMA ownership is explicit per role, with student-only as the ordinary default.

`core/samplers/composed.py::ComposedSampler` already separates token selection,
position selection, transition and commitment. Its `sample` is under `no_grad()`;
it cannot be reused as a differentiable generator during training. Share pure
transition computations, not a call to that no-grad executor. Teacher rollouts
may use a no-grad execution path with an observer for reveal targets.

The current loop always requests a single-time posterior. Learned maps and
endpoint refinement need prediction/step operations that can express two times
or no time. Extract the common bounded iteration, clamping, decoding and NFE
accounting once those operations are concrete; retain separate small step
policies for posterior transitions, map jumps and endpoint refinement. Do not
pretend a map is an ancestral update or duplicate the full sampler for each paper.
Count every prediction and nested refinement in actual NFE.

## Implementation order and checks

1. Extract the current forward-denoising path without changing its numerical
   behavior. Introduce explicit loss reductions and metric statistics; verify
   existing ordinary training, insertion terms and accumulation equivalence.
2. Add sequence-alpha and teacher-rollout target components. Verify alpha's CE
   limit, sequence coupling and masked reductions; SDTT target/reveal masks,
   frozen teacher, promotion/resume, and ordinary sampler output equivalence.
3. Add model-role ownership and alternating updates. Test exact gradient routes,
   teacher immutability, auxiliary separation, difference-loss sign and student
   export. Check Gaussian/Dirichlet estimators against tiny analytic problems;
   test bridge versus forward marginals separately. Then add DMD/D-MMD/IDLM
   presets with their distinct estimators and objectives.
4. Add two-time map and autonomous endpoint interfaces. Check identity/diagonal
   behavior, semigroup consistency on an analytic flow, JVPs against finite
   differences, target stop-gradient, and refinement/NFE accounting. Add map and
   Beckmann presets after those tests pass.

Reject incompatible combinations during construction: absent input derivatives,
wrong prediction parameterization, unsupported bridge, non-differentiable state
estimator on a pathwise objective, or mismatched teacher token space. Do not
silently detach or approximate a missing operation. Task data, held-out tuning,
Ray scheduling and the existing nested experiment pipeline stay outside this
procedure design.

## Paper references

- [Flow Map Language Models](https://arxiv.org/html/2602.16813v3)
- [Discrete Flow Maps](https://arxiv.org/html/2604.09784v2)
- [Discrete Beckmann Transport Models](https://arxiv.org/html/2609.15903v2)
- [Gaussian DMD](https://arxiv.org/html/2609.40235v1) and
  [Dirichlet/simplex DMD, Appendix G](https://arxiv.org/html/2609.35553v1#A7)
- [Discrete Moment Matching Distillation](https://arxiv.org/html/2603.20155v1)
- [Alpha Diffusion Loss](https://arxiv.org/html/2609.38066v1)
- [Self-Distillation Through Time](https://arxiv.org/html/2410.21035v2)
- [Inverse-distilled DLM](https://arxiv.org/html/2602.19066v1)
