# Variable-length diffusion

The insertion modifier adds forward deletion and reverse insertion to a shared
`DiffusionView` assembly. A single continuous process uses one view; hybrids use
several views with exactly the same sequence edits. Continuous positions are
inserted directly into the continuous state, at the current global noise level.

## Core mechanism: structure and content are separate

Ordinary token diffusion changes the state of existing positions. Structural
corruption changes which positions exist. These are composed operations:

```text
clean tokens:                   A B C D
forward deletion:               A     D       (B and C are absent, not masked)
content corruption of survivors: noisy(A), noisy(D)
training gap targets:            [0, 2, 0]    (before A, between A/D, after D)
reverse generation:              denoise survivors + insert states into gaps
```

The positions of deleted tokens are known only when constructing training
labels. The network receives the compact sequence; it does not receive the
original offsets or the missing token count as input. A sequence with `n`
survivors has `n + 1` insertion gaps, including both ends. A learned boundary
state lets the model predict an insertion even when there are no survivors.

The gap head predicts a missing-token count and a clean-token distribution for
each gap. The count controls **where and how many** positions to create; the
content process controls **what state those positions have** and how they denoise.
A shared edit map moves surviving positions without changing their order. In a
hybrid, the same map is applied to every view, so their positions stay aligned.

In this implementation, **deletion is forward corruption; insertion is the
reverse structural operation**. Generation inserts and denoises. It does not
currently delete already-generated tokens or learn a reverse deletion head.

## Composition at a glance

| Aspect | What it controls | Interaction with insertion/deletion |
| --- | --- | --- |
| Content process and representation | The noisy value at an existing position | Supply the forward kernel used to initialize births and the ordinary denoising update. |
| Structural schedule | Which positions survive and the reverse insertion rate | Independent of the content schedule; both use the current global time. |
| Hybrid views | Several noisy representations of the same token sequence | Share edits and clean predictions; each view applies its own birth law. |
| Self-conditioning | Feedback from a previous prediction | Feedback follows surviving positions; new positions start with zero feedback. |
| Block diffusion | Which context is visible and which region may change | Only the active logical block is corrupted/expanded; completed blocks are frozen. |
| Adaptive training-time sampling | Which times receive training examples | Can fit token loss alone or include the weighted insertion losses. |
| Inference timestep grid | Where reverse updates are evaluated | Currently a fixed descending grid in `InsertionSampler`, independent of adaptive training-time sampling. |

## Recipes

```bash
# Gaussian continuous state with learned insertion.
uv run python main.py +experiment=insertion_continuous task=text8

# Gaussian + absorbing hybrid, with aligned continuous births and mask tokens.
uv run python main.py +experiment=insertion_hybrid task=text8

# The same modifier and sampler, with a spherical content process.
uv run python main.py +experiment=insertion_continuous task=text8 \
  hybrid_view@model.views.content=spherical
```

The presets reuse `HybridDiffusion` for the content assembly and install
`VariableLengthDiffusion` through the `variable_length=insertion` config group in
`TrainingModule`. Model, EMA, optimizer and checkpoint construction include the
gap head and any trainable insertion laws. The inner assembly can have one or
many views. Existing native single-process algorithms are not automatically
converted; construct the corresponding one-view assembly explicitly.

Select a sequence task such as Text8, TinyGSM or OpenWebText. Fixed-shape Sudoku
metrics reject variable-length generation. Text8 still trains on its configured
windows, so this alone does not supply a natural variable-length training corpus.

## Ownership

| Component | Responsibility |
| --- | --- |
| `core/sequence_edits.py` | Shared deletion/insertion maps, lengths, gather/scatter of surviving state and feedback. |
| `core/processes/deletion.py` | Sample forward deletion and retain alignment for training targets only. |
| `core/noise_schedules/insertion.py` | Structural survival probability and insertion clock. |
| `core/modifiers/variable_length/birth.py` | Per-view initialization of new positions and compatible training corruption. |
| `core/modifiers/variable_length/model.py` | Boundary features, gap count and clean-content heads over one shared backbone. |
| `core/objectives/insertion.py` | Gap count and clean-content supervision. |
| `core/modifiers/variable_length/training.py` | Coordinate the content assembly with structural training. |
| `core/samplers/insertion.py` | Advance content, apply shared insertions, initialize births, preserve prompts and report lengths. |

No Gaussian equations or process-name switches are embedded in the structural
sampler. Each view retains its representation, local time map, schedule, forward
kernel and content update. Named insertion policies are selected by
`DiffusionView.name`, so changing view order does not change policy assignment.

## Initialization at the current noise level

At a sampler step from global noise time `t` to `s < t`, existing positions first
advance using each view's update (or the insertion law's birth-relative update).
For continuous views, the gap head's clean-content posterior initializes new
positions at **s**, through the destination view point:

```
clean_estimate = representation.posterior_mean(gap_posterior)
birth_state = process.corrupt_state(clean_estimate, point_s.coefficients, active)
```

`ForwardInsertion(estimate="mean")` implements this operation, including Gaussian
`alpha(s) * clean_estimate + sigma(s) * normal_noise`. The time map of each view
is applied before evaluating its schedule. Each birth receives independent
forward noise, even when multiple births share a gap. Existing states are moved
without being forward-corrupted again.

`ForwardInsertion(estimate="sample")` instead samples a clean token from the gap
posterior, encodes it, and applies the forward kernel. This is the generic default
for unlisted continuous views and works with spherical and simplex processes. It
avoids treating an unconstrained posterior mean as a valid point on every
representation's manifold. The Gaussian preset explicitly selects the mean rule.

`MaskInsertion` initializes the corresponding absorbing position as a mask.
Its training corruption respects the insertion/unmasking coupling: conditional
on a birth at noise time `b >= t`, the mask survival probability is
`(1 - keep(t)) / (1 - keep(b))`. Prompts are never corrupted or deleted. A hybrid
therefore creates continuous state and an aligned mask simultaneously.

The current presets use the following birth policies. A process can be a single
view or one branch of a hybrid; it does not need a masked branch to support
insertion.

| View | New position at destination time `s` |
| --- | --- |
| Gaussian | Predict the clean posterior mean, then draw `alpha(s) * mean + sigma(s) * noise`. |
| Masked / absorbing | Create `[MASK]`; subsequent unmasking follows the coupled survival law above. |
| Simplex / Dirichlet | Sample a clean token, then apply the Dirichlet forward kernel at `s`. |
| Spherical | Sample a clean token, encode its unit-sphere state, then apply spherical corruption at `s`. |
| Uniform discrete | Draw uniform token noise, independent of the gap posterior. |

Continuous births use the current global noise level. Uniform births always
start as uniform noise. `UniformInsertion` conditions training corruption and
reverse bridges on each token's birth time `b`, with effective keep rate
`(keep(t) - keep(b)) / (1 - keep(b))`. This is zero at birth and one at the clean
endpoint. The sampler carries birth times through sequence edits. With multiple
views, each view maps global times through its own time map and schedule. For block generation, this clock is restarted for each
active block; completed blocks remain clean throughout.

## Training path and objectives

CANDI uses noise time zero for clean data and one for the empty endpoint. The
structural schedule is `survival(t) = (1 - t)**power`, with `power >= 1`.
Each editable clean token receives a birth time; tokens not yet born are removed.
Surviving continuous tokens receive their usual global-time corruption. Source
indices are used to gather labels and count deleted tokens in each gap, and are
never exposed as transformer position IDs.

The shared backbone predicts content for surviving positions. A learned left
boundary makes the empty sequence executable; neighboring hidden features and
a right boundary feed heads predicting missing counts and clean gap content.
The objective combines the inner token objective with:

- The insertion-hazard-weighted count loss `expected - target * log(expected)`.
- A weighted clean-content cross entropy over the deleted tokens assigned to
  each gap, computed without a dense target histogram.

The gap content head learns a distribution over missing tokens in the gap.
Continuous births in a gap use the same predicted distribution; masked and uniform
births ignore it. For these discrete views it is auxiliary training supervision.
Set `variable_length.predict_gap_tokens=false` and
`variable_length.objective.content_weight=0` to remove the content head and loss
while retaining gap-count prediction. A birth law requiring a clean posterior
rejects this setting. Several continuous births in a gap use the same distribution. Their clean estimates are not an
ordered joint prediction. Losses are normalized by the original loss-mask count,
so random deletion does not change the batch denominator. Training and validation
use training posteriors, independently of inference temperature/top-k settings.

This is an **approximate continuous insertion model**, not an exact time reversal
or a reproduced FlexMDM/DID likelihood objective. Even with only masking, the
inner preset's token CE does not include the paper's full variational weighting.
The coupled mask corruption and structural count supervision are reusable pieces;
the continuous birth rule is the explicit estimate-x0-then-forward-kernel rule.
No quality or speed improvement is implied by the integration tests.

## Sampling, prompts and padding

The sampler begins with compacted protected prompt tokens. Placeholder runs in
`prompt_mask` identify editable gaps but do not prescribe how many tokens to
insert. A fully supplied compact prompt defaults to appending. Direct callers can
supply `editable_gaps` to `InsertionSampler.sample` for explicit infilling regions.
Insertions split an editable gap into editable subgaps; closed prompt interiors
remain closed as indices shift.

For a reverse step `t -> s`, define the deleted fraction `D(t) = 1 - survival(t)`.
The integrated insertion hazard is `H(t, s) = log(D(t)) - log(D(s))`. For the
linear survival schedule (`power=1`), this is `log(t/s)`. The model predicts a
missing count `m_g` for each editable gap, and the sampler draws
`K_g ~ Poisson(m_g * H(t, s))`. This freezes the predicted rates over the step
(Poisson tau-leaping), so finite steps can under- or overshoot the desired length.

Each reverse step performs these operations:

1. Predict survivor content, gap counts and clean gap content from the current states.
2. Advance existing content states from `t` to `s` using each view's normal update.
3. Draw insertion counts and construct one shared sequence edit.
4. Initialize every birth through its view's birth law at `s`, preserving prompts
   and independently drawing its forward noise.
5. Move surviving state and feedback through the edit, then repeat.

Generation finishes when the time grid reaches `epsilon > 0`, followed by a
final content prediction. It does **not** finish because a learned stop decision
fires, a target token count is reached, or all current masks have disappeared.
The grid and endpoint define the sampling budget; predicted counts determine
length. This finite-step scheme is approximate.

Token and gap-content sampling have separate posterior parameterizations.
`parameterization.temperature` shapes survivor predictions;
`variable_length.gap_predictor.temperature` shapes birth-content predictions
when that predictor is configured. Neither setting rescales Poisson insertion
rates. The self-conditioned Text8 presets explicitly set both to `0.7`.

`max_length` is a storage/sampling bound, including prompts, not a target length.
The default `overflow=error` reports excess insertions. Explicit
`overflow=truncate` subsamples proposed births within capacity and reports which
samples were truncated; it never removes prompt tokens and changes the sampled
length distribution.

Returned tensors use right padding and metadata includes `lengths`,
`inserted_tokens`, `truncated` and NFE. The shared evaluator passes lengths to
Text8/TinyGSM/OpenWebText metrics so padding is excluded, and preserves generated
lengths and insertion truncation flags in its report. The existing OpenWebText
reference scorer still uses its configured reference-token length cap.
Training accepts `valid_mask`; padded data modules must preserve an
`attention_mask` so padding is neither a prompt nor a training target.

Self-conditioning is supported: one fused feedback vector is gathered for
survivors, and births start with zero feedback. Attention excludes padding, and
position IDs are reconstructed from the current compact sequence.

## Composition with block diffusion

Wrap the variable-length assembly in `BlockDiffusion`. Source blocks define
**logical edit domains**, not fixed physical offsets throughout generation:

```text
completed prefix       active block                 future blocks
fixed clean tokens  |  delete / noise (training)  |  absent from model input
                    |  insert / denoise (sampling)|
```

Training and validation process all source blocks in parallel. Each block has
its own deletion map, gap targets and corrupted states; its noisy positions can
see only that block and clean context from earlier source blocks. One shared
backbone pass handles every block (plus a bootstrap pass when self-conditioning
requires it), with loss normalized by the original target-token count across
all blocks. There is no one-block-per-step training mode. Generation completes
one block's entire reverse schedule, freezes its
output, and moves to the next. Its actual generated length determines where the
next block begins. Self-conditioning resets between blocks, and all views within
the active block share the same edits.

There are three different lengths to keep separate:

- `block_diffusion.block_size`: the clean source span used to define training
  blocks and partition the requested generation window into logical blocks.
- The **current active length**: changes under deletion/insertion.
- `block_diffusion.max_length`: a per-block capacity, not an exact terminal length.
  The inner sampler's `max_length` also bounds each invocation; the effective
  active cap is the smaller bound. Neither is a full-output cap across blocks.

For example, the Text8 block presets use source windows of 256 tokens, source
blocks of 32, and a per-block cap of 64. Inference runs eight logical blocks,
each with its own reverse clock. It does not enforce eight outputs of exactly 32
tokens. Learning lengths near 32 is an empirical result of training, not an
invariant. Exact 32-token terminal blocks would require an additional
length-conditioned generation rule; that is not implemented.

Fully protected blocks are copied directly. Variable blocks currently sample
one example at a time and re-encode completed prefixes without KV caching,
even if caching is requested. Packed training sequences are required, and the
optional clean autoregressive auxiliary loss is not supported on this path.
See [block diffusion](block-diffusion.md) for visibility and API details.

## Extension boundaries

An insertion law is an `nn.Module` with `validate(view)`,
`initialize(view, posterior, point)` and
`corrupt_training(view, tokens, point, birth_times, editable)` methods. Trainable
law parameters are registered. A replacement structural schedule supplies
`birth_times`, `rate` and `integrated_rate`. Content processes retain their usual
`corrupt_state`, representations and posterior-update contracts.

Sequence-coupled blurring requires a layout-aware birth law and orchestration;
the generic law rejects it. Block diffusion supports dynamic composition by
making each source block an independent edit domain, freezing completed blocks
as clean context, and hiding future blocks. See
[block diffusion](block-diffusion.md). The token-feedback auxiliary objective
still rejects the modifier because its targets need explicit realignment.
Hidden/posterior self-conditioning is supported.

Deletion is currently the **forward corruption** operation. The reverse sampler
inserts and denoises; it does not learn to remove already-generated tokens.
Reusable edit maps include deletion, but generation-time deletion would require
a corresponding learned law and training objective.

Related formulations: [FlexMDM](https://arxiv.org/abs/2509.01025),
[From Interface to Inference](https://arxiv.org/abs/2607.26504), and
[Deletion-Insertion Diffusion](https://arxiv.org/abs/2603.23507).

## What has been demonstrated

Tests exercise alignment, birth kernels, empty sequences, prompt preservation,
padding, self-conditioning, checkpoint reconstruction, future-block isolation,
per-block capacity, and validation over all blocks. The block-composition
parallel-training regression run completed with 98 passed and 8 skipped tests,
including numerical prediction/gradient equivalence against isolated block
contexts and coverage of every block in one fused pass. The exact 22M masked
and Gaussian configurations also passed GPU training, validation, checkpointing
and eight-block generation smoke tests with self-conditioning.

Text8 pilots use approximately 22M parameters with self-conditioning. The
single-view presets cover Gaussian, masked, simplex and spherical insertion;
the block presets cover masked and Gaussian insertion. All six executed training
and generation. These are evidence that insertion is not restricted to a masked
state space and can compose with block-local generation. They do not establish
an exact reverse process, competitive language-model quality or a speed benefit.

The following historical 1,000-step block pilot used the superseded
one-block-per-step training implementation. It is not evidence for the current
parallel training path, and equal-step comparisons are not fair because the
old path supervised fewer target tokens per step. The snapshot (seed 42 evaluation, EMA weights,
temperature 0.7 for both posteriors, eight samples per model) produced:

| Content process | Mean block length | Block length range | Truncated sequences | Valid-word rate | Valid-word rate, length >= 5 |
| --- | --- | --- | --- | --- | --- |
| Masked | 32.31 | 32–37 | 0/8 | 59.5% | 4.9% |
| Gaussian | 32.25 | 31–35 | 0/8 | 56.8% | 3.4% |

Text remained mostly garbled at that checkpoint. Word validity alone is not a
quality metric: the whole-sequence simplex pilot achieved a high valid-word rate
while repeatedly generating words such as “nine.” Inspect sample text, diversity,
longer words, special tokens, lengths and truncation together. The composite
insertion objective is not a reported likelihood/perplexity and should not be
compared numerically with another method's token-only loss.

Reproduce the experiment recipes with:

```bash
uv run python main.py +experiment=text8_insertion_gaussian_sc_22m
uv run python main.py +experiment=text8_insertion_masked_sc_22m
uv run python main.py +experiment=text8_insertion_simplex_sc_22m
uv run python main.py +experiment=text8_insertion_spherical_sc_22m
uv run python main.py +experiment=text8_block_insertion_masked_sc_22m
uv run python main.py +experiment=text8_block_insertion_gaussian_sc_22m
```

The recipes specify 5,000 optimizer steps, not a claim of convergence. Gaussian
uses adaptive token-only time feedback; simplex and spherical include insertion
losses; masked uses uniform time sampling. Block Gaussian also uses adaptive
token-only feedback. The adaptive samplers warm up for 1,000 steps. Callback
sampling uses 64 examples for whole-sequence runs and 8 for block runs.
Check `config.yaml` for the resolved settings of a particular checkpoint.

Local evidence is saved under `core/outputs/<experiment>/`: resolved config,
`metrics/version_*/metrics.csv`, `checkpoints/`, and
`task_callback/step_00001000.json` (or later steps). Evaluations preserve
`generated_lengths`, `insertion_truncated`, and, for blocks, `block_lengths`.
These generated artifacts are ignored by Git and are not bundled with a clean
clone. W&B is disabled in these presets unless explicitly configured.

The corrected parallel-block Gaussian run subsequently completed 5,000 steps.
Its final 32-sample test evaluation (EMA, seed 23, temperature 0.7) had 75% valid
words, 37.4% valid words of length at least five, mean block length 32.35
(range 31–37), and no truncated sequences. The text still contains malformed
words, misplaced EOS tokens, and repetition. The small callback evaluation
peaked at step 4,000 rather than improving monotonically. See the
[portable results and all 32 samples](../../results/text8-insertion/parallel-block-gaussian-5k.json).

## Adaptive training-time feedback

Set `time_sampler=loss_derivative time_sampler.adaptive=true
variable_length.time_sampling_include_aux=true` to fit the training-time sampler
to the combined token and insertion objective. The default is `false`, preserving
surviving-token-only feedback. With the option enabled, each example contributes
its surviving-token loss plus weighted insertion-count and insertion-content
losses, divided by its original target-token count. Fully deleted examples still
contribute; examples without target tokens are excluded. The insertion objective's
`count_weight` and `content_weight` also apply to this feedback. Custom insertion
objectives must support `reduction="none"`, returning per-example sums.

This changes only training-time sampler feedback, not the optimization objective,
per-view adaptive noise-schedule feedback, or the generation timestep grid.
Enabling it requires a new run or an explicit resumed-run configuration change;
it does not retroactively change a running process.

### Insertion-count calibration

`InsertionSampler.count_scale` multiplies each gap's Poisson intensity before
sampling counts. Its default is 1; values below/above 1 decrease/increase birth
rates. It does not change the relative gap preferences at a given step or the
birth-state law. Protected gaps remain excluded and the length limit still applies.

`examples/tune_insertion_counts.py SOURCE OUTPUT --project PROJECT` creates a
fixed-checkpoint manifest from completed insertion cases. It searches count scale
in `[0.25, 4]` on a log scale, holding each case's previously selected token
settings and final decoding fixed. The original validation/final partitions and
NFE budget are retained. Submit this manifest with the normal nested-tuning CLI.

For mask and uniform births, the gap-token posterior is not used to initialize
new positions: births are masks or uniform noise. Consequently, gap-token
inference temperature is not searched for those laws.
