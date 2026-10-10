# Hybrid modifier

A hybrid maintains independent noisy views of the **same clean tokens**. Each view is a passive composition of a process, noise schedule, representation, local time map, and sampler update. The modifier sums the representations' model embeddings and runs one shared transformer. No mapping between the noisy states is required.

The responsibilities are:

- **Process:** corrupt a clean representation and produce a prior state; own transition mathematics.
- **Representation:** encode clean tokens, map noisy states into transformer embeddings, and map the shared token posterior into a clean-state mean where needed.
- **Schedule and time map:** convert global time into local corruption coefficients.
- **Sampler update:** advance its own state using the process and the shared token posterior.
- **Hybrid modifier:** orchestrate corruption, embedding fusion, prediction, and the training objective.

`DiffusionView` stores these components and registers their trainable parameters/buffers. It has no process-specific execution logic. There are no discrete/Gaussian/simplex branch adapter classes.

## Configuring combinations

```bash
# Gaussian embedding + masked corruption (default views)
uv run python main.py +experiment=hybrid

# Spherical + uniform
uv run python main.py +experiment=hybrid \
  hybrid_view@model.views.first=spherical \
  hybrid_view@model.views.second=uniform \
  model.time_conditioning=negative_log_keep

# Simplex + Gaussian
uv run python main.py +experiment=hybrid \
  hybrid_view@model.views.first=simplex \
  hybrid_view@model.views.second=gaussian
```

Available view presets: `masked`, `uniform`, `gaussian`, `simplex`, `spherical`, and `blurring`. These are composable component recipes, not claims of paper-equivalent training results. Change a view's schedule, representation, time map, or update directly in configuration. The default fusion sums model embeddings; each representation has a configurable scale. All views must agree on token vocabulary, sequence layout, and transformer hidden size.

## Existing assemblies

```bash
uv run python main.py +experiment=candi_hybrid
uv run python main.py +experiment=uniform_simplex_hybrid
```

| Preset | Branches | Sampler | Special policy |
|---|---|---|---|
| `candi_hybrid` | Absorbing masking + rank-calibrated one-hot Gaussian | CANDI entropy bounded | Synchronize committed tokens |
| `uniform_simplex_hybrid` | Posterior uniform + simplex | Fixed-step hybrid | Independent states |

CANDI retains the reference input fusion and frozen-posterior Gaussian update. Its explicit `synchronize_committed=true` policy sets the continuous view of committed mask tokens to the same clean token before the next prediction. Other presets leave this off. The masking branch also fixes revealed-token posterior outputs.

Training masking intentionally corrects a historical bug: selected positions always become masks. The old path redrew uniform tokens and inferred revelation from equality with the clean token, accidentally retaining some selected positions. Thus importing an old checkpoint can validate inference, but cannot establish equal training distributions.

## Composition contracts

Every branch must consume a clean-token posterior. Uniform branches therefore use `PosteriorUniformProcess`, not the plug-in uniform transition. The framework rejects plug-in branches; see [discrete diffusion](discrete-diffusion.md) for why the mathematics differs.

The current objective is token CE. RePlaid hybrid likelihood objectives are not implemented. A new process implements `corrupt_state(clean, coefficients, corruption_mask)` and `prior_state(clean, coefficients)`; its representation implements `encode`, `clean_state`, `to_model_embedding`, and `clean_embedding`. Its update is a callable in the sampler layer. This allows composition without editing the modifier's training/model code.

Processes can expose `observed_tokens(state)` for exact clean observations, as absorbing corruption does. The modifier applies that constraint regardless of view order. Uniform observations are not treated as clean commitments. Gaussian schedule/representation compatibility is checked at construction.

View-local time maps tie each physical schedule to shared global time. Adaptive schedules receive masked training-loss observations at their local times and their buffers are checkpointed. Sequence-aware schedules receive the corruption mask and sequence layout; blurring therefore respects block boundaries without blurring-specific logic in the hybrid algorithm.

## Other modifiers

[Self-conditioning](../modifiers/self-conditioning.md) retains **one** feedback vector for the fused model, not one per branch. Posterior-mean feedback uses the fused clean embedding. Intermediate token feedback runs once in the shared backbone. [Block diffusion](../modifiers/block-diffusion.md) applies to the complete hybrid trajectory, so all views refer to the same active block and observed prefix.

## Historical checkpoint evaluation

```bash
uv run python main.py +experiment=candi_hybrid mode=evaluate \
  evaluation.legacy=candi evaluation.checkpoint=/path/to/last.ckpt \
  evaluation.max_examples=2000 loader.eval_batch_size=64
```

The importer maps the shared embedding and parameters, validates shapes, and reconstructs EMA ordering without duplicate registrations. `evaluation.use_ema=false` selects final weights.

The imported 20,000-step Sudoku Hard CANDI checkpoint solved 1,566/2,000 puzzles (78.30%) at mean 3.503 NFE, compared with its historical 1,560/2,000 (78.00%) and 3.493 NFE. These are inference results with old weights, not corrected-mask retraining or bitwise parity.

A separate fresh 2,000-step standard-masking run solved 746/2,000 (37.3%), with 89.74% token accuracy and mean 9.1495 NFE. A fresh 2,000-step uniform-plus-simplex run solved 237/2,000 (11.85%) with 82.87% token accuracy and 181 NFE. These short checks demonstrate trainability and generation; they are not converged benchmarks. See [recorded results](../validation/reproduction-results.md).

Sources: [modifier](../../core/modifiers/hybrid/), [representations](../../core/representations/noisy_inputs.py), [updates](../../core/samplers/hybrid_updates.py), [view configurations](../../configs/hybrid_view/).

## EB for masked hybrids

Use `sampler=masked_hybrid_entropy_bounded` for a hybrid with an absorbing view,
including Gaussian embedding + masked corruption. This sampler uses the shared
global time grid and each view's configured schedule and update, independently of
view order. EB selects tokens using cumulative entropy excluding the largest
entropy in the selected prefix (default bound 0.1). Selected tokens are committed
clean in every view; unselected absorbing positions stay masked. Other views
advance with their configured updates and zero supplied transition noise.
The default commitment is greedy, with a maximum of 180 prediction steps.

This is a different sampler from `hybrid_entropy_bounded`, which retains the
historical CANDI-specific time conditioning and view layout. Neither changes the
checkpoint's training corruption. EB may finish before traversing the full time
grid; exhausting the grid without committing every token raises an error.

## Comparing frozen and revisable EB

`core.samplers.parallel_entropy_bounded.ParallelEntropyBoundedSampler` extends
frozen EB to posterior-based continuous-only and uniform hybrid assemblies.
`freeze=true` retains cumulative clean commitments in every view. With
`freeze=false`, all generated positions are reconsidered each step. A released
continuous commitment is forward-corrupted from its prior committed token at the
destination schedule coefficients; already noisy positions use the configured
reverse update. Uniform states use their posterior reverse transition. Prompts
remain fixed, and a sequence ends early only when all generated positions are
selected together. Absorbing views reject `freeze=false`, because revealed tokens
are exact observations under their posterior.

If revisable sampling reaches the step limit, unfinished positions receive an
endpoint prediction and metadata marks the sequence as not entropy-complete.
Frozen EB on continuous-only models is an inference experiment, not a
paper-native sampler. The historical CANDI sampler remains separate.

`examples/evaluate_adaptive_sweep.py` records the 24-run adaptive Sudoku sweep's
regular evaluations and evaluates both EB bounds (0.1 and 0.01), with frozen and
revisable variants for uniform hybrids. Its output includes per-puzzle JSON,
checkpoint/config provenance, and a compiled CSV/Markdown comparison.

## Variable-length views

The [insertion modifier](../modifiers/variable-length.md) applies one shared
sequence edit to every view. New continuous positions are forward-corrupted from
a gap clean-content estimate at the current global noise level; absorbing views
receive aligned masks. The same contract supports a single content view; a
masked branch is not required. Insertion uses one global time for structure and
per-view time maps/schedules for content, rather than a separate age clock for
new tokens.

The modifier can also be wrapped in [block diffusion](../modifiers/block-diffusion.md).
All views share active-block boundaries, while completed blocks are frozen clean
context. See the insertion reference for birth policies, self-conditioning,
adaptive feedback, length semantics and approximation limits.
