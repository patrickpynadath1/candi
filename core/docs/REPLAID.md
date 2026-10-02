# Continuous RePlaid

`+experiment=replaid` assembles continuous RePlaid entirely from core components.
`+experiment=replaid_sudoku_hard` selects the same method with Sudoku Hard data.
The default preset now ports the corrected `0303` implementation and the frozen
source of the successful `replaid-hard-50k-20260929-autocast-fix` run;
it does not claim reproduction of the paper's large language-model experiments.

```bash
python main.py +experiment=replaid_sudoku_hard
```

For a 2,000-step initial check, append `trainer.max_steps=2000` and
`checkpoint.every_n_steps=2001`. An existing dataset cache can be selected with
`data.cache_dir=/path/to/cache`. Disable the generation callback with `~callback`
if comparing training runs without intermediate sampling changing the random stream.

## Components and ownership

- `algorithms/replaid.py`: a Gaussian algorithm specialization that prepares time
  allocation and self-conditioning selection, routes gradient views, evaluates
  the objective, and supplies adaptation observations. It contains no reverse-step
  equations or sampling loop.
- `representations/embedding.py`: trainable, unit-normalized token embeddings,
  dimension 16 by default. Reads normalize the table; a post-optimizer hook also
  projects the actual parameter onto the unit sphere.
- `noise_schedules/learned_log_snr.py`: monotone negative-log-SNR gamma, its
  derivative, VP path coefficients, learned endpoints, and variance-gradient hooks.
  Schedule parameters and calculations are float64. Initial endpoints are -13.3
  and 5.0. The default is a 1,024-unit positive-weight tanh network with independent
  endpoints. The older softplus schedule remains available explicitly.
- `time_sampling/adaptive_allocation.py`: batch allocation between reconstruction
  and diffusion, stratified diffusion times, and checkpointed first/second-moment
  EMA buffers (momentum 0.99). Derived variance remains meaningful when the
  reconstruction sub-batch has one row. Statistics respect the loss mask. This policy receives explicit observations from the
  algorithm; it does not invoke the objective.
- `parameterizations/gaussian_prior.py`: adds the Gaussian output-prior correction
  to network logits before the posterior pathways. State, alpha, sigma and the
  embedding table are supplied explicitly. Training uses no temperature/filtering.
- `objectives/replaid.py`: tokenwise reconstruction CE, diffusion embedding MSE
  weighted by negative one-half SNR derivative, terminal prior KL, and batch
  allocation factors. Final aggregation is float64.
- `models/continuous_dit.py`: configurable direct input projection, additive
  projected time embedding, and AdaLN vocabulary output head. These choices are
  enabled by the RePlaid preset; other Gaussian presets retain their existing
  extra input projection and output architecture.
- `processes/gaussian.py`: Gaussian corruption and the reverse conditional mean
  and variance. Coefficients are cast to the state dtype for corruption.
- `samplers/gaussian_ddpm.py`: reverse-time loop with configurable prompt policy,
  fresh-bootstrap or previous-feedback conditioning, final-transition noise, and
  final categorical argmax. RePlaid uses fresh bootstrap and final-step noise: with
  conditioning enabled, NFE is `2 * (steps + 1)` (362 for 180 transitions).
  The Sudoku preset keeps prompts clean during both training and sampling.

The Gaussian DDPM sampler offers `prior_mode=standard_normal` (RePlaid's reference
prior) and `prior_mode=schedule` (scale the prior by terminal schedule sigma, useful
for VE schedules). Its reverse kernel supports positive, monotone-SNR Gaussian
paths and is independent of the embedding/one-hot representation.

## Training and gradient routes

For each microbatch, allocation first reserves reconstruction rows at time zero;
remaining rows use one random offset plus a uniform grid modulo one. The split is
rounded from the ratio of reconstruction and diffusion standard-deviation EMAs,
with at least one row of each type. Singleton batches use two draws, one for each
estimator, while reported token counts retain the original batch size.

The RePlaid preset uses expected-embedding self-conditioning and fixed-count
selection: exactly `ceil(0.25 * batch_size)` randomly chosen examples during
training, all examples during validation. Selected examples receive a detached
bootstrap prediction of the same corrupted state and time. Their main prediction
still updates the denoiser, but their schedule/input/target embedding paths are
explicitly detached. The embedding table used for output-prior logits and for the
predicted expected embedding is also detached on these rows; probabilities retain
their denoiser gradients.

The terminal prior KL has its own ordinary-gradient path, including on selected
rows. Interior schedule shape receives the original variance-minimizing gradient
multiplier; endpoints receive ordinary loss gradients. Reconstruction and selected
rows contribute zero to the shape-adaptation multiplier.

The general self-conditioning defaults remain probability 0.9 and zero-initialized
feedback injection. RePlaid explicitly selects probability 0.25 and the reference's
ordinary initialization for its mean-feedback projection. Hidden feedback remains
available as an architectural alternative but is not the reference preset.

The unconditional RePlaid preset uses `algorithm.corrupt_all=true` and noisy
prompt inpainting. The corrected Sudoku preset selects `corrupt_all=false` and
`sampler.prompt_mode=clean`, retaining clean puzzle-prefix embeddings during both
training and inference. Bootstrap mean feedback also retains those known embeddings.
Both schedule adaptation and batch-allocation statistics use masked solution means.

## Optimizer and checkpoint state

The trainer supports optional named-parameter optimizer groups generically. The
RePlaid preset sets the embedding and schedule-shape learning rate to 0.01 with no
weight decay; endpoints use the same learning rate with weight decay 0.1. Backbone
parameters use the normal optimizer defaults and the shared warmup schedule.

The update order deliberately matches the original: optimizer update, EMA update,
then representation projection. EMA therefore observes the pre-projection update;
codebook reads still normalize rows during evaluation.

Learned schedule parameters, allocation EMA buffers, all model and representation
parameters, optimizer state, EMA state, and random state use the normal core
checkpoint lifecycle. Per-forward schedule hooks and sampling feedback are local
transient state, never reused across batches or saved in checkpoints.

## Verification and historical comparison limits

`core/tests/test_replaid.py` covers analytical derivatives and VP identity,
shape-versus-endpoint gradients, selected-row gradient ownership, Gaussian
output-prior equivalence, hand-computed masked objective reductions, reverse
posterior limits, exact fixed-count selection, optimizer groups, unit projection,
EMA ordering, state restoration, and CPU/CUDA sampling with prompt preservation.

The historical one-time audit `core/outputs/replaid_comparison/test_reference_parity.py` reads
the old reference files and compares schedule values/derivatives/gradients and
mapped-weight model predictions. It is deliberately outside the permanent core
test suite so removal of legacy files does not break core tests or imports.

The recorded, superseded 2,000-step training comparisons share method hyperparameters, dataset, model size,
optimizer budget and evaluation settings. They are not bitwise trajectories:
component construction changes random initialization order, core uses native
attention, dataloader bookkeeping differs, and fixed-count bootstrap rows are
processed in batch order rather than permutation order. The reverse kernel uses
sigma directly instead of subtracting rounded FP32 alpha squared from one, avoiding
cancellation near the clean endpoint while retaining the same mathematical kernel.

That older mixed-precision comparison had one numerical-policy difference:
core keeps input/time projections, output-prior dot products, and expected-embedding
conversion in FP32. The old implementation's ambient BF16 autocast also affects
those operations despite their `.float()` operands. Transformer blocks and output
heads use BF16 in both runs. We retain core's FP32 policy and report this difference;
the mapped-weight, same-RNG FP32 audit verifies the underlying objective and gradient
mathematics, including all denoiser, embedding, and schedule parameters.

## Initial Sudoku Hard comparison

Both implementations trained for 2,000 optimizer steps, with the same 48,000/2,000
Sudoku Hard data split, seed 1, 28.6M-parameter architecture, effective batch 256,
optimizer rates, 2,500-step warmup, and reference self-conditioning policy. Evaluation
used all 2,000 held-out puzzles, seed 1, batch 64 and 180 DDPM steps plus the final
prediction (181 model calls). No temperature/filter tuning was applied.

| Implementation | Weights | Solved | Cell accuracy |
| --- | --- | --- | --- |
| Original RePlaid | EMA | 0/2,000 | 32.0051% |
| Core RePlaid | EMA | 0/2,000 | 33.7635% |
| Original RePlaid | Final/raw | 0/2,000 | 29.7567% |
| Core RePlaid | Final/raw | 0/2,000 | 32.4410% |

These results verify execution and show similarly poor solving quality at this
budget. They do not establish successful Sudoku reproduction. Raw-weight evaluation
also fails to solve puzzles, so EMA alone does not explain the poor results.

Core's final logged training loss was 0.707660. Its held-out EMA objective estimate
was 0.695034: prior 0.0105705, reconstruction 0.000118246 and diffusion 0.684345.
These held-out components are not the final training microbatch's components.
The EMA gamma endpoints were [-6.40366, 3.97688], versus raw [-5.32792, 3.99188].

A read-only checkpoint inspection found both implementations at the minimum one
reconstruction example per microbatch, with reconstruction standard-deviation EMA
approximately 1.475e-35. The reference's population standard deviation is exactly
zero for a one-example reconstruction group. Both also learned closely spaced
Sudoku digit embeddings. These observations are possible follow-up leads, not an
established cause of the generation results; the port preserves the reference
allocation and embedding mathematics.

The generated solution's clue-consistency metric is distinct from prompt
preservation: the sampler preserves supplied prompt tokens, but a generated solution
can still disagree with the clues. Configs, logs and per-example core EMA records
are saved under `core/outputs/replaid_comparison/core/`.

## Corrected-reference validation

The corrected preset enables unit-embedding second-moment input scaling:
`z / sqrt(alpha² / d + sigma²)` and expected-embedding feedback scaled by `sqrt(d)`.
No-grad bootstrap calls and the continuous transformer use autocast without cached
weight casts, preventing detached casts from contaminating the main pass.
Schedule adaptation and allocation statistics both exclude loss-masked positions.

Three one-time audits in `outputs/replaid_comparison/test_corrected_reference.py`
compare the schedule, mapped model outputs, and full loss/parameter gradients to
the successful run's frozen source. These audits read that source only for testing;
production core has no dependency on it. Permanent regression tests cover singleton
statistics, masked updates, fresh-bootstrap NFE, clean prompts, last-step noise,
and real CUDA updates to the output head and attention weights.

The earlier 2,000-step comparison recorded in this document used the superseded
softplus-schedule implementation. Its metrics do not evaluate these corrections.
The successful reference used 50,000 steps and 200k/10k data, whereas this Sudoku
preset retains core's 48k/2k dataset and default training budget. Do not label it
an exact full-budget reproduction. Existing old-schedule checkpoints require their
saved old configs; the new schedule and running-moment state are not interchangeable.
