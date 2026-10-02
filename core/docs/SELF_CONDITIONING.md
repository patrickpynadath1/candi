# Optional self-conditioning

The integration supports MDLM (ancestral/entropy-bounded sampling), Gaussian
diffusion with one-hot or embedding representations (Euler/Heun), and spherical
flow matching (spherical flow sampler).
Enable the feature through the shared config; no MDLM-specific feedback logic is
needed:

```bash
python main.py self_conditioning.enabled=true
python main.py self_conditioning.enabled=true self_conditioning.source=hidden
```

`self_conditioning.source=posterior_mean` is the default. It uses the expected
**input token embedding** for MDLM and the representation embedding for Gaussian
and spherical diffusion, computed in FP32 from the predictor's unshaped,
mathematically constrained posterior. This includes observed-token constraints
and softcapping but excludes temperature, top-k, top-p, and greedy commitment.
For one-hot representations, the expected embedding is the probability vector.
For spherical embeddings, the expectation is **not normalized back onto the sphere**;
its norm carries uncertainty. `hidden` uses the final contextual transformer state
before the output head. The model applies an affine LayerNorm in FP32 and adds
the result directly to its input, without a dense feedback projection. The
LayerNorm scale and bias both start at zero. This applies to training bootstraps
and inference feedback. Missing hidden feedback is a zero tensor passed through
the same LayerNorm; after training, its learned bias can contribute.
Both sources are detached before feeding back into the network.

Expected-embedding feedback uses a zero-initialized, bias-free projection to
match model dimensions. Both adapters therefore contribute exactly zero initially.
The existing Gaussian embedding/input projections remain in place. Adapters are
constructed before optimizer and EMA creation and included in checkpoints.
Disabled models retain their original state-dict keys and initialization RNG behavior.
Earlier enabled self-conditioning checkpoints use a different architecture and
parameter names and require their matching historical code.

The benchmark results below predate this zero-initialized LayerNorm architecture
and the new 0.9 default; they are historical comparisons, not its quality results.

## Training and validation

`self_conditioning.probability` defaults to 0.9. Independently selected examples
receive a no-gradient bootstrap forward on the **same corrupted input and time**
as their main prediction. The remaining examples receive zero feedback. The
bootstrap has no incoming self-conditioning. It does not compute a training
objective or submit observations to adaptive schedules.

`self_conditioning.evaluation_probability` defaults to 1.0 for validation loss:
all examples receive a bootstrap prediction, with model dropout disabled. This
is separate from the training probability. Set either probability to zero to
omit bootstrapping on its respective loss path. Neither setting controls
inference trajectory reuse. The original RePlaid training recipe selects exactly
`ceil(0.25 * batch_size)` examples, whereas this generic policy selects each example
independently; a RePlaid-specific selection policy can be added separately.

This is generic detached prediction feedback. It does not implement RePlaid's
additional embedding/schedule gradient rules, or claim to reproduce its
fixed-size selection of self-conditioned examples.

## Generation

Each sampler creates a local `PredictionSession` for its trajectory. The first
prediction has no feedback; subsequent predictions receive the detached signal
from the previous network evaluation. Generating the feedback does not require
another model call, so NFE is unchanged. Euler commits feedback after its step.
Heun freezes incoming feedback across both stages and commits the last prediction
only after the step completes. A failed step discards pending feedback. A new session starts empty, including
after failed generation. No trajectory tensors are retained by the model,
algorithm, or shared sampler instance.

Adaptive Dormand–Prince sampling currently rejects enabled self-conditioning.
It needs accepted/rejected-step lifecycle handling and invalidation of cached
FSAL derivatives when the feedback changes. The guard avoids treating
history-dependent feedback as an ordinary adaptive ODE field.

Calling `algorithm.model_logits(...)` remains a stateless raw-logits endpoint.
Calling `algorithm.sampling_prediction(...)` without a session is also stateless.
Sampler integrations explicitly own session creation and reuse.

## Component boundaries and extension

The implementation lives under `core/modifiers/self_conditioning/`.
Configuration flags and checkpoint parameter names are unchanged by this
directory move.

- `core/models/prediction.py`: typed optional model features and the architectural
  conditioning adapters.
- `core/modifiers/self_conditioning/sources.py`: tensor conversions for feedback sources.
- `core/modifiers/self_conditioning/prediction.py`: bootstrap policy and prediction routing;
  its caller supplies the model, predictor, prepared inputs, and mean conversion callback.
- `core/algorithms/posterior.py`: generic posterior endpoints and model-input
  preparation interface. `MDLM` only prepares its ordinary inputs and computes
  its existing loss.
- `core/modifiers/self_conditioning/session.py`: sampler-owned prediction feedback state.

Gaussian and spherical algorithms supply prepared model inputs and a posterior-mean
conversion callback through the shared base. They do not own feedback branches.
The spherical algorithm retains its original FP32 logit promotion. Continuous
samplers route both velocity and final decoding predictions through their local
session; a session's `posterior_mean` explicitly uses its own prediction endpoint.

Posterior-mean conditioning requires predictor capabilities for unshaped
clean-token posteriors and probabilities. Construction fails with an informative
message naming the predictor and source if those capabilities are missing; it
never interprets arbitrary scores as probabilities. Hidden-state feedback does
not require probability capabilities. Uniform/DUO/SEDD still need their endpoint
and sampler integrations; their unsupported configuration is rejected rather
than silently ignored. Autoregressive methods additionally need causal position
alignment and cache semantics. These are integration requirements, not a claim
that the methods are mathematically incompatible with self-conditioning.

## Validation

`core/tests/test_self_conditioning.py` covers both sources, detached bootstrap and
main-pass gradients, the same corrupted state/time across passes, probability
selection, validation behavior, unshaped feedback, one adaptive observation per
loss, ephemeral sessions and failed-step rollback, optimizer/EMA/checkpoint
restoration, flag-only MDLM integration, disabled checkpoint/RNG compatibility,
Gaussian representation dimensions, unnormalized spherical expectations, Heun
stage freezing, the adaptive ODE guard, and CUDA bfloat16 training/generation.

Two real MDLM Sudoku Hard runs reached 20 optimizer steps in CUDA bfloat16 with
`hidden_size=32`, one transformer block, and no AdaLN. These are functional smoke
checks, not quality benchmarks. Both checkpoints were reloaded and evaluated on
two held-out puzzles with ancestral and entropy-bounded samplers. They completed
without errors and preserved prompts; neither solved a puzzle at this scale.
The posterior-mean run used 256 training/16 validation puzzles and batch 16;
the hidden-state run used the existing 8 training/2 validation smoke cache and
batch 4. Their validation losses were 2.37885 and 2.11335 respectively; those
values are not a controlled comparison because the datasets and batches differ.
Ancestral used 16 reverse steps plus final denoising (17 NFE); EB took 89 NFE.

Resolved configs, CSV metrics, checkpoints, and generation results are under
`core/outputs/self_conditioning_cuda_20261002/`, with `{source}_train`,
`{source}_ancestral`, and `{source}_entropy_bounded` directories.

## Full-size Sudoku Hard check: October 2, 2026

Three runs used 2,000 optimizer steps, seed 1, 48,000 training puzzles, the
baseline 512-wide eight-block DiT, batch size 256, BF16, and the original
2,500-step warmup. Hidden-feedback runs selected bootstrap examples independently
with probability 0.5. Evaluations used EMA weights, seed 1, all 2,000 held-out
puzzles, and unchanged sampling settings (temperature 1, no top-k/top-p filtering).
These are single-seed early-training results, not final-convergence comparisons.

| Training configuration | Final logged training loss | Ancestral solved | EB solved | Ancestral cell accuracy | EB cell accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| No self-conditioning | 0.3159 | 0/2,000 (0%) | 394/2,000 (19.70%) | 69.86% | 83.62% |
| Original unnormalized hidden feedback | 0.6509 | 0/2,000 (0%) | 0/2,000 (0%) | 10.84% | 40.92% |
| L2-normalized hidden feedback | 0.3392 | 8/2,000 (0.40%) | 57/2,000 (2.85%) | 73.58% | 77.95% |

Training losses are the final logged batch values, not held-out losses. Ancestral
uses 180 steps plus final greedy denoising (181 NFE). EB uses bound 0.1 and greedy
commitments; mean NFE is 22.762 without conditioning, 68.6995 with unnormalized
feedback, and 14.416 with normalized feedback.

The original implementation fed back the unnormalized residual stream. At the
2,000-step checkpoint, input token embedding norms were approximately 0.60, not
unit-normalized. In EMA EB diagnostics, input embedding RMS was about 0.026 while
the first projected feedback RMS was 23.94 (917 times larger). Repeated reuse
grew projected feedback RMS to 1.42e21. FP64 norm diagnostics confirmed the growth;
FP32 squared-norm diagnostics overflowed. This motivated per-token L2 normalization
in the shared hidden source, applied consistently during training and inference.
The corrected run kept projected feedback RMS around 0.79–0.89 times input RMS.
Expected-embedding feedback remains unchanged.

Normalization recovered training loss and improved generation, but did not make
this configuration competitive with the no-conditioning control under EB at this
budget. Additional diagnostic EB evaluations on the same fixed 128-puzzle subset
with the normalized checkpoint's EMA weights gave:

| Inference feedback | Solved | Mean network evaluations |
| --- | ---: | ---: |
| Reuse preceding hidden state (production behavior) | 2/128 | 14.625 |
| Disable feedback on the same trained model | 7/128 | 23.25 |
| Fresh bootstrap at the current state for every prediction | 10/128 | 33.984 |
| Independently trained no-conditioning control | 29/128 | 22.695 |

The fresh-bootstrap diagnostic matches training's two-pass input construction
and counts both calls. Its improvement suggests a training/inference feedback
mismatch contributes to the remaining gap; it does not establish that this is
the sole cause. It is an experiment-only adapter, not a change to sampler behavior.
The same ordering appeared with raw weights (9, 16, and 19 solved respectively
for the three inference variants on the normalized checkpoint).

Artifacts are under `core/outputs/`:

- `mdlm_no_self_conditioning_sudoku_hard_control/`
- `mdlm_hidden_self_conditioning_sudoku_hard/` (original unnormalized run)
- `mdlm_normalized_hidden_self_conditioning_sudoku_hard/`

Each contains the resolved configuration, CSV training history, final checkpoint,
and full per-puzzle evaluation records. The normalized run contains
`comparison.json`, `feedback_diagnostics.json`, and
`fresh_bootstrap_diagnostics.json`. The original run contains the diagnostic
script; use `--unnormalized` to reproduce its old behavior, or `--run-dir` to select
the normalized run. The historical original MDLM run reached 0.3035 training loss
and 49/128 EB solves at step 2,000; it is a historical reference, distinct from the
fresh core control. Its local training history was also retained for inspection.

Normalization validation: 276 core tests passed (16 CUDA-dependent skips in the
sandbox), and all 39 self-conditioning tests passed separately on CUDA, including
BF16 and both hidden/expected-embedding sources.
