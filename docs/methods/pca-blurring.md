# Sequence-adaptive feature PCA blurring

Implemented as `core.processes.pca_blurring.PCABlurringProcess`, composed through
`configs/hybrid_view/pca_blurring.yaml` and the existing forward-renoising sampler.
The initial runnable task recipe is `+experiment=sudoku_pca_blurring`.
This is feature-space corruption, with no positional DCT mixing.

## Forward marginal and mean treatment

For the positions selected by `corruption_mask`, compute their centered feature
covariance C = V diag(lambda) V^T in FP32. Ignore fixed prompt/padding positions
when estimating this covariance. Use population covariance (divide by active
count); empty or singleton active sets produce a finite zero covariance.
Stop gradients through both eigenvectors and eigenvalues/rates. Gradients still
flow into the learnable clean embedding values being attenuated.

Let m = mean(lambda) + ridge and r_i = 1 + strength * lambda_i/(lambda_i + m).
The defaults are strength=3 and ridge=1e-6. Set

    a_i(t) = (1-t)^r_i
    sigma(t) = t
    X_t = X_0 V diag(a_i(t)) V^T + t epsilon.

The existing LinearInterpolationSchedule provides alpha=1-t and sigma=t;
the process raises alpha to r_i. Every r_i lies in [1, 1+strength]. Thus all
features disappear at t=1, including covariance-null directions, while t=0
returns clean input exactly. Apply the operator to **uncentered X_0**, without
adding its mean back afterward. A constant sequence therefore decays as
(1-t)X_0, rather than preserving its mean. Only generated positions become iid
N(0,I) at t=1; prompt positions remain fixed by design.

Rates depend smoothly on eigenvalues rather than ordinal rank. Equal eigenvalues
receive equal rates, so rotating a degenerate eigenspace does not change the
operator. Eigenvector signs cancel. Eigenvalue crossings do not create the hard
mode swaps of rank-based attenuation. Recomputing a detached basis is an explicit
surrogate-gradient choice, not differentiation through the full forward map.

## Consistency and limits of sampling

Conditioned on the SAME X_0 and its fixed V, for s<t<1 define

    R_i = a_i(t)/a_i(s)
    Q_i = sigma(t)^2 - R_i^2 sigma(s)^2.

Use a Gaussian transition with mean X_s V diag(R) V^T and covariance
V diag(Q) V^T independently at each position. Q is nonnegative because a decreases
and sigma increases. These transitions compose to reproduce the specified
marginals. `forward_transition` implements this conditional construction and
checks negative variances; a pure-noise source endpoint is deliberately rejected.
The extended state (X_t, X_0) is Markov, but X_t alone is not established to be
Markov after marginalizing out X_0. Independence of the terminal distribution
alone does NOT justify a standard Gaussian reverse update.

The shipped sampler is the existing ForwardResamplingSampler:

1. Predict clean-token probabilities from the current noisy state.
2. Sample one clean token sequence (or use the existing greedy option).
3. Encode that sequence, recompute its PCA, and draw the target-time forward kernel.
4. Preserve prompts and continue; terminal token decoding is greedy by default.

This is an **approximate denoise/renoise chain**, not the conditional reverse
bridge or an exact marginal reverse process. Each PCA draw is exact for the
predicted clean sequence's forward marginal. `update: null` intentionally avoids
advertising a native reverse rule; select forward_resampling, not hybrid/native.
PCA of current noisy states is not used: its covariance approaches isotropy at
the noise endpoint and cannot identify the clean basis. A learned marginal
score/velocity could be investigated separately; it is not implemented here.

The initial scope is fixed-length whole-sequence Sudoku generation. Do not reuse
this process in parallel block training expecting block-local isolation: the
current covariance aggregates all active positions in an example.

## What this prevents—and what it does not

Rotation equivariance removes a fixed globally privileged feature axis. Moving
energy into a new dominant principal direction increases its attenuation rate.
This does not prove prevention of information hiding: the model can redistribute
energy, encode useful information in small-variance modes, or exploit the mean.
Those channels still decay to zero, but can remain more informative at finite
times. Stop-gradient PCA does not eliminate these possibilities. Sudoku has only
12 embedding codewords, so centered covariance is intrinsically low rank; full
128-dimensional PCA will contain many null directions.

## Sudoku recipe and verification

The config uses the existing cached 48k/2k hard-Sudoku split, trainable unit
embeddings of dimension 128, the eight-layer 512-width backbone, CE, no SC,
20k-step BF16 training, batch 256, 2,500 warmup, and the existing CDCD sampler
with the physical-time endpoint clamp used by the completed CDCD campaigns.
CDCD alters the continuous view's time mapping in training and inference.
The production sampler config is 127 updates plus terminal prediction = 128 NFE.
A one-seed production Sudoku run was launched on 2026-11-01, on GPU 6 via Ray.
Plan/source: `/home/patrick/.cache/duo/plans/sudoku-pca-blurring-cdcd-20k-20261101/`.
Results/logs/checkpoints: `/home/patrick/.cache/duo/nested-tuning/sudoku-pca-blurring-cdcd-20k-20261101/`.
`launch.json` records the command, frozen source and environment. W&B is disabled;
logging is local only because external upload approval was rejected. Training uses
the cached Sudoku dataset; inference tuning copies the established renoising
validation/test partitions and sampled/greedy policies at 128 NFE. Initial driver
PID 193684 is historical; verify live state from the launch artifacts.

Tests in `core/tests/test_pca_blurring.py` cover exact endpoints, mean attenuation,
prompt preservation, active-mask covariance isolation, rotation equivariance,
degenerate and near-degenerate spectra, near-noise continuity, Gaussian isotropic
limit, finite gradients, conditional transition moment/composition checks, and
Hydra-configured Sudoku loss/backward/generation with CDCD fitting.
Existing CDCD tests are preserved, including distributed updates and checkpoint
migration. An additional regression covers floating-point endpoint clamping.

Reproducible isolated validation (select an idle GPU explicitly):

    CUDA_VISIBLE_DEVICES=6 UV_PROJECT_ENVIRONMENT=/home/patrick/.codex/worktrees/54f2/duo/.venv \
      uv run --no-sync python -m scripts.validation.pca_blurring \
      --output results/pca-blurring/validation.json

This only reads the existing dataset cache and writes local reports; it neither
submits jobs nor logs to W&B. The smoke uses a small two-layer backbone, 128-D
embeddings, 20 optimizer updates on eight real examples, BF16, and 8-NFE generation.
It is an execution/gradient test, not an accuracy or convergence claim.

Measured on an RTX A6000, B=16, L=512, D=128, FP32 PCA:

| Operation | Mean ms | Extra peak allocated MiB |
| --- | ---: | ---: |
| Covariance + full eigendecomposition | 28.49 | 12.01 |
| Randomized rank-32 PCA, 2 iterations | 20.92 | 7.09 |
| Full corruption including full PCA | 28.36 | 17.02 |

These are 20-iteration microbenchmarks after warmup; small timing differences are
noise. Truncation is a timing comparison only: it does not implement the full
spectral kernel. Updating a basis every four calls would amortize the measured
basis cost to ~7.12 ms/call, but reusing a basis after changing predicted clean
sequences changes the kernel. It is not enabled. Timings exclude the denoiser and
are not full-model speedups. Detailed report: `results/pca-blurring/validation.json`.

## Comparison protocol still to run

Use matched architecture, embeddings, optimizer/training budget, seed, CDCD time
mapping and 128-NFE renoising for all models. Strength=0 in this same process is an
exact isotropic Gaussian path control (up to roundoff), retaining identical
representation/scalar schedules. Existing fixed-feature/DCT blur should be a
separate baseline; its schedule endpoints differ and must be reported explicitly.
Use feature-only and positional-only blur ablations before combining them.

Record held-out Sudoku solve accuracy, sampler execution success/finite-output
rate, spectral effective rank/top-mode energy/mean energy over training, examples
per second, wall time and peak GPU memory. Any programming-task code-execution
metric is not applicable to Sudoku. No matched performance claim or complete
training comparison has been made by the smoke test.
