# Spherical flow implementation

Reference: [Language Modeling with Hyperspherical Flows, v1](https://arxiv.org/html/2605.11125v1),
[released code](https://github.com/jdeschena/s-flm).

The Sudoku experiments retain `sphere_dit`, matching the paper's Table 1.
`sphere_arch` is an independent backbone option used for the paper's headline
GSM8K results.

## Corrections

- Stochastic top-k velocity draws from normalized probabilities. Top-k values
  exceeding the vocabulary size are capped. Probability top-k and nucleus
  filtering also apply to final decoding.
- `algo.slerp_precision=float64` enables double-precision manifold updates;
  `sampling.use_float64=true` preserves double-precision probabilities and
  promotes geometry as needed. States are cast back before the next denoiser call.
- SphereArch includes the learned pre-activation MLP scale and sqrt(d) factor,
  release-style initialization, dropout, projection axes, signed output scale,
  and appended optional time token. Both GELU and SwiGLU, with or without
  time-conditioned gates, are supported. Conditioning matrices are not projected.
- Coincident spherical endpoints have a zero logarithmic map, with a smooth
  exponential-map derivative at zero. Exactly antipodal endpoints have no unique
  shortest path; the geometry helper chooses a deterministic orthogonal direction.
- The adaptive Sudoku and TinyGSM SFM presets select `adaptive_coordinate: physical`.
  Loss observations retain their actual corruption coordinate across refits, and
  the inverse CDF preserves the truncation endpoints. This is an intentional
  numerical correction to the released legacy fitter. `legacy_time` remains
  available for reproducing earlier runs.

The sampler retains the release's NFE convention: N denoiser calls, N-1 manifold
updates, then decoding with the last prediction. It does not add an extra endpoint
call. Truncation remains explicitly configured; Sudoku's 0.093 value assumes
vocabulary size 12 and embedding dimension 512.

## Checkpoints and validation

Use fresh training runs for corrected SphereArch comparisons. Old SphereArch
checkpoints lack the learned MLP scale; strict resume is not architecture-compatible.
Do not resume a legacy adaptive buffer as a physical-coordinate buffer: preserve
its saved configuration for reproduction or start a fresh corrected run.

Regression coverage includes stochastic frequencies, probability filtering,
inference dtypes, prompt preservation, NFE counts, spherical singularities and
gradients, matrix projection axes, both MLP variants, repeated truncated schedule
refits, and real loss/backward/checkpoint/sampling paths for both backbones.
These checks validate implementation behavior, not reproduction of published
training accuracy.
