# Spherical flow matching

`+experiment=spherical_flow_matching` selects a trainable unit embedding representation,
SphereDiT, spherical corruption, unconstrained posterior prediction, cross entropy,
and posterior-derived manifold integration. The algorithm only coordinates training
and prediction; the sampler owns generation and tangent velocity construction.

Sudoku Hard presets preserve the original 512-dimensional, 8-block/8-head SphereDiT,
batch size 256, learning rate 0.0003, 2500-step warmup, EMA 0.9999, and 20000-step
training budget:

- `+experiment=sudoku_spherical`: fixed linear spherical keep coordinate.
- `+experiment=sudoku_spherical_adaptive`: original repository adaptive Sudoku preset.
- `+experiment=sudoku_spherical_truncated_adaptive`: dimension-aware truncation,
  physical-coordinate adaptation, 12800 observations, refit every 50 steps after
  step 1000, EMA 0.9, and uniform mixture 0.001. Accumulation is one, matching the
  original truncated/paper Sudoku preset. Generation evaluates 128 held-out puzzles
  every 1000 steps with 180 model calls.

The adaptive component (`noise_schedule=spherical_adaptive`) defaults to
`coordinate: legacy_time`, matching original S-FLM sampled-time adaptation.
The named Sudoku adaptive experiments explicitly select `physical`, as did this
repository's Sudoku presets. These are intentional distinct settings.

Time increases toward noise. The schedule emits a keep coordinate; corruption
interpolates from the token embedding to independent uniform sphere noise by
`1 - keep`. Automatic truncation replaces the maximum keep coordinate with the
original dimension/vocabulary bound. Consequently even the lowest-time endpoint
is partially corrupted: it is not silently remapped to a fully clean endpoint.

The sampler preserves clean prompt embeddings at each step. `velocity: exact`
averages tangent vectors to token embeddings; `velocity: sample` chooses a token
from the posterior and uses its tangent vector. The final requested model call
produces tokens rather than taking another integration step, preserving original
NFE accounting. Posterior `commitment: sample` matches original ancestral final
sampling; use `parameterization.commitment=greedy` for deterministic final decoding.
Temperature/top-k/top-p remain posterior configuration. Geometry uses float64 by
default, and `sampler.use_float64` separately controls posterior probability math.

SphereArch's additional weight projections and legacy `top_k_velocity` (a distinct
velocity-only approximation) are not included. Ordinary posterior top-k is supported.

Example validation run (shortened budget, unchanged architecture/batch):

```bash
CUDA_VISIBLE_DEVICES=4 python main.py \
  +experiment=sudoku_spherical_truncated_adaptive \
  trainer.max_steps=2000 checkpoint.every_n_steps=2000 \
  data.cache_dir=/home/patrick/.cache/duo/data_cache
```
