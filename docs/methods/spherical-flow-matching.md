# Spherical flow matching

Spherical flow matching encodes tokens as trainable unit vectors and interpolates on the sphere toward independent uniform sphere noise. It uses SphereDiT, an unconstrained posterior, CE, and posterior-derived tangent velocities. The algorithm coordinates training and prediction; the process and sampler own geometry and integration.

```bash
uv run python main.py +experiment=spherical_flow_matching
uv run python main.py +experiment=sudoku_spherical
uv run python main.py +experiment=sudoku_spherical_adaptive
uv run python main.py +experiment=sudoku_spherical_truncated_adaptive
```

## Coordinates and adaptation

Time increases toward noise. The schedule returns a keep coordinate and corruption uses interpolation fraction `1 - keep`. Dimension-aware truncation replaces the maximum keep with the vocabulary/dimension bound; the lowest-time endpoint therefore remains partially corrupted. It is not remapped to a fully clean endpoint.

The generic `noise_schedule=spherical_adaptive` defaults to `coordinate: legacy_time`, matching original S-FLM sampled-time adaptation. The named Sudoku adaptive presets explicitly choose `physical`, matching this repository's original Sudoku recipes. These coordinates are intentionally distinct; see [noise schedules](../components/noise-schedules.md).

The truncated Sudoku recipe uses a 12,800-observation buffer, refits every 50 steps after step 1,000, EMA 0.9, and uniform mixture 0.001. Its model is width/dimension 512 with eight blocks and eight heads; batch 256, learning rate 0.0003, warmup 2,500, EMA 0.9999, and a 20,000-step budget. Accumulation is one.

## Sampling

`sampler.velocity=exact` averages token-directed tangent vectors under the posterior; `sample` draws a token and follows its tangent. The last requested model call decodes tokens rather than taking an additional integration step. Thus the standard setting requests 180 model calls.

The sampler keeps prompt embeddings clean at every step. Geometry defaults to float64; `sampler.use_float64` separately controls probability calculations. `parameterization.commitment=sample` matches the reference final categorical draw; `greedy` selects deterministic decoding. Temperature/filtering remain predictor settings.

SphereArch's additional weight projections and the legacy velocity-only `top_k_velocity` approximation are not included. Ordinary posterior top-k remains available. Shared [self-conditioning](../modifiers/self-conditioning.md) supplies local trajectory feedback rather than a separate spherical wrapper.

## Evidence

A shortened 2,000-step run with the truncated adaptive preset completed 19 schedule refits and solved 200/2,000 Sudoku Hard puzzles (10.0%) with EMA and 180 model calls. Digit accuracy was 82.791%. The run was still within the original 2,500-step warmup, so this is an early-training check, not completed-budget reproduction. Geometry, velocity, adaptation, checkpoint loading, and mixed-precision behavior have regression coverage. See [validation](../validation/reproduction-results.md).

Sources: [algorithm](../../core/algorithms/spherical_flow_matching.py), [sampler](../../core/samplers/spherical.py), [presets](../../configs/experiment/).
