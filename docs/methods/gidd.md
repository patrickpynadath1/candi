# GIDD mask/uniform corruption

[Generalized Interpolating Discrete Diffusion (GIDD)](https://proceedings.mlr.press/v267/von-rutte25a.html)
defines a general family of discrete interpolation processes. This implementation
adds its mask/uniform information path and marginal plug-in reverse transition.
It follows the [official implementation](https://github.com/dvruette/gidd/tree/f67acc961cc858cc69e8e6d91e7eb67a0a79151c).

## Recipe

- **Information path:** one token sequence with clean-token, uniform-replacement,
  and mask mass. The input vocabulary has one extra mask symbol; uniform noise
  is distributed over the clean vocabulary only.
- **Denoising task:** predict the clean-token distribution with cross-entropy.
  Self-conditioning uses the existing shared modifier.
- **Inference settings:** start all-mask, use GIDD's marginal plug-in reverse
  update, then use the shared sampler's terminal decoding.

This is a corruption option with a CE baseline, **not a reproduction of the
paper's ELBO or reweighted GIDD+ loss**. The paper's additional self-correction
sampler is not included. Insertion and EB combinations have not been validated
for this process.

Although the implementation uses the shared hybrid-view interface, this preset
has one noisy view. Mixing masking and replacement noise in that view is
different from presenting multiple noisy views to the backbone.

## Schedule

For time `t` in `[0, 1]`, let

```text
B = 2**gamma * p_uniform / (1 - p_uniform)
c(t) = B * (t * (1 - t))**(gamma / 2)
alpha(t)   = (1 - t) / (1 + c(t))
mask(t)    = t / (1 + c(t))
uniform(t) = c(t) / (1 + c(t))
```

The forward marginal is `alpha * one_hot(x0) + uniform / V` over clean tokens,
with the remaining `mask` mass on the mask symbol. Uniform replacements can
coincidentally reproduce the original token. `p_uniform` is the peak uniform
mixing mass, reached at `t=0.5`, rather than the observed token-error rate.

Defaults are `p_uniform=0.2` and `gamma=1.0`. The schedule accepts
`0 <= p_uniform < 1` and `0 < gamma <= 2`, which ensure valid forward conditional
mixing. `p_uniform=0` gives the linear absorbing-mask path. Both endpoints are
exact: clean at zero, all-mask at one.

## Configuration

Inspect the composed recipe without launching training:

```bash
uv run --locked python main.py +experiment=gidd_corruption --cfg job
```

Train the default Sudoku recipe with self-conditioning:

```bash
uv run --locked python main.py +experiment=gidd_corruption \
  self_conditioning.enabled=true \
  model.views.content.schedule.p_uniform=0.2 \
  model.views.content.schedule.gamma=1.0
```

The [preset](../../configs/experiment/gidd_corruption.yaml) inherits the common
corruption-factorial training configuration. To compose this information path
into another hybrid recipe, select `hybrid_view@model.views.content=gidd`.
Task configuration remains separate from the corruption choice.

Implementation: [schedule](../../core/noise_schedules/gidd.py),
[process](../../core/processes/gidd.py),
[view configuration](../../configs/hybrid_view/gidd.yaml), and
[update](../../core/samplers/hybrid_updates.py).
Tests cover path validity, reverse probabilities against an explicit forward
transition matrix, the mask-only limit, and training/generation with and without
self-conditioning. These checks do not establish trained-model benchmark parity.
