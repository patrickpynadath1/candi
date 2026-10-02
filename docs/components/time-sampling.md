# Time sampling

Training-time sampling chooses which global times to train on. A noise schedule
then maps those times to corruption coefficients. Inference discretization belongs
to samplers; changing a training distribution does not automatically change the
generation grid or introduce an importance correction.

## Implementations

| `time_sampler` | Return / purpose |
| --- | --- |
| `antithetic` | UniformTimeSampler; bounded uniform draws, optionally stratified |
| `discrete` | Wraps a base sampler and rounds onto positive D3PM steps |
| `loss_derivative` | Simplex's optional fitted inverse-CDF training-time sampler |
| `adaptive_allocation` | RePlaid reconstruction/diffusion allocation plus stratified times |

[UniformTimeSampler](../../core/time_sampling/uniform.py) draws in
`[epsilon,1)`, with `epsilon=1e-3` by default. Antithetic mode spreads draws across
uniform strata. `sample(count, device, ...)` accepts explicit global batch size,
rank/world size, and accumulation index/count. It partitions a global set into
`[world_size, accumulation_steps, local_batch]`; inconsistent divisibility or
partition indices raise. The trainer/algorithm supplies context; the sampler does
not inspect Lightning. Set `time_sampler.antithetic=false` for ordinary uniform
sampling under the same selector.

[DiscreteTimeSampler](../../core/time_sampling/discrete.py) maps a base time to
`min(floor(t*steps)+1,steps)/steps`. The grid contains positive times through one;
the objective and sampler step counts must describe the same D3PM construction.

## Adaptive distributions

[LossDerivativeTimeSampler](../../core/time_sampling/loss_derivative.py) defaults
to `adaptive=false`. When enabled, it collects detached masked losses, gathers
observations across ranks, bins them, interpolates the loss profile, and forms a
CDF from its nonnegative derivative. Warmup, buffer capacity, refit cadence, and
EMA are explicit settings. A flat or insufficiently observed profile leaves the
previous/uniform clock intact. This is a distribution over times, not a learned
corruption formula. It does not automatically importance-correct CE.

The simplex inference helper can use `linear`, `cosine`, or `adaptive` grids.
Adaptive inference requires an enabled, fitted training sampler and errors if it
is unavailable. The helper returns descending times with exact zero/one endpoints.
See [SimplexInferenceGrid](../../core/time_sampling/simplex_grid.py).

[AdaptiveTimeAllocation](../../core/time_sampling/adaptive_allocation.py) returns
`TimeAllocation(time, reconstruction_count)`. It estimates reconstruction and
diffusion standard deviations and allocates a leading reconstruction subset at
time zero. Remaining examples receive stratified diffusion times. RePlaid's
objective accounts for subset sizes; this object is not a drop-in replacement for
a sampler returning only a tensor. Its EMA statistics are checkpointed buffers.

## Extending

Keep sampling, observation updates, and corruption mapping separate. Document
whether the output is times alone or an allocation object and whether an objective
needs correction weights. Register fitted state, suppress observations during
validation/generation, and test distributed partitioning and checkpoint resume.
See [configs](../../configs/time_sampler/) and [objectives](objectives.md).
