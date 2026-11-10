# Training time samplers

Recipe context: [denoising task](../concepts/denoising-task.md).

A training time sampler decides how often training visits each time `t`. It is
used in training only and has no role at inference: generation never consults it,
in any recipe (`core/tests/test_generation_ignores_time_sampler.py` checks every
preset). The [noise schedule](noise-schedules.md) fixes what each `t` means (the
corruption level) and is shared by training and generation, and so is a
[learned time map](noise-schedules.md#learned-time-maps) when a recipe has one. The config group keeps its
historical name, `denoising_task/time_sampler` (selected as `time_sampler`), since
saved run configs and protocols depend on it.

In training only the combination of the two, plus the loss weighting, matters.
They separate because generation reuses the schedule and not the sampler, and
because an adaptive schedule's linearity constraint (CDCD learns the schedule so
the loss rises linearly in `t`) pins the schedule's shape. Inference
discretization belongs to [samplers](samplers.md). Changing the training time
sampler changes neither the generation grid nor the noise schedule an adaptive
schedule learns, and it adds an importance correction only when its
`objective_weighting` asks for one.

## The contract

Every training time sampler declares, besides `sample(count, device, ...)`
([`core/time_sampling/density.py`](../../core/time_sampling/density.py)):

- `time_density(time)`: for the times `sample` just returned, the density of each
  draw relative to uniform on `[0, 1]`. A uniform sampler reports exactly 1. The
  `epsilon` floor of the base uniform draw is ignored, because it rescales every
  density by the same factor of at most `1 / (1 - epsilon)`. A sampler whose draws
  have no density (a point mass) sets `time_density = None` and says why.
- `objective_weighting`: how a non-uniform draw enters the optimized loss.
  - `importance_corrected`: the per-token loss is divided by the draw density, so
    the expected objective is the uniform-time objective and only the estimator
    variance changes.
  - `uncorrected`: the loss is used as drawn, so oversampling a region of `t` also
    up-weights it in the objective. This is what the heuristic adaptive samplers
    have always done.

  For a uniform sampler the two give the same objective.

The algorithm computes the density where it samples `t` and uses it twice. The
optimized loss follows `objective_weighting`. Adaptive noise schedules and clocks
get the density with their observation and divide their fitting weights by it
([noise schedules](noise-schedules.md#fitting-under-a-non-uniform-training-time-sampler)).
Observers always see the unweighted per-`t` loss. An objective that already
divides by the density itself declares `divides_by_time_density` (the hybrid
ELBO), and the algorithm then does not correct a second time. A consumer that
needs the density raises if a sampler does not declare `time_density`.

| `time_sampler` | Density of a draw | Default `objective_weighting` |
| --- | --- | --- |
| `antithetic` (`UniformTimeSampler`) | 1 | `uncorrected` (identical to corrected) |
| `discrete` | `steps * P(step)`, the base density at the step's cell midpoint (1 for a uniform base) | the base's |
| `cdcd` | slope `F'` of its learned CDF (1 until fitted) | `uncorrected` (CDCD's time warping) |
| `adaptive_allocation` | 1 for diffusion rows; NaN for the `t = 0` reconstruction rows (a separate estimator) | `uncorrected` only; `GaussianELBO` weights the estimators |
| `stratified` (`StratifiedUniformTimeSampler`) | 1 | `uncorrected` (REPLAiD hybrid sets `importance_corrected`) |
| `piecewise_fixed` (`PiecewiseTimeSampler`) | the bin's `bins * p_k` | `importance_corrected` |
| `FixedTimeSampler` (block-mixture override) | none (`time_density = None`) | `uncorrected` only |

Every existing recipe keeps its behaviour. Defaults reproduce what each sampler
did before (no algorithm divided by a density, and the hybrid ELBO divided by
`q`), and the YAML files of existing samplers are unchanged. Set
`+time_sampler.objective_weighting=importance_corrected` (with `+`, because the
key is not in those files) to change one.
Variable-length training applies `importance_corrected` to both its token and
insertion losses.

### With a learned time map

When a recipe has a learned time map `u = Q(t)` (`algorithm.time_map`, see
[noise schedules](noise-schedules.md#learned-time-maps)), the training time
sampler draws `t` and the map places the time the views read. The density of the
mapped time relative to uniform is the sampler's draw density times the map's
density at `u` (`mapped_time_density` in
[`density.py`](../../core/time_sampling/density.py)). That product is what
adaptive noise schedules of the mapped views receive, and what the sampler's
`objective_weighting` divides by when it is `importance_corrected`. With a uniform
sampler this is "uniform in `t` through the map", which is importance sampling in
`u` with density `q(u)`: the REPLAiD hybrid's ELBO divides by `q(u)` itself
(`divides_by_time_density`), so its `stratified` sampler declares
`importance_corrected`, and `uncorrected` is refused for it.

## Implementations

| `time_sampler` | Return / purpose |
| --- | --- |
| `antithetic` | UniformTimeSampler; bounded uniform draws, optionally stratified |
| `discrete` | Wraps a base sampler and rounds onto positive D3PM steps |
| `adaptive_allocation` | RePlaid reconstruction/diffusion allocation plus stratified times |
| `stratified` | One float64 uniform draw per stratum, in random row order (the REPLAiD hybrid's coordinator times) |
| `piecewise_fixed` | Fixed piecewise-constant density on equal-width bins (`weights`), exact inverse CDF of antithetic uniform draws |

### A non-uniform sampler with an adaptive noise schedule

`+experiment=cdcd_mid_range_times` is the CDCD Gaussian recipe (`cdcd_gaussian_adaptive`
schedule) with `time_sampler=piecewise_fixed`, `weights: [1, 3, 3, 1]`. Training
visits mid-range `t` three times as often as the ends. The adaptive schedule still
learns the schedule that uniform times would give, and with
`importance_corrected` the objective is unchanged too, so only the gradient
variance differs from `+experiment=cdcd`. Generation uses the learned schedule
and never touches the sampler.

```bash
uv run --locked python main.py +experiment=cdcd_mid_range_times --cfg job
uv run --locked python main.py +experiment=cdcd_mid_range_times \
  'time_sampler.weights=[1,8,8,1]' time_sampler.objective_weighting=uncorrected
```

### Learned maps that generation reads live on the information path

A fitted map that reshapes how the process moves through `t` is not a training
time sampler. The learned loss CDFs (`cdcd_loss_cdf`, and SDM's loss-derivative
CDF `loss_derivative`, which spends more training on the `t` where the loss
changes fastest) and the learned importance density (`piecewise_importance`) are [learned time maps](noise-schedules.md#learned-time-maps)
in `information_path/time_map`, placed at `algorithm.time_map`, and the training
time sampler of those recipes draws uniform times:

| Recipe | Learned time map | Training time sampler |
| --- | --- | --- |
| `path_schedule=cdcd` (hybrid-sc, continuous-schedules, sc-vs-depth CDCD cases), `sudoku_pca_blurring` | `cdcd_loss_cdf`, read by `adaptive_time_views: [0]` | `antithetic`, `epsilon: 0.001` |
| `replaid_hybrid` | `piecewise_importance`, read by `adaptive_time_views: [0, 1]` | `stratified`, `importance_corrected` |
| `simplex`, `sudoku_simplex*` | `loss_derivative`, warping the whole path (`adaptive: false`, the identity, except in `sudoku_simplex_adaptive`; `sampler.time_grid=adaptive` reads it) | `antithetic`, `epsilon: 1.0e-05` |

These draw the same training times, fit the same maps and generate the same
tokens as when the map sat in the sampler slot. Saved configs and checkpoints in
that older layout migrate on load
([`legacy_checkpoints.py`](../../core/training/legacy_checkpoints.py)).
The former `time_sampler=loss_derivative` option is gone: the loss-derivative
clock is always a learned time map (`information_path/time_map=loss_derivative` at
`algorithm.time_map`), and saved configs naming `LossDerivativeTimeSampler`
migrate to it whatever recipe they belong to. `cdcd` stays available as a
training time sampler for recipes whose generation does not read its map (an
adaptive sampler on a Gaussian or simplex path without view-local times).

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

The simplex sampler's grid can be `linear`, `cosine`, or `adaptive`. `adaptive`
applies the recipe's fitted learned time map (`algorithm.time_map`, e.g.
`information_path/time_map=loss_derivative`) to a uniform grid and errors if
there is none or it has not been fitted; it never reads the training time
sampler. The grid has descending times with exact zero/one endpoints. See
[SimplexInferenceGrid](../../core/samplers/simplex_grid.py).

[AdaptiveTimeAllocation](../../core/time_sampling/adaptive_allocation.py) returns
`TimeAllocation(time, reconstruction_count)`. It estimates reconstruction and
diffusion standard deviations and allocates a leading reconstruction subset at
time zero. Remaining examples receive stratified diffusion times. The shared loss
accepts such draw objects (anything with `.time`) and passes them to the objective as
`ObjectiveContext.time_draw`; `GaussianELBO` accounts for subset sizes. The sampler
declares `min_batch = 2` (singleton batches are replicated) and observes the
objective's `reconstruction` and `diffusion` terms through `observe_terms`. Its EMA
statistics are checkpointed buffers.

### Learned importance density

The REPLAiD hybrid's learned piecewise-constant importance density is a learned
time map (`information_path/time_map=piecewise_importance`,
[`PiecewiseImportanceTimeMap`](../../core/noise_schedules/learned_time_maps.py));
see [noise schedules](noise-schedules.md#learned-time-maps). Its training time
sampler is `stratified`: stratified float64 uniform times `t`, which the map sends
to the views' local time `u = Q(t)`.

## Insertion-loss feedback

[Variable-length diffusion](../modifiers/variable-length.md) feeds the
loss-derivative time map (a whole-path learned warp of the training times). Select
it with `+information_path/time_map@algorithm.time_map=loss_derivative`.

By default, the map observes loss averaged over surviving target tokens;
fully deleted examples do not contribute token-only observations. Set
`variable_length.time_sampling_include_aux=true` to instead observe, per example,

```text
(surviving-token loss sum + weighted insertion-count loss
                         + weighted insertion-content loss)
/ original target-token count
```

This retains fully deleted examples, excludes examples without targets, and
uses the insertion objective's existing count/content weights. Feedback is
detached and updated only during training. The option does not change the
optimization objective, does not add importance weighting, and does not change
per-view adaptive noise-schedule feedback. Under block insertion, every block supplies an observation using its own loss
and original target count; all blocks are trained together. Blocks from one
example share the global time sampled for that example.

The insertion sampler still uses a fixed descending inference grid. Fitting a
training-time distribution does not enable adaptive insertion inference or
learn the structural survival schedule. Specialized allocation samplers such
as RePlaid's `adaptive_allocation` are not interchangeable with this tensor-time
interface.

## Extending

Keep sampling, observation updates, and corruption mapping separate. A new
training time sampler declares `time_density` (or `time_density = None` with the
reason) and `objective_weighting` (validate it with `check_objective_weighting`),
and its density must integrate to 1 and match its draws' histogram
(`core/tests/test_time_sampler_density.py`). Document
whether the output is times alone or an allocation object and whether an objective
needs correction weights. Register fitted state, suppress observations during
validation/generation, and test distributed partitioning and checkpoint resume.
See [configs](../../configs/denoising_task/time_sampler/) and [objectives](objectives.md).
