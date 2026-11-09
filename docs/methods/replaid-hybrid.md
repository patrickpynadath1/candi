# Conditional REPLAiD hybrid

The `replaid_hybrid` preset is a recipe over [`HybridDiffusion`](hybrids.md) with no
algorithm class of its own; every part is a reusable component selected by config.
Its trained token table lives in a 16-dimensional unit-normalized space. Gaussian
noise applies only to unrevealed tokens; revealed tokens also enter a separate
clean-context projection. Posterior self-conditioning is disabled. One learned
importance clock places the shared local time that determines both the Gaussian log
noise-to-signal and the absorbing-mask probability. Its time-conditioning floor is
`1e-30`, preserving the large log-noise values learned by the original hybrid
checkpoint rather than clipping them at the general model default.

| Recipe part | Config | Component |
|---|---|---|
| Gaussian view | `hybrid_view@model.input.views.content: learned_log_snr_gaussian` | Gaussian process, `LinearLogSNRSchedule` (learned endpoints), the recipe's embedding representation |
| Reveal view | `hybrid_view@model.input.views.revealed: forced_target_mask` | `AbsorbingProcess` with the `ForcedTargetMask` proposal, keep rate `1 - u` (`LogLinear`, `epsilon: 0`) |
| Model input | `model/input@model.input: revealed_context` | `RevealedContextInput`: linear continuous input plus clean context at revealed positions |
| Time sampler | `time_sampler: piecewise_importance`, `algorithm.adaptive_time_views: [0, 1]` | `PiecewiseImportanceTimeSampler`: stratified uniform draws, inverse-CDF quantile map, proposal density, variance surrogate |
| Parameterization | `parameterization: gaussian_prior` | Gaussian-prior correction on the Gaussian view (`prior_view: 0`) |
| Objective | `objective: hybrid_elbo` | `HybridELBO`: reveal CE, Gaussian expected-embedding error, terminal Gaussian KL |
| Algorithm | `algorithm: hybrid_diffusion`, `synchronize_committed: true` | The shared hybrid loss |
| Sampler | `sampler: revealed_hybrid` | Composed sampler that decodes the final token state |

## Training step

Both views are adaptive-time views: the coordinator draws stratified uniform times
`t`, the sampler's quantile map gives the local time `u = Q(t)` (detached), and both
views' schedules read `u`. The Gaussian view corrupts every target; the reveal view
masks with probability `u` and forces one target; synchronizing committed tokens
then restores the clean embedding wherever the reveal view kept the token, so Gaussian
noise applies only to masked tokens. `HybridELBO` reads the masks, coefficients,
states and the proposal density `q(u)` from `ObjectiveContext`, divides its interior
estimator by the detached density, and reports each row's estimate. In training the
time sampler's `adaptation_loss` adds the zero-valued score-function surrogate on that
estimate, so the density logits receive the second-moment (variance) gradient and
nothing else; endpoints and embeddings receive ELBO gradients. Validation adds no
surrogate. Generation uses the same quantile map on its uniform grid.

The objective requires binary loss masks, `corruption_mask == loss_mask`, and
rejects block sequence layouts (`supports_sequence_layout = False`). Self-conditioning
and block-diffusion combinations are not implied by this recipe. Optimizer groups
apply the special learning rate to the codebook and clock without decay.

## Checkpoints

Native checkpoints include model, representation, schedule endpoints, importance
clock and EMA. The former `RePlaidHybrid` class stored the shared clock under both
views (`model.input.views.{0,1}.schedule.clock.*`); now the endpoints are
`model.input.views.0.schedule.{gamma_min,gamma_delta_raw}` and the bin logits
`time_sampler.clock_logits`. Loading migrates such checkpoints automatically
(`core.training.legacy_checkpoints.migrate_legacy_checkpoint`): it renames the keys,
drops the duplicate and reorders EMA and optimizer state. Saved run configs naming
the removed class are rebuilt from this preset by the runner
(`migrate_legacy_config`), keeping the saved task, data, trainer, evaluation and
sampler settings and the old class's arguments (model sizes, clock endpoints and
bins, context scale); the checkpoint's own saved config is migrated the same way
before the recipe check. The resolved, migrated config is written to the output
directory.

For an exported evaluation bundle (old or new):

```bash
uv run --locked python main.py --config-path /absolute/path/to/bundle --config-name config \
  mode=evaluate evaluation.checkpoint=/absolute/path/to/bundle/model.ckpt \
  data.cache_dir=/local/data task.evaluation.cache_dir=/local/data \
  output_dir=/absolute/path/to/evaluation hydra.run.dir=/absolute/path/to/evaluation
```

`config.yaml` selects EMA; `config-raw.yaml` selects raw weights. A bundle's saved
`evaluation.checkpoint`, data and output paths are those of the machine that wrote
it, so override them as above. Set `evaluation.device=cuda` only when GPUs have been
allocated. Sampling/evaluation uses the same task/tokenizer specified by the saved
config. Exported historical bundles are for evaluation or an explicit warm start;
they do not claim to restore the old Lightning optimizer/loop/RNG state for exact
training resumption.

Validation covers independent estimator/gradient checks, two CPU Lightning
updates, shared-clock restoration, prompt preservation and actual model-call
NFE counts (`core/tests/test_replaid_hybrid.py`, `core/tests/test_replaid_components.py`).
The move from the `RePlaidHybrid` class to this recipe was checked with fixed random
variates: losses, all parameter gradients (including clock and endpoints) and a short
generation agree to float rounding.
