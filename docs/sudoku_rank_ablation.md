# Sudoku ablations: explicit experiment matrix

The authoritative list is [`configs/sweeps/sudoku_rank.yaml`](../configs/sweeps/sudoku_rank.yaml).
It contains **eight pure continuous and eight hybrid configurations**, following
per-variant mode restrictions. With easy/medium/hard Sudoku and three seeds,
the full plan has **144 training runs**, each evaluated at NFE 16/32/64/128/256.
No full training sweep has been launched.

| Mode | Space | Requested ablation | Manifest key |
|---|---|---|---|
| Pure | Embeddings | Gaussian adaptive | `embedding_linear_adaptive` |
| Pure | Embeddings | Gaussian rank geo | `embedding_geometric_rank` |
| Pure | Embeddings | Gaussian rank linear | `embedding_linear_rank` |
| Pure | Embeddings | Spherical truncated adaptive | `slerp_truncated_adaptive` |
| Pure | Embeddings | Spherical truncated rank geo | `slerp_geometric_rank` |
| Pure | One-hot | Gaussian adaptive | `onehot_gaussian_adaptive` |
| Pure | One-hot | Gaussian rank geo | `candi_geometric` |
| Pure | One-hot | Gaussian rank linear | `candi_orig` |
| Hybrid | Embeddings | Gaussian linear interpolation | `embedding_linear` |
| Hybrid | Embeddings | Rank linear | `embedding_linear_rank` |
| Hybrid | Embeddings | Rank geo | `embedding_geometric_rank` |
| Hybrid | Embeddings | Rank geo + adaptive | `embedding_geometric_rank_adaptive` |
| Hybrid | Embeddings | SLERP truncated rank geo + adaptive | `slerp_geometric_rank_adaptive` |
| Hybrid | One-hot | Rank linear | `candi_orig` |
| Hybrid | One-hot | Rank geo | `candi_geometric` |
| Hybrid | One-hot | Rank geo + adaptive | `candi_geometric_adaptive` |

“Gaussian adaptive” uses Gaussian corruption with a loss-linearizing adaptive
clock and **no analytic rank coupling**. The base path in both representations is
`z=(1-rho)*clean + rho*N(0,I)`. `onehot_gaussian` uses one-hot clean states;
`candi_embedding` uses learned normalized clean embeddings. Adaptation is an
online approximation to loss linearization, not a guarantee at every step.

Embedding rank linear means rank target `.01 + .48*rho`; rank geo means
`.01*49**rho`. These are distinct from a linear interpolation fraction.
Gaussian embedding rank paths retain their configured exact-endpoint bridges.
One-hot rank paths use CANDI's additive Gaussian parameterization and finite
rank bounds; representation comparisons therefore also retain these documented
path/endpoint differences.

Spherical truncated adaptive uses SLERP with an affine interpolation fraction
inside the paper-style interval `alpha in [0, alpha_max]`, where
`alpha_max=(2/pi)*asin(min(1,sqrt(2*log(2*(V-1)/delta)/d)))`, `delta=.1`.
SFM calculates this value at initialization from the tokenizer vocabulary and
`model.input_embedding_dim`; for V=12,d=512 it is approximately .092734, and
for V=12,d=16 it is approximately .613277. The adaptive clock changes traversal
inside that interval. It does not impose a geometric rank curve. Spherical rank
variants instead truncate using the .01-to-.49 rank bounds.

Use `--modes pure` or `--modes hybrid` to select the corresponding eight rows.
Variant-specific mode restrictions are enforced by the runner; it does not form
an unrestricted Cartesian product. `candi_orig` denotes the paper's linear rank
coupling with the common .01/.49 bounds, not a legacy-checkpoint reproduction.

## Mathematics and source-to-code mapping

Sources:

- [CANDI v3, equations 5/6/10 and §5.1](https://arxiv.org/html/2510.22510v3#S5.SS1)
- [CANDI embedding analysis, Appendix B.4](https://arxiv.org/html/2510.22510v3#A2.SS4)
- [S-FLM v1, Algorithm 4 and Appendix B.2](https://arxiv.org/html/2605.11125v1#A2.SS2)

`algos/rank_schedules.py` defines rank as the fraction of the **other V−1**
tokens that outrank the correct token. One-hot Gaussian rank is
`Phi(-1/(sqrt(2)*sigma))`; its analytic inverse eliminates vocabulary scaling
and lookup clipping. For geometric coupling,
`r(rho)=.01 * (.49/.01)**rho`, hence `sigma(rho=1)=28.2065247`.
`algos/candi_rank.py` implements this coupling and explicit Bernoulli masking.
It also uses finite physical-rho conditioning at the exact noisy endpoint.

For distinct unit embeddings, `z=(1-u)*e_i+u*epsilon`, with standard Gaussian
`epsilon`, a competitor with cosine `c=e_i·e_j` wins with probability
`Phi(-((1-u)/u)*sqrt((1-c)/2))`. Averaging these probabilities gives expected
rank exactly for a fixed embedding table, without assuming independence of
competitor events. Dot-product, cosine, and nearest-unit-embedding rankings
agree. This is an input-geometry diagnostic, not the learned output-head rank.

For iid uniform unit embeddings, cosine density is proportional to
`(1-c*c)**((d-3)/2)`. Gauss-Jacobi quadrature integrates over this density.
The resulting monotone table is inverted to obtain interpolation `u` from
target rank. The table is a checkpointed buffer; it does not change as the
learned embeddings change. Evaluation writes exact current-table rank curves
to `rank_diagnostics.json`, so departures from the iid approximation remain
visible. Individual token embeddings are normalized; their probability
mixtures need not have unit norm. Gaussian noise is not normalized.

The embedding rank path follows `.01 → .49` rank over the complete schedule.
Its default `algo.rank_endpoint_width=0` therefore keeps the interpolation
strictly between `u=0` and `u=1`. A positive endpoint width is available only
as an explicit alternative that bridges to exact clean and pure-noise states.

One-hot CANDI retains the paper-style finite-sigma Gaussian prior approximation:
at the noisy endpoint, training has `onehot(x)+sigma*epsilon`, while the prior
is `sigma*epsilon`. Its rank is .5 under a pure-noise draw versus the .49
conditional training target. This discrepancy is explicit and is not a claim
of an exact finite-sigma terminal marginal. It applies to both modes.

## Pure versus hybrid

`algo.pure_continuous=true` corrupts all answer positions continuously and
trains with raw CE. Hybrid mode independently preserves positions with
probability `1-rho`; CE is weighted by `-alpha'(t)/(1-alpha(t))`, matching the
existing hybrid objective. Both modes always preserve the puzzle prefix.

The hybrid sampler maintains irreversible reveals. The pure sampler performs
continuous updates and decodes the last clean prediction. Both use the same
continuous denoised-estimate transition within a schedule pair and count one
network call per step. Embedding transitions use actual interpolation
increments, so applying an additional time-warp derivative would double-count
the warp. The exact clean prediction transition is first-order consistent
with the corresponding probability-flow ODE.

These paired modes isolate the hybrid mechanism as implemented, including its
objective weighting. An additional hybrid-unweighted-CE control would be a
separate ablation if we later want to isolate masking from objective weighting.

## Adaptive traversal

There is exactly one sampled clock: `t → rho=g(t) → continuous_noise(rho)`.
These ablations do not independently sample continuous and discrete times.
Both new algorithm classes reject `factorized_adaptive_schedule=true`.
The existing factorized CANDI implementation is retained for a future, separate
two-dimensional ablation; it is not included in this manifest.

`noise=rank-adaptive` enables the corrected implementation through
`adaptive_coordinate=physical`; `noise.adaptive` switches adaptation on/off.
The legacy coordinate mode remains the default for old configurations.

The new mode records `(physical corruption, raw generated-span CE)`, fits a
spline, clamps negative derivatives, integrates the derivative plus a coverage
floor, and uses an inverse CDF with bias-corrected schedule EMA. Endpoints are
anchored exactly. Observations from different refits therefore refer to the
same physical coordinate. Distributed workers gather observations before each
fit. Snapshot JSONs contain physical CE, density, learned progress, and CE
mapped through the learned warp. The fitter does not receive the hybrid
likelihood weight. Pure models use unweighted CE without inverse-density
importance correction; hybrid models retain their own objective's chain rule.

Defaults: warmup 1000 optimizer steps, refit every 50, 12800-example buffer
(50 × global batch 256), EMA .9, coverage floor .001. Algorithm 4 is the
reference; explicit endpoint integration and stable physical coordinates are
implementation corrections, rather than bit-for-bit reproduction of its code.

## Commands

All commands run from the repository root. Plan/train/eval/prepare are dry-run
unless `--execute` is supplied (except `plan`, which always previews).
`smoke` runs real CPU training and checkpoint-based evaluation by default.

The current cross-task campaign is restricted to Sudoku Hard, Sudoku Extreme,
and Maze. Its launcher expands the 16 rows above into 48 jobs per seed and uses
a shared FIFO queue. Each configured GPU runs at most one job; when it becomes
idle, it claims the next unclaimed job. Occupied GPUs wait without reserving a
job. Completed jobs are skipped, while interrupted jobs resume from `last.ckpt`.

```bash
# Preview and validate all 48 seed-1 configurations without launching them.
.venv/bin/python new_scripts/launch_reasoning_ablation_campaign.py \
  --root outputs/reasoning_core_ablation_seed1

# Launch the shared queue as a detached supervisor.
.venv/bin/python new_scripts/launch_reasoning_ablation_campaign.py \
  --root outputs/reasoning_core_ablation_seed1 \
  --gpus 0 2 3 4 5 6 --execute --detach
```

The default project is `reasoning-candi-core-ablations`. Add `--seeds 1 2 3`
for 144 jobs after the seed-1 campaign is accepted.

```bash
.venv/bin/python new_scripts/sudoku_rank_ablation.py plan

# All 16 configurations on easy puzzles; small real models, two optimizer steps.
.venv/bin/python new_scripts/sudoku_rank_ablation.py smoke \
  --difficulties easy --root /tmp/sudoku-rank-check

# All three difficulties; defaults to seed 1 and two CPU worker queues.
.venv/bin/python new_scripts/sudoku_rank_ablation.py smoke \
  --smoke-workers 4 --root /tmp/sudoku-rank-check

# Future runs, once the ablation manifest is finalized:
.venv/bin/python new_scripts/sudoku_rank_ablation.py train --gpus 0 1 --execute
.venv/bin/python new_scripts/sudoku_rank_ablation.py eval --gpus 0 1 --execute
.venv/bin/python new_scripts/sudoku_rank_ablation.py aggregate
```

Use `--variants`, `--modes`, `--difficulties`, and `--seeds` to select subsets.
Adding a variant requires a new manifest entry with an experiment and Hydra
overrides. Each GPU receives a sequential queue. Datasets are prepared once
before concurrent workers start. Failed jobs report their logs and can resume
from the last checkpoint; completed stages are skipped. Configuration
fingerprints reject accidental reuse of a directory for a different run.
Changing code intentionally requires a new run root or an explicit checkpoint
migration; configuration fingerprints do not certify code equivalence.

Artifacts include resolved Hydra configs, exact command arguments, stage logs
and status/timing JSONs, checkpoints, adaptive snapshots, per-puzzle predictions,
rank diagnostics, `summary.csv`, and `summary_by_variant.csv` (seed mean/std).
Evaluation reports actual NFE, exact solve rate, digit-only accuracy, grid
validity, clue consistency, valid solutions satisfying clues, and time per
puzzle. Timing includes batch preparation and metric calculation, not model
loading. The default evaluation uses the existing validation split; reserve
an independent final test set before selecting and reporting final ablations.

The full configurations use 48k training / 2k validation puzzles per difficulty,
20k optimizer steps, global batch 256, and fixed data seed 42. Difficulty is
the generator's target clue count (40/35/30), not a solver difficulty rating.

## Acceptance checks

Focused tests in `tests/unit/algos/test_rank_schedules.py` and
`test_rank_ablations.py` cover analytic and Monte Carlo rank, quadrature
convergence, endpoints, monotonicity, synthetic loss linearization, repeated
refits, recursive checkpoint restoration, real-model gradients, and prompt
preservation. Shipping config checks include both new algorithm classes.

Smoke tests exercise `main.py`, actual optimizer steps, validation, checkpoint
serialization, fresh-process loading, and Sudoku generation. They validate
plumbing, not puzzle-solving quality or long-run numerical stability. GPU
throughput and multi-GPU synchronization need validation on a working CUDA host.

The CPU smoke exposed an existing Lightning iterator bug: replacing a loader
in `on_train_start` did not replace the iterator already created by Lightning.
The trainer now rebuilds that iterator and restores the non-distributed
sampler's random state when resuming; regression tests cover active sampler use.

### Verified on 2026-09-05

- 239 focused and regression tests passed across algorithms, real configs,
  samplers, training/checkpoint hooks, data loading, and Sudoku evaluation.
- All 36 CPU smoke cases (six schedules × two modes × three difficulties,
  seed 1) trained for two optimizer steps and evaluated four puzzles at four
  actual NFE using fresh-process checkpoint loading.
- All 36 evaluation files and rank diagnostic curves were finite, and the
  diagnostic paths were monotone with exact physical-progress endpoints.
- Both adaptive hybrid families resumed from optimizer step 1, reached step 3,
  and restored/advanced their adaptive state to three refits.
- At that time, the original manifest contained 108 full-run configurations.
  The current explicit matrix at the top of this document supersedes that list.

Smoke artifacts: `/tmp/duo_rank_final/smoke/summary.csv` and
`/tmp/duo_rank_final/smoke/summary_by_variant.csv`. Resume logs:
`/tmp/duo_rank_resume/`. CUDA was unavailable on this host; these checks do not
establish GPU throughput, distributed execution, or trained-model accuracy.


## Spherical rank coupling

`algo=candi_slerp` uses `SLERPRankSchedule` from `algos/rank_schedules.py`.
The dimension comes from `model.hidden_size`; the calibration table and dimension
are checkpointed. The iid spherical expectation is integrated with Gauss-Jacobi
quadrature and inverted monotonically. It is separate from the Gaussian linear
interpolation calibration. See [the derivation](spherical_rank_schedule.md).

The single clock is `t -> rho=1-alpha(t) -> r=.01*49**rho -> u_d(r)`.
Hybrid corruption preserves each answer position with probability `1-rho`;
other positions receive SLERP noise. Pure mode corrupts every answer position.
Both keep puzzle prompts fixed, and use the shared ContDIT backbone. Hybrid CE
weighting and raw-CE adaptive observations follow the existing CANDI embedding
variant. The spherical transition integrates the posterior-weighted logarithmic
map using `(u_t-u_s)/u_t`, then applies the existing irreversible reveal process.
No extra time-warp derivative is applied to those finite physical increments.

The default `rank_endpoint_width=0` preserves exactly the requested .01 and .49
rank targets. This intentionally truncates both ends of the continuous path,
as in SFM truncation. Sampling starts from a uniform sphere, whose expected rank
is .5; this is an explicit prior approximation at the .49 training endpoint.
Set `algo.rank_endpoint_width=0.01` for boundary bridges to exact clean and
independent-noise endpoints (geometric rank then holds only in the interior).
Calibration assumes iid uniform embeddings; learned embeddings may depart from
that distribution. Diagnostics label the reported curve as iid, not empirical.

Run the spherical rank variants in their configured modes and all difficulties:

```bash
.venv/bin/python new_scripts/sudoku_rank_ablation.py smoke \
  --root /tmp/duo_slerp_rank \
  --variants slerp_geometric_rank slerp_geometric_rank_adaptive

.venv/bin/python new_scripts/sudoku_rank_ablation.py train \
  --root outputs/sudoku_rank \
  --variants slerp_geometric_rank slerp_geometric_rank_adaptive \
  --gpus 0 1 --execute
```

Omit `--execute` on the train command to inspect the run plan.


### Explicit matrix validation (2026-09-06)

- 500 algorithm, model, sampler, and Sudoku regression tests passed.
- All 16 configured rows completed two-step CPU training and fresh-process
  checkpoint evaluation on easy Sudoku (seed 1, four evaluation puzzles, NFE 4).
- Artifacts: `/tmp/duo_requested_matrix/smoke/summary.csv`.
- The full 144-run plan is configured; full training has not been launched.
