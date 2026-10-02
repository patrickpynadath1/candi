# Gaussian diffusion

`GaussianDiffusion` operates on continuous representations of discrete tokens:

$$z_t=\alpha(t)e(x_0)+\sigma(t)\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

The [representation](../components/representations.md) owns token encoding, the posterior mean $\sum_v p_v e_v$, and decoding. One-hot and learned embeddings share the algorithm and continuous samplers. The [schedule](../components/noise-schedules.md) owns coefficients and derivatives; the [model](../components/models.md) owns preconditioning and time conditioning. The extra model input projection is retained.

The algorithm draws times, corrupts encoded tokens, obtains an unshaped clean-token posterior, and evaluates CE or a configured weighted objective. The sampler converts that posterior into a mean and a velocity, then integrates toward clean data. Direct neural velocity prediction is not implemented.

```bash
uv run python main.py +experiment=gaussian_diffusion_sudoku
uv run python main.py +experiment=gaussian_diffusion_embedding_sudoku
uv run python main.py +experiment=embedding_flow_matching task=text8
```

## Presets and correspondence

These configurations assemble available components; they do not imply legacy checkpoint compatibility or benchmark reproduction.

| Experiment | Construction | Difference or boundary |
|---|---|---|
| `cont_diff_onehot` | One-hot vocabulary-normalized rank lookup | Legacy training path, core probability-flow sampling |
| `onehot_gaussian` | One-hot linear interpolation; no preconditioning | Fixed schedule |
| `onehot_gaussian_adaptive` | Same with S-FLM adaptation | Original sampled-time coordinates; local legacy rank-adaptive experiments used physical coordinates |
| `embedding_flow_matching` | Trainable normalized embeddings, linear interpolation | Additional input projection retained |
| `embedding_ve` | Embeddings, exponential VE, sigma 0.01–2 | Same training/inference path; legacy inference used a different rank schedule |
| `candi_embedding_pure_linear` | Embedding linear interpolation | Pure continuous branch |
| `candi_embedding_pure_linear_rank` | Sphere-rank approximation, linear rank progression | Rank 0.01–0.49; no endpoint extensions |
| `candi_embedding_pure_geometric_rank` | Sphere-rank approximation, geometric progression | Same interval and endpoints |
| `candi_rank_pure_linear` | One-hot analytic inverse, competitor normalization | Rank 0.01–0.49; sigma intentionally clamped to 0.2–4 |
| `candi_rank_pure_geometric` | Same, geometric rank progression | Same clamp |
| `semicat_gaussian` | One-hot linear interpolation, optional label smoothing | Core clean-to-noise time conditioning, not legacy reversed conditioning |
| `cfm_gaussian` | Semicat counterpart's objective/path | ContinuousDiT, not the original CFM backbone |
| `flm_gaussian` | Argmax-error calibration plus coefficient rescaling | Finite sigma 0.2–4, core backbone/global-time conditioning rather than FLMDIT |

`gaussian_base` supplies the shared assembly. One-hot rank offers exact lookup and analytic approximation; embedding rank uses unit-sphere geometry, not the current learned table's pairwise geometry. Construction rejects incompatible representation/schedule combinations. Contextual or sequence-compressing encoders are not implemented.

FLM's error target spans the error interval achievable by the finite lookup table, rather than the original full chance-normalized interval. Optional learned loss weighting defaults to global time, not the original pre-interpolation alpha coordinate.

## Solvers and objectives

```bash
uv run python main.py +experiment=gaussian_diffusion_sudoku sampler=heun
uv run python main.py +experiment=gaussian_diffusion_sudoku sampler=ode
uv run python main.py +experiment=flm_gaussian parameterization.softcap=30 objective=cross_entropy_learned
```

Fixed Gaussian presets default to Euler. Euler and Heun integrate posterior-derived velocity; the adaptive ODE solver counts rejected stages against its budget. Solvers preserve prompts at every internal stage and include final posterior decoding in NFE. They support both tokenwise representations. The Gaussian DDPM sampler is also available for suitable monotone-SNR paths; its prior policy must match the intended terminal distribution.

Changing a preset does not reproduce every original finite-step update or endpoint grid. [RePlaid](replaid.md), [blurring](blurring-diffusion.md), [spherical](spherical-flow-matching.md), and [hybrids](hybrids.md) have specialized assemblies, despite sharing some Gaussian components.

## Validation

All listed method presets were composed and instantiated with small models, checked for finite loss/gradients and prompt-preserving samples, and composed with Text8. This is execution coverage, not a benchmark for every preset.

A 2,000-step one-hot Sudoku Hard pilot achieved 394/2,000 exact solves (19.70%) with EMA and Euler at 181 NFE. On that same checkpoint, Heun achieved 19.25% at 181 NFE and adaptive ODE 17.80% at mean 75 NFE, without shaping tuning. Representation extraction preserved all per-puzzle one-hot results. Embedding geometry has smoke and schedule-parity checks; these are not embedding quality claims. Details are in [reproduction results](../validation/reproduction-results.md).

Sources: [algorithm](../../core/algorithms/gaussian_diffusion.py), [experiment configurations](../../configs/experiment/).
