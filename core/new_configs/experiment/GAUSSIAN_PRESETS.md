# Gaussian method presets

These presets compose the current core Gaussian components. They default to the
configured task (Sudoku Hard in the main config); `task=text8`, for example, can
replace it. They retain the additional model projection, unshaped training
predictions, and posterior-based continuous samplers. They are not claims of
legacy checkpoint compatibility or reproduced benchmark scores.

| Experiment | Path / representation | Reproduction notes |
| --- | --- | --- |
| `cont_diff_onehot` | One-hot, vocabulary-normalized rank lookup | Legacy training path; core probability-flow sampling |
| `onehot_gaussian` | One-hot, linear interpolation, no preconditioning | Fixed schedule; see adaptive variant below |
| `onehot_gaussian_adaptive` | Same, with S-FLM adaptation | Explicit original S-FLM sampled-time coordinates; local legacy rank-adaptive experiments used physical coordinates |
| `embedding_flow_matching` | Trainable normalized embeddings, linear interpolation | Additional model projection retained |
| `embedding_ve` | Trainable normalized embeddings, exponential VE, sigma 0.01–2 | Uses the same path in training and sampling; legacy inference used a different rank schedule |
| `candi_embedding_pure_linear` | Embedding linear interpolation | Pure Gaussian branch only |
| `candi_embedding_pure_linear_rank` | Embedding sphere-rank, linear rank progression | Rank interval 0.01–0.49; no endpoint extensions |
| `candi_embedding_pure_geometric_rank` | Embedding sphere-rank, geometric rank progression | Same interval and endpoint settings |
| `candi_rank_pure_linear` | One-hot analytic inverse, competitor normalization | Target rank 0.01–0.49, **sigma clamped to 0.2–4** |
| `candi_rank_pure_geometric` | Same, geometric rank progression | Same intentional clamp |
| `semicat_gaussian` | One-hot linear FM, optional label smoothing | Core clean-to-noise time conditioning rather than legacy reversed conditioning |
| `cfm_gaussian` | Same objective/path as the Semicat counterpart | Uses ContinuousDiT, not the original CFM backbone |
| `flm_gaussian` | Argmax-error calibration, linear coefficient rescaling | Finite sigma range 0.2–4; core backbone/global-time conditioning rather than FLMDIT |

Example:

```bash
python main.py +experiment=candi_rank_pure_geometric
python main.py +experiment=embedding_flow_matching task=text8
python main.py +experiment=flm_gaussian parameterization.softcap=30 objective=cross_entropy_learned
```

All fixed presets use the core Euler sampler by default; the available Heun and
ODE solvers can be selected separately. Original finite-step transition formulas
and exact endpoint grids are not reproduced by merely changing presets.
Embedding-rank schedules are calibrated to the unit-sphere approximation, not the
current learned table's individual geometry. FLM's error target is linear across
the error interval attained by the finite sigma table, not its original full
chance-normalized probability interval. Learned weighting is optional and defaults
to global-time conditioning, not FLM's pre-interpolation alpha coordinate.

`gaussian_base` is the shared assembly config. Existing
`gaussian_diffusion_sudoku` and `gaussian_diffusion_embedding_sudoku` configs are
unchanged. RePlaid, BlurringDiffusion, FRM and hybrid CANDI variants are not exposed
as Gaussian presets because the necessary specialized components are not ported.

Validation: every named preset composed and instantiated with a small model, then
produced finite CPU loss/gradients and samples preserving prompt tokens. Task
replacement was checked by composing all presets with Text8. These are execution
checks, not quality benchmarks.
