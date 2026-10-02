# Simplex diffusion port

This implementation **copies and refactors** the working implementation from
`/home/patrick/.codex/worktrees/0303/duo`. Core has no runtime dependency on that
worktree or the deprecated modules. The source remains unchanged.

## Component ownership

| Source file | Core destination |
| --- | --- |
| `processes/simplex_math.py` | `processes/simplex_math.py`: copied log-space Gamma/Beta/Dirichlet sampling and reverse-bridge equations |
| `processes/corruption.py:SimplexCorruptionProcess` | `processes/dirichlet.py`: forward corruption, clean faces, reference prior, explicit reverse transition |
| Concentration function in `simplex_math.py` | `noise_schedules/simplex.py`: physical coefficients, with `alpha=1-t` |
| `schedules/adaptive_time.py` | `time_sampling/loss_derivative.py`: copied binned adaptive sampling and checkpointed observations |
| `schedules/grids.py` | `time_sampling/simplex_grid.py`: linear, cosine, or fitted quantile inference clock |
| Simplex input code in `algos/simplex.py` | `models/simplex.py`: expectation or argmax input using the shared DiT backbone |
| `samplers/simplex.py` | `samplers/simplex.py` and `simplex_utils.py`: copied sampling loop, guidance, confidence commits |
| `algos/simplex.py` and `SimplexObjective` | `algorithms/simplex_diffusion.py`: orchestration using the existing posterior and cross-entropy objective |

The process owns the reference distribution and receives schedule coefficients
explicitly. It never reaches into the algorithm. The sampler owns the loop,
clean categorical draws, clock, churn, optional confidence commits, and local
prediction sessions. Neither the model nor algorithm contains bridge math.

The sampler state is always **float64 log probabilities**. Exact zero-probability
faces remain `-inf`; there is no epsilon smoothing. Only the model input is
converted to neural-network precision. Expectation input computes `exp(log_P) @ E`;
argmax input looks up the largest coordinate. Changing the input adapter never
replaces the full state stored by the sampler. Dropped conditioning positions use
all-`-inf` states, mapped to zero embeddings by either adapter.

At time zero, corruption returns the exact clean vertex without sampling. This
also handles the divergent concentration of `epsilon/t` at that endpoint.
The reverse bridge applies only for `0 < s < t <= 1`; the final sampling step
uses a categorical clean-token draw, exactly as in the source.

## Configurations

```bash
python main.py +experiment=sudoku_simplex
python main.py +experiment=sudoku_simplex_argmax
python main.py +experiment=sudoku_simplex_self_conditioning
python main.py +experiment=sudoku_simplex_adaptive
```

The baseline uses expected-embedding input, float32 neural computation,
constant-then-linear normalized variance (`nu0=.4`, `nu1=.75`, `pivot=.8`),
180 sampling calls, churn 1, and training time epsilon `1e-5`.

Other concentration settings are explicit overrides:

```bash
python main.py +experiment=sudoku_simplex noise_schedule.kind=constant noise_schedule.nu0=0.4
python main.py +experiment=sudoku_simplex noise_schedule.kind=ddim noise_schedule.epsilon=1.0
python main.py +experiment=sudoku_simplex sampler.churn=0.2 sampler.time_grid=cosine
```

`ddim` retains the source name for the **concentration function** `c(t)=epsilon/t`;
it does not switch the sampler to Gaussian DDIM.

The adaptive preset uses 50 loss bins, a 65,536-example observation buffer,
1,000 clock grid points, warmup 1,000 optimizer steps, refit every 50 steps, and
density momentum .9. The clock changes where training examples are drawn;
it does not change the physical corruption formula or add importance weighting.
Validation does not update the profile. As in the reference training runs,
the adaptive preset keeps a linear inference grid for training callbacks.
Adaptive inference can be selected separately with `sampler.time_grid=adaptive`
when evaluating a fitted checkpoint. An adaptive inference grid requires a
fitted profile and raises an informative error before that point. For an early
smoke check keep `sampler.time_grid=linear`. All adaptive buffers are checkpointed.

## Self-conditioning and optional sampling features

Hidden self-conditioning uses the existing shared component: detached bootstrap,
per-example probability .9 during training, zero-initialized affine LayerNorm,
and direct hidden addition. Sampling reuses the preceding hidden state and adds
no bootstrap evaluations. The source implementation's global model wrapper is
not copied. Each generation call owns fresh local feedback; classifier-free
conditional and unconditional calls have separate sessions.

Expected-embedding self-conditioning also works by setting
`self_conditioning.source=posterior_mean`. This is an additional core option;
the copied simplex reference uses hidden self-conditioning.

Classifier-free guidance requires nonzero `algorithm.condition_dropout` during
training and a conditioning prompt at generation:

```bash
python main.py +experiment=sudoku_simplex_self_conditioning algorithm.condition_dropout=0.1 sampler.guidance_scale=2.0
```

The copied guidance interpolates log probabilities. Nontrivial guidance needs
finite log probabilities, so top-k/top-p filtering that introduces infinities
is rejected with a clear error. Temperature and softcapping remain predictor
settings. `sampler.confidence_power=1.0` enables the copied confidence commits,
with prompt tokens excluded from the committed count. Guidance doubles NFE;
confidence commits do not. The final confidence-mode prediction is greedy, as
in the source.

## Protocol boundaries

These presets use **current core Sudoku Hard**: 48,000 training puzzles,
2,000 held-out puzzles, 12 vocabulary entries, and 180-token sequences.
They do not silently adopt the source's paper protocol: 200,000/5,000 puzzles,
14-token serialization, fixed-decay EMA, or 50,000-step training budget
(38,462 with self-conditioning). Core's existing optimizer, EMA, training-budget,
and task defaults remain in effect. Matching paper dataset sizes alone would
not reproduce the source serialization or EMA policy.

The source's separate `simplex_dmd` teacher/student distillation extension is
**not included**. This port covers ordinary simplex diffusion, its adaptive
clock, hidden feedback, and optional CFG/confidence sampler.

## Validation

Copied regression tests check oracle reverse-bridge mean/variance for all three
concentration families and churn 0/.2/1; tiny Gamma shapes; exact faces; known
non-uniform data recovery through a full exact-Bayes chain; expected/argmax
inputs with and without hidden self-conditioning; prompt preservation; NFE;
adaptive checkpoint restoration; and independent CFG feedback sessions.

A direct same-seed comparison with the source gives identical float64 outputs
(`atol=rtol=0`) for log-Gamma, log-Beta, log-Dirichlet, and all nine tested
concentration/churn bridge combinations. The compact evidence is in
`core/outputs/simplex_port_validation/source_parity.json`.
These mathematical and execution checks do not establish trained Sudoku quality
or a paper reproduction.

Validation on the port: 34 simplex tests passed with CUDA enabled, including
four CUDA forward/backward and generation cases. The full core CPU suite passed
320 tests with 17 skipped before those four CUDA-only cases were added. Ruff
passed for the changed Python files.
