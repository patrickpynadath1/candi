# Simplex diffusion

Simplex diffusion represents each token by a point on the probability simplex and corrupts it with a Dirichlet distribution. The port copied the working reference's log-space Gamma/Beta/Dirichlet and reverse-bridge mathematics, then separated the process, schedule, clock, model adapter, and sampler. Production code has no dependency on the former worktree.

The state is **float64 log probabilities**. Exact zero-probability faces remain `-inf`; the state is not epsilon-smoothed. The model alone converts this to neural precision, using either expected embeddings `exp(log_P) @ E` or an argmax token lookup. Both modes retain the complete log-space state for sampling.

## Assembly and recipes

The Dirichlet process owns the reference categorical distribution and forward/reverse transitions. The concentration schedule supplies physical coefficients with clean weight `1-t`. The model predicts a clean-token posterior and the objective is CE. The sampler draws clean tokens and calls the bridge with explicit source/destination coefficients.

```bash
uv run python main.py +experiment=sudoku_simplex
uv run python main.py +experiment=sudoku_simplex_argmax
uv run python main.py +experiment=sudoku_simplex_self_conditioning
uv run python main.py +experiment=sudoku_simplex_adaptive
```

Defaults use expected-embedding input, float32 neural computation, constant-then-linear normalized variance (`nu0=.4`, `nu1=.75`, `pivot=.8`), churn 1, and 180 sampling calls. Corruption at time zero returns an exact clean vertex. Reverse bridges operate for `0 < s < t <= 1`; final decoding draws a clean token.

```bash
uv run python main.py +experiment=sudoku_simplex noise_schedule.kind=constant noise_schedule.nu0=0.4
uv run python main.py +experiment=sudoku_simplex noise_schedule.kind=ddim noise_schedule.epsilon=1.0
uv run python main.py +experiment=sudoku_simplex sampler.churn=0.2 sampler.time_grid=cosine
```

`ddim` names the concentration function `c(t)=epsilon/t`; it does not select a Gaussian DDIM sampler.

## Adaptive clock

The adaptive preset uses 50 loss bins, a 65,536-example observation buffer, 1,000 grid points, 1,000-step warmup, refitting every 50 steps, and density momentum .9. It changes the distribution of training times without changing physical corruption or adding importance weighting. Validation does not update it; buffers are checkpointed.

Training callbacks retain the reference's linear inference grid. Set `sampler.time_grid=adaptive` separately when evaluating a fitted checkpoint. It fails explicitly if no profile has been fitted yet. See [time sampling](../components/time-sampling.md).

## Conditioning and guidance

Hidden [self-conditioning](../modifiers/self-conditioning.md) uses the shared modifier: detached bootstrap, probability .9, zero-initialized affine LayerNorm, and direct hidden addition. Sampling reuses previous feedback. Expected-embedding feedback is another core option; it is not the copied reference default.

```bash
uv run python main.py +experiment=sudoku_simplex_self_conditioning algorithm.condition_dropout=0.1 sampler.guidance_scale=2.0
```

Classifier-free guidance requires trained condition dropout and a generation prompt. Conditional and unconditional paths have separate feedback sessions. Guidance interpolates log probabilities; nontrivial guidance rejects filtering that creates nonfinite probabilities. It doubles NFE. Optional `sampler.confidence_power=1.0` enables confidence commits, excluding prompt tokens from the committed count; confidence-mode final decoding is greedy.

## Validation and boundaries

Same-seed float64 Gamma, Beta, Dirichlet, and nine concentration/churn bridge fixtures matched the source exactly. Regression checks cover bridge moments, tiny shapes, exact faces, exact-Bayes chains, both model adapters, adaptive state restoration, guidance, and CUDA generation. These checks do not establish trained standalone Sudoku quality.

Core Sudoku uses its 48k/2k split, 12-token vocabulary, and 180-token sequences. The source paper protocol's 200k/5k data, 14-token serialization, fixed-decay EMA, and larger budget were not silently adopted. The separate `simplex_dmd` distillation extension is not ported. See [hybrids](hybrids.md) for the trained uniform-plus-simplex check.

Sources: [algorithm](../../core/algorithms/simplex_diffusion.py), [Dirichlet process](../../core/processes/dirichlet.py), [copied mathematics](../../core/processes/simplex_math.py), [sampler](../../core/samplers/simplex.py).
