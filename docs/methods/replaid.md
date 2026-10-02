# Continuous RePlaid

RePlaid specializes Gaussian diffusion with learned unit embeddings, a learned negative-log-SNR schedule, a likelihood objective, adaptive reconstruction/diffusion allocation, and specific gradient routes. It is not ordinary CE Gaussian diffusion with a different preset.

```bash
uv run python main.py +experiment=replaid_sudoku_hard
uv run python main.py +experiment=replaid_sudoku_hard trainer.max_steps=2000 checkpoint.every_n_steps=2001
```

The corrected preset follows the successful historical implementation's construction, not a claim of paper-scale reproduction.

## Components

| Component | Responsibility |
|---|---|
| Representation | Trainable unit embeddings, dimension 16 by default, normalized reads and post-update projection |
| Learned schedule | Monotone negative-log-SNR gamma, derivatives, VP coefficients, learned endpoints, shape adaptation |
| Time sampler | Reconstruction/diffusion batch allocation and moment statistics |
| Gaussian-prior predictor | Explicit output-prior correction, followed by ordinary posterior pathways |
| Objective | Reconstruction CE, weighted embedding MSE, terminal prior KL, allocation factors |
| Algorithm | Gradient routing, adaptation observations, bootstrap selection |
| Gaussian DDPM sampler | Reverse loop, prompt policy, feedback policy, final decode |

Let $\gamma=-\log(\mathrm{SNR})$. The VP coefficients satisfy $\alpha^2=\operatorname{sigmoid}(-\gamma)$ and $\sigma^2=\operatorname{sigmoid}(\gamma)$. The default schedule is a 1,024-unit positive-weight tanh network with independent endpoints initially -13.3 and 5.0. It is **learned**, not a fixed tanh interpolation. Schedule calculations and parameters use float64. The older softplus implementation remains an explicit option, not the current default.

For unit embeddings, the predictor adds

$$\ell_v=f_\theta(z_t,t)_v+\frac{\alpha_t}{\sigma_t^2}z_t^T e_v.$$

The algorithm supplies the state, coefficients, and table; the predictor never reaches into the representation. Training does not apply temperature or filtering.

## Loss and adaptation

The objective combines reconstruction CE at time zero, diffusion embedding MSE weighted by $-\tfrac12\,d\mathrm{SNR}/dt$, and terminal Gaussian prior KL. Terms and final reductions use float64. The batch allocation rescales reconstruction and diffusion contributions to account for their sample counts.

The time sampler allocates at least one example to each estimator according to standard-deviation estimates from checkpointed first/second-moment EMAs (momentum .99). Diffusion times are stratified. Singleton batches draw both estimators while reporting the original token count. Statistics respect the loss mask and remain meaningful when the reconstruction group contains one example.

Interior schedule shape receives the variance-minimizing gradient multiplier; endpoints retain ordinary loss gradients. Reconstruction and self-conditioned rows do not contribute to shape adaptation. The algorithm supplies these observations explicitly; schedule and sampler do not invoke the loss themselves.

## Self-conditioning and architecture

The preset selects expected-embedding [self-conditioning](../modifiers/self-conditioning.md), fixed-count bootstrap selection `ceil(.25 * batch_size)`, and all rows during validation. Selected rows receive a detached bootstrap. Their main prediction still updates the denoiser, while schedule and embedding gradient paths are detached selectively. Terminal prior KL retains its ordinary gradient path.

This .25 probability and ordinary initialization of the mean-feedback projection deliberately override generic self-conditioning defaults. Inputs use unit-embedding scaling `z / sqrt(alpha²/d + sigma²)` and feedback is scaled by `sqrt(d)`. Autocast weight caching is disabled where needed to prevent detached bootstrap casts affecting the main pass.

The backbone uses direct input projection, additive projected time embedding, and an AdaLN vocabulary head. Embedding/schedule-shape parameter groups use learning rate .01 without decay; endpoints use .01 with weight decay .1. Other weights use the normal optimizer settings. The update order is optimizer, EMA, then representation projection, preserving the reference ordering.

## Sampling and checkpoints

The generic preset corrupts all positions and uses noisy prompt inpainting. `replaid_sudoku_hard` instead preserves clean puzzle prompts in training and sampling (`algorithm.corrupt_all=false`, `sampler.prompt_mode=clean`), including bootstrap mean feedback.

The DDPM sampler uses fresh bootstrap conditioning and final-transition noise. With conditioning enabled, NFE is `2 * (steps + 1)`, or 362 for 180 transitions. Its default prior is standard normal; `sampler.prior_mode=schedule` is available for other suitable Gaussian paths.

Schedule parameters, allocation buffers, representation, model, EMA, optimizer, and RNG state use normal checkpointing. Old softplus checkpoints require their saved configuration; new schedule and moment states are not interchangeable. Per-call hooks and sampling feedback are transient.

## Validation limits

Permanent tests cover coefficients and derivatives, gradient ownership, output-prior correction, masked reductions, allocation, projection/EMA order, reverse limits, checkpoint restoration, and CUDA updates. Historical mapped-weight audits also compared the corrected reference's predictions and full gradients.

An older 2,000-step comparison reported zero Sudoku solves for both implementations, but used the superseded softplus construction. Those results do **not** evaluate the corrected preset. The successful historical run used 50,000 steps and 200k/10k data; current Sudoku defaults use 48k/2k. Core also retains FP32 input/time projections, prior dot products, and expected-embedding conversion where the old ambient BF16 policy differed. Consult [recorded results](../validation/reproduction-results.md) for run-specific evidence rather than treating these configurations as full-budget reproduction.

Sources: [algorithm](../../core/algorithms/replaid.py), [objective](../../core/objectives/replaid.py), [schedule](../../core/noise_schedules/learned_log_snr.py), [allocation](../../core/time_sampling/adaptive_allocation.py).
