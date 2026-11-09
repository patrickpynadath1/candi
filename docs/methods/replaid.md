# Continuous RePlaid

RePlaid is Gaussian diffusion with learned unit embeddings, a learned negative-log-SNR schedule, a likelihood objective, adaptive reconstruction/diffusion allocation, and specific gradient routes. There is no RePlaid algorithm class: the `replaid` presets are recipes over the one-view [`GaussianDiffusion`](gaussian-diffusion.md) path, and each of those choices is a component selected by config.

```bash
uv run python main.py +experiment=replaid_sudoku_hard
uv run python main.py +experiment=replaid_sudoku_hard trainer.max_steps=2000 checkpoint.every_n_steps=2001
```

The corrected preset follows the successful historical implementation's construction, not a claim of paper-scale reproduction.

## Components

| Recipe part | Config | Responsibility |
|---|---|---|
| Representation | `information_path/representation=embedding` (`dimension: 16`, `trainable`, `project_after_update`) | Trainable unit embeddings, normalized reads and post-update projection |
| Noise schedule | `information_path/noise_schedule=plaid_log_snr` with `adaptation_term: diffusion` | Monotone negative-log-SNR gamma, derivatives, VP coefficients, learned endpoints, variance adaptation of the shape |
| Time sampler | `denoising_task/time_sampler=adaptive_allocation` | Reconstruction/diffusion batch allocation (`TimeAllocation` draws), stratified times, moment statistics |
| Parameterization | `denoising_task/parameterization=gaussian_prior` | Gaussian-prior logit correction, then the ordinary posterior pathways |
| Objective | `denoising_task/objective=gaussian_elbo` | Reconstruction CE, weighted embedding MSE, terminal prior KL, allocation factors |
| Algorithm | `denoising_task/algorithm=gaussian_diffusion` (`corrupt_all`) | The shared one-view loss; no RePlaid-specific code |
| Self-conditioning | `prediction.selection=fixed_count`, `selected_gradients=denoiser_only`, `fixed_feedback=clean` | Bootstrap selection, gradient routing of selected rows, clean prompt feedback |
| Sampler | `inference/sampler=gaussian_ddpm` | Reverse loop, prompt policy, feedback policy, final decode |

Let $\gamma=-\log(\mathrm{SNR})$. The VP coefficients satisfy $\alpha^2=\operatorname{sigmoid}(-\gamma)$ and $\sigma^2=\operatorname{sigmoid}(\gamma)$. The default schedule is a 1,024-unit positive-weight tanh network with independent endpoints initially -13.3 and 5.0. It is **learned**, not a fixed tanh interpolation. Schedule calculations and parameters use float64. The older softplus implementation (`learned_log_snr`) remains an explicit option, not the current default.

For unit embeddings, the parameterization adds

$$\ell_v=f_\theta(z_t,t)_v+\frac{\alpha_t}{\sigma_t^2}z_t^T e_v.$$

`HybridDiffusion.prediction_transform` supplies the state, coefficients and codebook of the parameterization's `prior_view` to `GaussianPriorPosterior.corrected_logits` before every training, sampling and bootstrap pathway; the predictor never reaches into the representation. Training does not apply temperature or filtering.

## Loss and adaptation

[`GaussianELBO`](../../core/objectives/gaussian_elbo.py) reads everything from `ObjectiveContext`: the view's path, representation and schedule, the time draw and the detached rows. It combines reconstruction CE at time zero, diffusion embedding MSE weighted by $-\tfrac12\,d\mathrm{SNR}/dt$ (computed from the path's $\alpha,\sigma$ and their derivatives), and the terminal Gaussian prior KL (the schedule at $t=1$). Terms and final reductions use float64. The allocation rescales reconstruction and diffusion rows by $B/r$ and $B/(B-r)$. The objective returns an `ObjectiveResult` whose named terms (`reconstruction`, `diffusion`, `prior`) the adaptive components observe.

The time sampler allocates at least one example to each estimator according to standard-deviation estimates from checkpointed first/second-moment EMAs (momentum .99), observed from the `reconstruction` and `diffusion` terms (`observe_terms`). Diffusion times are stratified. It declares `min_batch = 2`, so the shared loss replicates a singleton batch and reports the original token count and loss sum. Statistics respect the loss mask and remain meaningful when the reconstruction group contains one example.

Interior schedule shape receives the variance-minimizing gradient multiplier $2L_b$ ($L_b$ the masked mean of the `diffusion` term): the schedule's `training_coefficients` registers the hook and `observe_terms` sets it; reconstruction rows and self-conditioned rows get zero. Endpoints retain ordinary loss gradients. The algorithm supplies these observations; schedule and sampler do not invoke the loss themselves.

## Self-conditioning and architecture

The preset selects expected-embedding [self-conditioning](../modifiers/self-conditioning.md), fixed-count bootstrap selection `ceil(.25 * batch_size)`, and all rows during validation. Selected rows receive a detached bootstrap. With `prediction.selected_gradients=denoiser_only` their main prediction still updates the denoiser, while their schedule coefficients, clean encodings, prior-correction table reads and objective embeddings are detached. Terminal prior KL retains its ordinary gradient path.

This .25 probability and ordinary initialization of the mean-feedback projection deliberately override generic self-conditioning defaults. Inputs use unit-embedding scaling `z / sqrt(alpha²/d + sigma²)` and feedback is scaled by `sqrt(d)`. Autocast weight caching is disabled where needed to prevent detached bootstrap casts affecting the main pass.

The backbone uses direct input projection, additive projected time embedding, and an AdaLN vocabulary head. Embedding/schedule-shape parameter groups use learning rate .01 without decay; endpoints use .01 with weight decay .1. Other weights use the normal optimizer settings. The update order is optimizer, EMA, then representation projection, preserving the reference ordering.

## Sampling and checkpoints

The generic preset corrupts all positions (`algorithm.corrupt_all=true`) and uses noisy prompt inpainting. `replaid_sudoku_hard` instead preserves clean puzzle prompts in training and sampling (`algorithm.corrupt_all=false`, `sampler.prompt_mode=clean`); `prediction.fixed_feedback=clean` makes the training bootstrap feed back clean prompt embeddings, as the sampler does.

The DDPM sampler uses fresh bootstrap conditioning and final-transition noise. With conditioning enabled, NFE is `2 * (steps + 1)`, or 362 for 180 transitions. Its default prior is standard normal; `sampler.prior_mode=schedule` is available for other suitable Gaussian paths.

Schedule parameters, allocation buffers, representation, model, EMA, optimizer, and RNG state use normal checkpointing. State names are unchanged from the former `RePlaid` class (a GaussianDiffusion subclass), so its checkpoints load directly; saved run configs naming `core.algorithms.replaid.RePlaid` are rebuilt from the `replaid` preset on load (keeping the saved `corrupt_all`, schedule, model sizes and sampler; see [hybrid checkpoints](replaid-hybrid.md#checkpoints)). Old softplus checkpoints require their saved configuration; new schedule and moment states are not interchangeable. Per-call hooks and sampling feedback are transient.

## Validation limits

Permanent tests cover coefficients and derivatives, gradient ownership, output-prior correction, masked reductions, allocation, projection/EMA order, reverse limits, checkpoint restoration, and CUDA updates (`core/tests/test_replaid.py`, `core/tests/test_replaid_components.py`). Historical mapped-weight audits also compared the corrected reference's predictions and full gradients. The move from the `RePlaid` class to this recipe was checked by fixing all random variates and comparing losses, every parameter gradient, allocation buffers and a three-step generation (agreement to float rounding).

An older 2,000-step comparison reported zero Sudoku solves for both implementations, but used the superseded softplus construction. Those results do **not** evaluate the corrected preset. The successful historical run used 50,000 steps and 200k/10k data; current Sudoku defaults use 48k/2k. Core also retains FP32 input/time projections, prior dot products, and expected-embedding conversion where the old ambient BF16 policy differed. Consult [recorded results](../validation/reproduction-results.md) for run-specific evidence rather than treating these configurations as full-budget reproduction.

Sources: [preset](../../configs/experiment/replaid.yaml), [objective](../../core/objectives/gaussian_elbo.py), [schedule](../../core/noise_schedules/learned_log_snr.py), [allocation](../../core/time_sampling/adaptive_allocation.py), [shared loss](../../core/modifiers/hybrid/training.py).
