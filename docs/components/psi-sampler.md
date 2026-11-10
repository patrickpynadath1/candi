# Ψ-samplers

`core/samplers/psi.py` implements the Ψ-samplers of *The Diffusion Duality, Chapter II:
Ψ-Samplers and Efficient Curriculum* (arXiv 2602.21185, Eq. 12; Suppl. A.4-A.5) as a
transition policy for `ComposedSampler`, generalised from discrete views to every
corruption. Read this way, Ψ formalises churn: one rule for discrete and continuous
states, with no new algorithm class.

## Update

At every step, with predicted clean token x̂_θ:

    Ψ_{s|t} = κ_t · (native reverse step z_t -> z_s given x̂_θ)
            + (1 - κ_t) · (forward marginal q_s(z_s | x̂_θ))

The mixture is drawn per position. Each generated position takes the view's native
reverse update with probability κ_t. Otherwise it is re-noised from x̂_θ to time s with
the view's forward kernel `corrupt_state`, exactly as `ForwardResamplingSampler` does.

- **Discrete views.** The forward branch is α_s · x̂ + (1 - α_s) · π, with π the mask
  token (absorbing) or uniform 1/K. This is the paper's
  α_s q_{0|t}(· | z_t, x_θ) + (1 - α_s) π, with x̂ ~ x_θ (MDLM's carry-over
  parameterisation keeps revealed tokens). Revealed tokens can re-mask only when κ < 1;
  uniform views keep full support.
- **Gaussian, simplex, spherical, vMF and blurring views.** The forward branch re-noises
  the prediction with that process's own kernel.
- **Hybrids.** One κ_t, one x̂_θ (sampled, or argmax with `clean: greedy`) and one branch
  choice per position are shared by all views.

Exact limits:

- κ_t = 1 is the native sampler, token for token for the same seed. Ψ then draws no
  random numbers of its own.
- κ_t = 0 is forward re-sampling, token for token for the same seed.

Prompts are clamped as in every composed sampler.

## κ schedules

κ_t comes from a ReMDM remasking scale σ_t (Suppl. A.4): κ_t = 1 - σ_t / (1 - α_s), with
σ_t^max = min{1, (1 - α_s) / α_t} (σ_t^max = 1 when α_t = 0).

| `schedule` | σ_t / κ_t |
| --- | --- |
| `rescale` | σ_t = η · σ_t^max |
| `cap` | σ_t = min{η, σ_t^max} |
| `loop` | σ_t = η on the hold, 0 elsewhere |
| `constant` | κ_t = 1 - η at every step (the paper's image setting, κ = 0.95) |

α_t is the clean fraction of a reference view's schedule (`keep_rate`):

- the keep rate of the first discrete view, if there is one;
- otherwise the first view's: Gaussian α²/(α² + σ²) (the signal-variance fraction),
  simplex α, and 1 - t for spherical, vMF and blurring paths, whose coefficients have no
  keep-rate reading.

`loop` also reshapes the time grid (`LoopTimeWarp`, applied by the sampler loop). The
reference clean fraction follows the ReMDM-loop profile:

- it rises linearly from 0 to `alpha_on` on (t_on, 1];
- it is held at `alpha_on` on [t_off, t_on];
- it rises linearly to 1 on [0, t_off).

Schedule times for these targets are found by bisection, so the network is conditioned
on the warped times. On the hold α_s = α_t, the native step is the identity and Ψ is a
pure corrector with κ = 1 - η / (1 - α_on). `alpha_on` (default 0.9, as ReMDM's scripts)
is the paper's α(t_on).

## Configuration and budget

```bash
uv run --locked python main.py +experiment=mdlm inference/sampler=psi \
  sampler.transition.schedule=rescale sampler.transition.eta=0.02 algorithm.predictor.top_p=0.9
uv run --locked python main.py +experiment=duo inference/sampler=uniform_psi
```

- `psi.yaml` matches `hybrid` and `forward_resampling`: greedy terminal decode,
  epsilon 1e-5, float32 probabilities. Set `sampler.precision=float64` to match
  `ancestral` exactly.
- `uniform_psi.yaml` matches `uniform_ancestral`.
- Commitments are refused (`commitment=none` only).
- Every step makes one network call and the terminal decode one more, so `steps_for_nfe`
  and the measured NFE equal the native sampler's.
- Nucleus sampling (the paper's p = 0.9) is the predictor's `top_p`.

The sweep policy `psi` (`configs/inference/inference_policies/psi.yaml`) tunes:

- the schedule (`rescale`, `cap`, `loop`);
- η in [0.005, 0.1] (log);
- loop t_on in [0.5, 1] and t_off in [0.05, 0.2], only when the schedule is `loop`;
- top_p in {0.9, 0.95, 1};
- the temperature.

It starts from the paper's language setting (rescale, η = 0.02 and 0.01). The `uniform_psi`
policy is the same for DUO-style presets.

Tests (`core/tests/test_psi_sampler.py`), for Gaussian, simplex, spherical, vMF, blurring,
masked and uniform views, two hybrids, MDLM and DUO:

- κ = 1 equals the native sampler and κ = 0 equals `ForwardResamplingSampler` (same seed);
- prompts and NFE are unchanged under every schedule;
- revealed tokens re-mask only for κ < 1, and the uniform re-noising keeps full support;
- the κ formulas, monotone clean fractions and the loop plateau.
