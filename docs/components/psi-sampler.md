# Ψ-samplers

`core/samplers/psi.py` implements the Ψ-samplers of *The Diffusion Duality, Chapter II:
Ψ-Samplers and Efficient Curriculum* (arXiv 2602.21185, Eq. 12; Suppl. A.4-A.5) as a
transition policy for `ComposedSampler`. They apply to masked (MDLM) and uniform-state
(DUO/UDLM) recipes and to the discrete view of a hybrid; no algorithm class changes.

## Update

For a discrete view with source keep rate α_t, destination keep rate α_s, prediction
x_θ and noise prior π (the mask token for absorbing views, uniform 1/K for uniform
views), each step draws from

    Ψ_{s|t} = κ_t · q_{s|t}(· | z_t, x_θ) + (1 - κ_t) · [α_s · q_{0|t}(· | z_t, x_θ) + (1 - α_s) · π]

`q_{s|t}` is the view's own ancestral posterior (the absorbing posterior, or the uniform
plug-in / posterior bridge of `UniformProcess` / `PosteriorUniformProcess`);
`q_{0|t}(· | z_t, x_θ)` is x_θ, with absorbing views carrying revealed tokens over
(Suppl. A.1). κ_t = 1 is exactly the ancestral sampler: the transition then calls the
view's native update, so the same seed gives the same tokens. For κ_t < 1, revealed
masked-view tokens can be re-masked and uniform views keep full support. Continuous views
of a hybrid always use their native update. Prompt positions are clamped as in every
composed sampler.

## κ schedules

κ_t comes from a ReMDM remasking scale σ_t (Suppl. A.4): κ_t = 1 - σ_t / (1 - α_s), with
σ_t^max = min{1, (1 - α_s) / α_t} (σ_t^max = 1 when α_t = 0).

| `schedule` | σ_t |
| --- | --- |
| `rescale` | η · σ_t^max |
| `cap` | min{η, σ_t^max} |
| `loop` | η on the hold, 0 (plain posterior) elsewhere |

`loop` also reshapes the time grid (`LoopTimeWarp`, owned by the sampler loop): the keep
rate follows the ReMDM-loop profile, rising linearly from 0 to `alpha_on` on (t_on, 1],
held at `alpha_on` on [t_off, t_on], and rising linearly to 1 on [0, t_off). Schedule
times for the target keep rates are found by bisection on the discrete view's schedule,
so the network is conditioned on the warped times. On the hold α_s = α_t = `alpha_on`, the
posterior is the identity and the step is a pure corrector with κ = 1 - η / (1 - α_on).
`alpha_on` (default 0.9, as ReMDM's scripts) is the α(t_on) of the paper.

## Configuration and budget

```bash
uv run --locked python main.py +experiment=mdlm inference/sampler=psi \
  sampler.transition.schedule=rescale sampler.transition.eta=0.02 algorithm.predictor.top_p=0.9
uv run --locked python main.py +experiment=duo inference/sampler=uniform_psi
```

`psi.yaml` matches `ancestral` (greedy terminal decode, float64 probabilities);
`uniform_psi.yaml` matches `uniform_ancestral` (epsilon 1e-3, ancestral terminal step).
Commitments are refused (`commitment=none` only). Every step makes one network call and
the terminal decode one more, so `steps_for_nfe` and measured NFE equal the ancestral
sampler's. Nucleus sampling (the paper's p = 0.9) is the predictor's `top_p`.

The sweep policies `psi` and `uniform_psi` (`configs/inference/inference_policies/`) tune
the schedule, η in [0.005, 0.1] (log), loop t_on in [0.5, 1] and t_off in [0.05, 0.2]
(only when the schedule is `loop`), top_p in {0.9, 0.95, 1} and the temperature, starting
from the paper's language setting (rescale, η = 0.02 and 0.01).

Tests: `core/tests/test_psi_sampler.py` (κ = 1 equals ancestral token for token, prompts and
NFE unchanged, re-masking only for κ < 1, full uniform support, the κ formulas, the loop
plateau, refusals).
