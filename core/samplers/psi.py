"""Ψ-sampler transitions: ancestral posteriors mixed with forward re-noising.

From "The Diffusion Duality, Chapter II: Ψ-Samplers and Efficient Curriculum"
(arXiv 2602.21185, Eq. 12 and Suppl. A.4-A.5). For a discrete view with keep
rates α_t (source) and α_s (destination), prediction x_θ and noise prior π:

    Ψ_{s|t} = κ_t q_{s|t}(· | z_t, x_θ) + (1 - κ_t) [α_s q_{0|t}(· | z_t, x_θ) + (1 - α_s) π]

π is the mask token for absorbing views and uniform 1/K for uniform views;
q_{0|t}(· | z_t, x_θ) is x_θ, with absorbing views carrying unmasked tokens over.
κ_t = 1 is the ancestral sampler. κ_t comes from a ReMDM remasking scale σ_t:
κ_t = 1 - σ_t / (1 - α_s), σ_t^max = min{1, (1 - α_s) / α_t}, with

- ``rescale``: σ_t = η σ_t^max;
- ``cap``:     σ_t = min{η, σ_t^max};
- ``loop``:    the sampler's time grid is warped (``LoopTimeWarp``) so α_t is piecewise
               linear: 0 -> α_on on (t_on, 1], held at α_on on [t_off, t_on], α_on -> 1
               on [0, t_off). Outside the hold, plain posterior steps (κ = 1); inside it
               (α_s = α_t = α_on) σ_t = η, i.e. κ = 1 - η / (1 - α_on).

This is a transition policy for ``ComposedSampler``; time progression, stopping
and NFE accounting stay in the sampler (one network call per step, as ancestral).
Continuous views of a hybrid recipe keep their native reverse update.
"""

import math

import torch

from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform import UniformProcess
from .policies import ReverseTransition
from .utils import sample_categorical

SCHEDULES = ("rescale", "cap", "loop")


def _rows(value, like):
    """Broadcast per-row keep rates over the trailing (length, vocab) axes."""
    while value.ndim < like.ndim:
        value = value.unsqueeze(-1)
    return value.to(like)


class KappaSchedule:
    def __init__(self, schedule="rescale", eta=0.02, t_on=0.55, t_off=0.05, alpha_on=0.9):
        if schedule not in SCHEDULES:
            raise ValueError(f"kappa schedule must be one of {SCHEDULES}")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must lie in [0, 1]")
        if schedule == "loop" and not (0 < t_off < t_on <= 1 and 0 < alpha_on < 1):
            raise ValueError("loop needs 0 < t_off < t_on <= 1 and alpha_on in (0, 1)")
        self.schedule, self.eta = schedule, float(eta)
        self.t_on, self.t_off, self.alpha_on = float(t_on), float(t_off), float(alpha_on)

    def __call__(self, keep_t, keep_s):
        """κ per row from source and destination keep rates (float64)."""
        keep_t, keep_s = keep_t.double(), keep_s.double()
        room = (1 - keep_s).clamp_min(0)
        if self.schedule == "loop":
            # The warped grid makes the hold phase exactly the steps with α_s = α_t.
            hold = torch.isclose(keep_s, keep_t, rtol=0, atol=1e-9) & (room > 0)
            sigma = torch.where(hold, torch.full_like(room, self.eta), torch.zeros_like(room))
            sigma = torch.minimum(sigma, room)
        else:
            sigma_max = torch.where(
                keep_t > 0, (room / keep_t.clamp_min(1e-300)).clamp(max=1), torch.ones_like(room)
            )
            sigma = self.eta * sigma_max if self.schedule == "rescale" else sigma_max.clamp(max=self.eta)
        kappa = torch.where(room > 0, 1 - sigma / room.clamp_min(1e-300), torch.ones_like(room))
        return kappa.clamp(0, 1)

    def warp(self, steps):
        return LoopTimeWarp(self.t_on, self.t_off, self.alpha_on) if self.schedule == "loop" else None


class PsiTransition(ReverseTransition):
    """Ψ transition for absorbing and uniform views; native updates elsewhere."""

    requires_tokens = False

    def __init__(self, schedule="rescale", eta=0.02, t_on=0.55, t_off=0.05, alpha_on=0.9):
        super().__init__()
        self.kappa = KappaSchedule(schedule, eta, t_on, t_off, alpha_on)

    @property
    def time_warp(self):
        return self.kappa.warp(None)

    def validate(self, views, commitment):
        super().validate(views, commitment)
        if commitment != "none":
            raise ValueError("Ψ transitions re-noise committed tokens; use commitment=none")
        if not any(isinstance(v.process, (AbsorbingProcess, UniformProcess)) for v in views):
            raise TypeError("Ψ-sampling needs a masked (absorbing) or uniform discrete view")

    def probabilities(self, view, state, probabilities, source, destination):
        """Ψ_{s|t} over the view's state values, or None for κ = 1 everywhere."""
        process = view.process
        if not isinstance(process, (AbsorbingProcess, UniformProcess)):
            return None
        keep_t, keep_s = source.coefficients[0], destination.coefficients[0]
        kappa = self.kappa(keep_t, keep_s)
        if bool((kappa == 1).all()):
            return None
        if isinstance(process, AbsorbingProcess):
            mask = process.mask_index
            if probabilities.shape[-1] == mask:
                probabilities = torch.nn.functional.pad(probabilities, (0, 1))
            probabilities = probabilities.double()
            observed = torch.nn.functional.one_hot(state, probabilities.shape[-1]).double()
            masked = state.eq(mask)[..., None]
            at, ass = _rows(keep_t, probabilities), _rows(keep_s, probabilities)
            # q_{s|t}: a masked token unmasks to x_θ w.p. (α_s - α_t)/(1 - α_t); others stay.
            posterior = probabilities * (ass - at) / (1 - at).clamp_min(1e-300)
            posterior[..., mask] = ((1 - ass) / (1 - at).clamp_min(1e-300))[..., 0]
            posterior = torch.where(masked, posterior, observed)
            clean = torch.where(masked, probabilities, observed)  # q_{0|t}: carry-over
            prior = torch.nn.functional.one_hot(torch.full_like(state, mask), probabilities.shape[-1])
        elif isinstance(process, UniformProcess):
            probabilities = probabilities.double()
            posterior = process.reverse_probabilities(state, probabilities, keep_t, keep_s).double()
            clean = probabilities
            prior = torch.full_like(probabilities, 1 / probabilities.shape[-1])
            ass = _rows(keep_s, probabilities)
        k = _rows(kappa, probabilities)
        return k * posterior + (1 - k) * (ass * clean + (1 - ass) * prior.double())

    def __call__(self, view, state, probabilities, tokens, source, destination, **context):
        mixture = self.probabilities(view, state, probabilities, source, destination)
        if mixture is None:  # κ = 1 (or a continuous view): the exact native update.
            return super().__call__(view, state, probabilities, tokens, source, destination, **context)
        return sample_categorical(mixture.clamp_min(0))


class LoopTimeWarp:
    """Map the linear grid to schedule times whose keep rate follows the loop profile.

    The target α(t) is piecewise linear (ReMDM-loop): (1 - t) α_on / (1 - t_on) above
    t_on, α_on on [t_off, t_on], and 1 - t (1 - α_on) / t_off below t_off. Schedule times
    are found by bisection on the first discrete view's keep rate (monotone in time).
    """

    def __init__(self, t_on=0.55, t_off=0.05, alpha_on=0.9, iterations=60):
        self.t_on, self.t_off, self.alpha_on = float(t_on), float(t_off), float(alpha_on)
        self.iterations = int(iterations)

    def target(self, t):
        t = t.double()
        high = (1 - t) * self.alpha_on / (1 - self.t_on) if self.t_on < 1 else torch.zeros_like(t)
        low = 1 - t * (1 - self.alpha_on) / self.t_off
        hold = torch.full_like(t, self.alpha_on)
        return torch.where(t > self.t_on, high, torch.where(t < self.t_off, low, hold))

    def __call__(self, algo, times):
        index = next(
            i for i, v in enumerate(algo.views) if isinstance(v.process, (AbsorbingProcess, UniformProcess))
        )

        def keep(time):
            return algo.points(time.to(times.dtype))[index].coefficients[0].double()

        lo, hi = torch.full_like(times, float(times.min())), torch.full_like(times, float(times.max()))
        floor, ceiling = keep(hi[:1]).item(), keep(lo[:1]).item()  # keep(1), keep(epsilon)
        goal = self.target(times).clamp(floor, ceiling)
        lo, hi = lo.double(), hi.double()
        for _ in range(self.iterations):  # keep decreases in time: larger keep, earlier time
            middle = (lo + hi) / 2
            above = keep(middle) > goal
            lo, hi = torch.where(above, middle, lo), torch.where(above, hi, middle)
        # Equal targets bisect identically, so the plateau has exactly α_s == α_t.
        return ((lo + hi) / 2).to(times.dtype)
