"""Ψ-sampler transitions: native reverse steps mixed with forward re-noising.

From "The Diffusion Duality, Chapter II: Ψ-Samplers and Efficient Curriculum"
(arXiv 2602.21185, Eq. 12 and Suppl. A.4-A.5), generalised to every corruption:

    Ψ_{s|t} = κ_t · (native reverse step z_t -> z_s given x̂_θ)
            + (1 - κ_t) · (forward marginal q_s(z_s | x̂_θ))

mixed per position: each generated position takes the native step with
probability κ_t and is otherwise re-noised from the predicted clean token x̂_θ
to time s with its view's forward kernel (``corrupt_state``, as in
``ForwardResamplingSampler``). For a discrete view the second branch is
α_s · x̂ + (1 - α_s) · π (π = mask, or uniform 1/K): the paper's formula. For
Gaussian, simplex, spherical, vMF and blurring views it re-noises the
prediction with that process's own kernel. One κ_t, one clean token and one
branch choice per position are shared by all views of a hybrid.

κ_t = 1 is the native sampler; κ_t = 0 is forward re-sampling. κ_t comes from a
ReMDM remasking scale σ_t: κ_t = 1 - σ_t / (1 - α_s), σ_t^max = min{1, (1 - α_s) / α_t}:

- ``rescale``:  σ_t = η σ_t^max;
- ``cap``:      σ_t = min{η, σ_t^max};
- ``loop``:     the sampler's grid is warped (``LoopTimeWarp``) so α_t is piecewise
                linear: 0 -> α_on on (t_on, 1], α_on on [t_off, t_on], α_on -> 1 on
                [0, t_off); on the hold σ_t = η, elsewhere κ = 1;
- ``constant``: κ_t = 1 - η at every step (the paper's image setting, e.g. κ = 0.95).

α_t is the reference view's clean fraction (``keep_rate``): the keep rate of the
first discrete view, else the first view's (Gaussian α²/(α²+σ²), simplex α,
otherwise 1 - t). This is a transition policy for ``ComposedSampler``: time
progression, stopping and NFE accounting stay in the sampler (one network call
per step, as ancestral and forward re-sampling).
"""

import math
from dataclasses import dataclass

import torch

from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform import UniformProcess
from .policies import ReverseTransition, select_state

SCHEDULES = ("rescale", "cap", "loop", "constant")
CLEAN = ("sample", "greedy")


def keep_rate(view, point):
    """Clean fraction α of a view at a point, one value per batch row."""
    coefficients = point.coefficients
    if isinstance(view.process, (AbsorbingProcess, UniformProcess)):
        return coefficients[0].double()
    alpha, sigma = getattr(coefficients, "alpha", None), getattr(coefficients, "sigma", None)
    if alpha is not None and sigma is not None:  # Gaussian path: signal-variance fraction
        alpha, sigma = alpha.double(), sigma.double()
        return alpha**2 / (alpha**2 + sigma**2)
    if alpha is not None:  # simplex path: the clean mixing weight
        return alpha.double()
    return (1 - point.time).double()


def reference_view(views):
    """The view whose schedule sets α_t: the first discrete view, else the first."""
    return next(
        (i for i, v in enumerate(views) if isinstance(v.process, (AbsorbingProcess, UniformProcess))),
        0,
    )


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
        if self.schedule == "constant":
            return torch.full_like(keep_s, 1 - self.eta)
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

    def warp(self):
        return LoopTimeWarp(self.t_on, self.t_off, self.alpha_on) if self.schedule == "loop" else None


@dataclass
class PsiStep:
    """Shared per-step choices: κ per row, clean tokens and the native-branch mask."""

    kappa: torch.Tensor
    clean: torch.Tensor
    native: torch.Tensor


class PsiTransition(ReverseTransition):
    """Ψ transition for every view; ``clean`` picks x̂_θ by sampling or argmax."""

    requires_tokens = False

    def __init__(
        self, schedule="rescale", eta=0.02, t_on=0.55, t_off=0.05, alpha_on=0.9, clean="sample"
    ):
        super().__init__()
        if clean not in CLEAN:
            raise ValueError(f"clean must be one of {CLEAN}")
        self.kappa = KappaSchedule(schedule, eta, t_on, t_off, alpha_on)
        self.clean = clean

    @property
    def time_warp(self):
        return self.kappa.warp()

    def validate(self, views, commitment):
        super().validate(views, commitment)
        if commitment != "none":
            raise ValueError("Ψ transitions re-noise committed tokens; use commitment=none")
        for view in views:
            if not callable(getattr(view.process, "corrupt_state", None)) or not callable(
                getattr(view.representation, "encode", None)
            ):
                raise TypeError("Ψ-sampling needs a forward corrupt_state kernel in every view")

    def prepare(self, views, logits, probabilities, source, destination, generated):
        """Draw this step's shared choices once for all views; None when κ = 1."""
        index = reference_view(views)
        kappa = self.kappa(
            keep_rate(views[index], source[index]), keep_rate(views[index], destination[index])
        ).to(generated.device)
        if bool((kappa == 1).all()):
            return None  # The native sampler, with its exact random-number use.
        clean = (
            probabilities.argmax(-1)
            if self.clean == "greedy"
            else torch.distributions.Categorical(logits=logits).sample()
        )
        if bool((kappa == 0).all()):
            native = torch.zeros_like(generated)  # Forward re-sampling, with its RNG use.
        else:
            native = torch.rand(generated.shape, device=generated.device) < kappa[:, None]
        return PsiStep(kappa, clean, native & generated)

    def __call__(self, view, state, probabilities, tokens, source, destination, *, step=None,
                 **context):
        if step is None:
            return super().__call__(view, state, probabilities, tokens, source, destination, **context)
        generated, prompt_tokens = context["generated"], context["prompt_tokens"]
        native = (
            super().__call__(view, state, probabilities, tokens, source, destination, **context)
            if step.native.any()
            else state
        )
        renoise = generated & ~step.native
        if not renoise.any():
            return native
        clean = torch.where(generated, step.clean, prompt_tokens)
        forward = view.process.corrupt_state(
            view.representation.encode(clean), destination.coefficients, generated
        )
        return select_state(step.native, native, forward)


class LoopTimeWarp:
    """Map the linear grid to schedule times whose keep rate follows the loop profile.

    The target α(t) is piecewise linear (ReMDM-loop): (1 - t) α_on / (1 - t_on) above
    t_on, α_on on [t_off, t_on], and 1 - t (1 - α_on) / t_off below t_off. Schedule times
    are found by bisection on the reference view's clean fraction (monotone in time).
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

    def __call__(self, algo, times, generated=None):
        index = reference_view(algo.views)
        view = algo.views[index]

        def keep(time):
            mask = None if generated is None else generated[:1].expand(len(time), -1)
            return keep_rate(view, algo.points(time.to(times.dtype), corruption_mask=mask)[index])

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
