"""The training time sampler contract: draw density and objective weighting.

A **noise schedule** fixes what each time ``t`` means (its corruption level,
shared by training and generation). A **training time sampler** decides how often
training visits each ``t``; it has no role at inference. In training only their
combination plus the loss weighting matters; they are separate components because
generation reuses the schedule and not the sampler, and because an adaptive
schedule's linearity constraint pins the schedule's shape.

Contract. Every training time sampler declares ``time_density``:

- a method ``time_density(time)`` returning, for the times its ``sample`` call
  just returned (same state, same step), the density of each draw relative to
  uniform on ``[0, 1]`` (a uniform sampler reports exactly 1). The ``epsilon``
  floor of the base uniform draw is ignored: it rescales every density by the
  same factor of at most ``1 / (1 - epsilon)``, which normalized weights cancel;
- or ``time_density = None`` when the draws have no density (a point mass); the
  sampler must say why in its docstring.

and ``objective_weighting``, how a non-uniform draw enters the optimized loss:

- ``importance_corrected``: the per-token loss is divided by the draw density,
  so the expected objective is the uniform-time objective and only the
  estimator variance changes;
- ``uncorrected``: the loss is used as drawn, so oversampling a region of ``t``
  also up-weights it in the objective (the heuristic adaptive samplers).

For a uniform sampler the two are the same objective. Observers (adaptive noise
schedules, adaptive time samplers) always receive the unweighted per-``t`` loss
plus the draw density, never the importance-weighted loss.
"""

import torch

OBJECTIVE_WEIGHTINGS = ("importance_corrected", "uncorrected")


def check_objective_weighting(value, supported=OBJECTIVE_WEIGHTINGS, owner="time sampler"):
    if value not in supported:
        raise ValueError(
            f"{owner} objective_weighting must be one of {list(supported)}, got {value!r}"
        )
    return value


def objective_weighting(sampler):
    """A sampler's declared weighting; undeclared samplers keep the loss as drawn."""
    return getattr(sampler, "objective_weighting", "uncorrected")


def training_time_density(sampler, time):
    """Detached per-draw density relative to uniform, or None when it has none.

    Raises when the sampler does not declare the contract at all, so a consumer
    that needs the density (an adaptive noise schedule, importance-corrected
    weighting) never silently assumes uniform draws.
    """
    if not hasattr(sampler, "time_density"):
        raise TypeError(
            f"{type(sampler).__name__} does not declare time_density: a training time "
            "sampler must report the density of its draws relative to uniform, or set "
            "time_density = None if it has none"
        )
    if sampler.time_density is None:
        return None
    if not torch.is_tensor(time):
        raise TypeError("time_density needs the sampled time tensor")
    return sampler.time_density(time).detach()


def weight_objective(per_token, sampler, density, objective=None):
    """Apply the sampler's objective weighting to a ``[B, ...]`` tokenwise loss.

    ``density`` is the draw density (``training_time_density``). An objective that
    declares ``divides_by_time_density`` already importance-corrects itself, so the
    loss is returned unchanged (no double correction).
    """
    weighting = objective_weighting(sampler)
    divides = getattr(objective, "divides_by_time_density", False)
    if weighting == "uncorrected" or divides:
        return per_token
    if density is None:
        raise ValueError(
            f"{type(sampler).__name__} reports no draw density, so "
            "objective_weighting=importance_corrected is unavailable; use uncorrected"
        )
    if not torch.isfinite(density).all() or (density <= 0).any():
        raise ValueError(
            "importance-corrected weighting needs a finite positive density for every draw"
        )
    work = torch.promote_types(per_token.dtype, torch.float32)
    density = density.reshape(-1, *([1] * (per_token.ndim - 1)))
    return (per_token.to(work) / density.to(per_token.device, work)).to(per_token.dtype)
