"""Capability checks a sampler runs before sampling.

An incompatible recipe is refused with a reason (TypeError) instead of failing deep
inside the loop. docs/validation/compatibility.md records every family x sampler cell.
"""


def require(condition, message):
    if not condition:
        raise TypeError(message)


def views(algo, sampler):
    """The view path: a diffusion recipe with ``views`` and ``points``."""
    require(
        hasattr(algo, "views") and hasattr(algo, "points"),
        f"{sampler} samples diffusion recipes on the view path; "
        "autoregressive models use inference/sampler=autoregressive",
    )


def process_method(algo, method, sampler, family):
    """A single-process recipe whose process implements ``method``."""
    process = getattr(algo, "process", None)
    name = type(process).__name__ if process is not None else "this recipe (no single process)"
    require(
        callable(getattr(process, method, None)),
        f"{sampler} requires a {family} process; {name} has no {method}",
    )
