"""Reusable fixed-grid execution, with no prediction or transition mathematics.

Updates may call inference zero, one, or multiple times. State can be a tensor,
a tuple of hybrid states, or another caller-owned object. Adaptive acceptance,
EB completion, and autoregressive cache policies retain their own loops.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LinearTimeGrid:
    start: float = 1.0
    end: float = 1e-5
    steps: int = 180

    def __call__(self, device, *, dtype=None):
        return torch.linspace(
            self.start, self.end, self.steps + 1, device=device, dtype=dtype
        )


class FixedGridLoop:
    """Execute an interchangeable update callable over supplied time points.

    ``update(state, time, next_time, index)`` owns the update procedure; callers
    retain initialization/finalization and actual model-call accounting. A step
    context can freeze prediction feedback across multiple solver stages.
    """

    def __call__(self, state, times, update, *, step_context=nullcontext):
        if times.ndim != 1 or len(times) < 2 or not torch.isfinite(times).all():
            raise ValueError(
                "time grid must be a finite vector with at least two points"
            )
        for index, (time, next_time) in enumerate(zip(times[:-1], times[1:])):
            with step_context():
                state = update(state, time, next_time, index)
        return state
