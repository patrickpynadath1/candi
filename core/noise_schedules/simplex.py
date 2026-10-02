"""Concentration schedules copied from the existing simplex implementation."""

from dataclasses import dataclass
import math
import torch
from torch import nn


def concentration(t, kind="constant_linear", nu0=0.4, nu1=0.75, pivot=0.8, epsilon=1.0):
    """Section 4.2 variance schedules, plus c=epsilon/(1-alpha)."""
    if kind == "ddim":
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return epsilon / t
    if kind not in {"constant", "constant_linear"}:
        raise ValueError(f"Unknown concentration schedule: {kind}")
    if not (0 < nu0 < 1 and 0 < nu1 < 1 and 0 <= pivot < 1):
        raise ValueError("nu values must be in (0,1), pivot in [0,1)")
    nu = torch.full_like(t, nu0)
    if kind == "constant_linear":
        nu = nu + ((t - pivot) / (1 - pivot)).clamp(0, 1) * (nu1 - nu0)
    return 1 / nu - 1


@dataclass(frozen=True)
class SimplexPath:
    time: torch.Tensor
    alpha: torch.Tensor
    concentration: torch.Tensor


class SimplexSchedule(nn.Module):
    """Physical interpolation stays alpha=1-t; adaptive clocks live elsewhere."""

    def __init__(
        self, kind="constant_linear", nu0=0.4, nu1=0.75, pivot=0.8, epsilon=1.0
    ):
        super().__init__()
        if not all(math.isfinite(value) for value in (nu0, nu1, pivot, epsilon)):
            raise ValueError("simplex schedule parameters must be finite")
        self.kind, self.nu0, self.nu1, self.pivot, self.epsilon = (
            kind,
            nu0,
            nu1,
            pivot,
            epsilon,
        )
        self(torch.ones(1, dtype=torch.float64))

    def forward(self, time):
        time = time.double()
        if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("simplex time must be finite and in [0,1]")
        # epsilon/t has infinite concentration at t=0. The process handles
        # this exact clean endpoint without drawing an infinite-shape Gamma.
        c = concentration(time, self.kind, self.nu0, self.nu1, self.pivot, self.epsilon)
        return SimplexPath(time, 1 - time, c)
