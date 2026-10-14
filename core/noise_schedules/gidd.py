"""Mask/uniform interpolation from GIDD, Section 4 (arXiv:2503.04482)."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GIDDPath:
    alpha: torch.Tensor
    mask: torch.Tensor
    uniform: torch.Tensor


class GIDDSchedule:
    """Uniform mixing peaks at p_uniform at t=1/2; both endpoints are exact.

    Restrict gamma to (0, 2] so every forward conditional mixing mass is
    nonnegative. p_uniform=0 gives the exact linear absorbing path.
    """

    def __init__(self, p_uniform=0.2, gamma=1.0):
        if not math.isfinite(p_uniform) or not 0 <= p_uniform < 1:
            raise ValueError("p_uniform must be finite and in [0, 1)")
        if not math.isfinite(gamma) or not 0 < gamma <= 2:
            raise ValueError("gamma must be finite and in (0, 2]")
        self.p_uniform, self.gamma = p_uniform, gamma
        self.B = 2**gamma * p_uniform / (1 - p_uniform)

    def __call__(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("GIDD time must be finite and in [0, 1]")
        time = time.double()
        c = self.B * (time * (1 - time)).pow(self.gamma / 2)
        return GIDDPath((1 - time) / (1 + c), time / (1 + c), c / (1 + c))
