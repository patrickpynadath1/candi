"""Change Gaussian state scale while preserving effective noise sigma/alpha."""

import torch
from torch import nn

from .gaussian_path import GaussianPath


class RescaledGaussianSchedule(nn.Module):
    """Normalize a Gaussian path to VE, VP, or linear interpolation coefficients.

    This rescales the state, not time. Coefficient derivatives include the
    time-dependent normalization; no endpoint epsilon changes the base path.
    """

    def __init__(self, base, path_type="vp"):
        super().__init__()
        if not hasattr(base, "representation_requirements"):
            raise TypeError("rescaling requires a Gaussian-path base schedule")
        if path_type not in {"ve", "vp", "linear"}:
            raise ValueError("path_type must be ve, vp, or linear")
        self.base = base
        self.path_type = path_type

    @property
    def representation_requirements(self):
        return self.base.representation_requirements

    @property
    def observe(self):
        # AttributeError intentionally preserves capability checks on fixed bases.
        return self.base.observe

    @property
    def loss_signal(self):
        return self.base.loss_signal

    @property
    def snapshot(self):
        return self.base.snapshot

    def training_time(self, time):
        return self.base.training_time(time)

    def forward(self, time):
        path = self.base(time)
        a, s, da, ds = path.alpha, path.sigma, path.dalpha_dt, path.dsigma_dt
        if not all(torch.isfinite(value).all() for value in (a, s, da, ds)):
            raise ValueError("Gaussian coefficients and derivatives must be finite")
        if torch.any(a < 0) or torch.any(s < 0):
            raise ValueError("Gaussian coefficients must be nonnegative")
        if self.path_type == "ve":
            if torch.any(a == 0):
                raise ValueError("VE rescaling is singular when alpha is zero")
            denominator, derivative = a, da
        elif self.path_type == "linear":
            denominator, derivative = a + s, da + ds
        else:
            denominator = torch.hypot(a, s)
            if torch.any(denominator == 0):
                raise ValueError("cannot rescale a path with alpha and sigma both zero")
            derivative = (a / denominator) * da + (s / denominator) * ds
        if torch.any(denominator == 0):
            raise ValueError("cannot rescale a path with alpha and sigma both zero")
        alpha, sigma = a / denominator, s / denominator
        return GaussianPath(
            alpha,
            sigma,
            (da - alpha * derivative) / denominator,
            (ds - sigma * derivative) / denominator,
        )
