"""Gaussian noise calibrated to one-hot argmax error probability."""

import math
import torch
from torch import nn

from core.representations.spec import RepresentationRequirements
from .gaussian_path import GaussianPath
from .utils import build_argmax_error_table


class OneHotArgmaxErrorSchedule(nn.Module):
    def __init__(
        self,
        vocab_size,
        sigma_min=0.2,
        sigma_max=4.0,
        error_min=None,
        error_max=None,
        coupling="linear",
        num_points=1000,
        quadrature_order=512,
        training_epsilon=0.0,
    ):
        super().__init__()
        if not isinstance(vocab_size, int) or vocab_size < 2:
            raise ValueError("vocab_size must be an integer >= 2")
        if not all(math.isfinite(x) for x in (sigma_min, sigma_max, training_epsilon)):
            raise ValueError("sigma bounds and training_epsilon must be finite")
        if not 0 < sigma_min < sigma_max or not 0 <= training_epsilon < 1:
            raise ValueError(
                "require ordered positive sigmas and 0 <= training_epsilon < 1"
            )
        if any(not isinstance(x, int) or x < 2 for x in (num_points, quadrature_order)):
            raise ValueError("table size and quadrature order must be integers >= 2")
        if coupling not in {"linear", "geometric"}:
            raise ValueError("coupling must be linear or geometric")
        sigmas, errors = build_argmax_error_table(
            vocab_size, sigma_min, sigma_max, num_points, quadrature_order
        )
        # Extreme sigma ranges can produce numerically indistinguishable errors.
        distinct = torch.cat(
            (torch.ones(1, dtype=torch.bool), errors[1:] > errors[:-1])
        )
        sigmas, errors = sigmas[distinct], errors[distinct]
        if len(errors) < 2 or not torch.isfinite(errors).all():
            raise ValueError("sigma range must resolve distinct finite argmax errors")
        error_min = errors[0].item() if error_min is None else error_min
        error_max = errors[-1].item() if error_max is None else error_max
        if not all(math.isfinite(x) for x in (error_min, error_max)) or not (
            0 <= error_min < error_max <= 1 - 1 / vocab_size
        ):
            raise ValueError(
                "error bounds must lie between zero and the random-guess error"
            )
        if error_min < errors[0].item() or error_max > errors[-1].item():
            raise ValueError(
                "error bounds are outside the sigma table; widen sigma_min/sigma_max"
            )
        if coupling == "geometric" and error_min <= 0:
            raise ValueError("geometric progression requires positive error_min")
        self.error_min, self.error_max = error_min, error_max
        self.coupling, self.training_epsilon = coupling, training_epsilon
        self.representation_requirements = RepresentationRequirements(
            kinds=("one_hot",), vocab_size=vocab_size
        )
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("errors", errors)

    def training_time(self, time):
        return (1 - self.training_epsilon) * time

    def forward(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("schedule time must be finite and in [0, 1]")
        clock = time.double()
        if self.coupling == "linear":
            target = self.error_min + clock * (self.error_max - self.error_min)
            derivative = torch.full_like(clock, self.error_max - self.error_min)
        else:
            log_ratio = math.log(self.error_max / self.error_min)
            target = self.error_min * torch.exp(clock * log_ratio)
            derivative = target * log_ratio
        target = target.clamp(self.errors[0], self.errors[-1])
        index = torch.searchsorted(self.errors, target.contiguous(), right=True).clamp(
            1, len(self.errors) - 1
        )
        lo, hi = self.errors[index - 1], self.errors[index]
        slope = (self.sigmas[index] - self.sigmas[index - 1]) / (hi - lo)
        sigma = self.sigmas[index - 1] + (target - lo) * slope
        return GaussianPath(
            torch.ones_like(time),
            sigma.to(time),
            torch.zeros_like(time),
            (slope * derivative).to(time),
        )
