"""A fixed piecewise-constant training time sampler.

The training time sampler decides how often training visits each ``t``; the noise
schedule still fixes what each ``t`` means. ``weights`` are relative masses of
equal-width bins of ``[0, 1]``, so ``[1, 3, 3, 1]`` visits mid-range times three
times as often as the ends. Draws are the exact inverse CDF of (optionally
antithetic) uniform draws, and ``time_density`` is the bin's density relative to
uniform. With ``objective_weighting=importance_corrected`` (the default) the loss
is divided by that density, so the expected objective is the uniform-time one and
only the estimator variance changes.
"""

import math

import torch

from .density import check_objective_weighting
from .uniform import UniformTimeSampler


class PiecewiseTimeSampler:
    def __init__(
        self,
        weights=(1.0, 3.0, 3.0, 1.0),
        epsilon=1e-3,
        antithetic=True,
        objective_weighting="importance_corrected",
    ):
        weights = [float(value) for value in weights]
        if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("piecewise time sampler weights must be finite and positive")
        self.uniform = UniformTimeSampler(epsilon, antithetic)
        total = sum(weights)
        self.probabilities = torch.tensor(
            [w / total for w in weights], dtype=torch.float64
        )
        self.objective_weighting = check_objective_weighting(objective_weighting)

    @property
    def bins(self):
        return len(self.probabilities)

    def inverse_cdf(self, uniform):
        """Map uniform probabilities in ``[0, 1]`` to training times."""
        p = self.probabilities.to(uniform.device)
        cdf = p.cumsum(0)
        value = uniform.to(torch.float64)
        index = torch.searchsorted(cdf, value.contiguous(), right=True).clamp_max(
            self.bins - 1
        )
        left = torch.cat((p.new_zeros(1), cdf[:-1]))[index]
        time = (index + ((value - left) / p[index]).clamp(0, 1)) / self.bins
        return time.clamp(0, 1).to(uniform.dtype)

    def sample(self, count, device, **context):
        return self.inverse_cdf(self.uniform.sample(count, device, **context))

    def time_density(self, time):
        p = self.probabilities.to(time.device)
        index = (time.to(torch.float64) * self.bins).long().clamp(0, self.bins - 1)
        return (self.bins * p[index]).to(time.dtype)
