"""Gaussian sigma lookup indexed by normalized expected coordinate rank."""

from core.representations.spec import RepresentationRequirements

import math
import torch
from torch import nn
from .utils import expected_rank, sigma_from_time_vectorized
from .gaussian_path import GaussianPath


class OneHotRankSchedule(nn.Module):
    def __init__(
        self,
        vocab_size,
        sigma_min=0.2,
        sigma_max=4.0,
        min_percentile=0.01,
        max_percentile=0.4,
        num_points=1000,
        training_epsilon=0.001,
        inversion="lookup",
        coupling="linear",
        normalization="vocabulary",
    ):
        super().__init__()
        if not all(
            math.isfinite(v)
            for v in (
                sigma_min,
                sigma_max,
                min_percentile,
                max_percentile,
                training_epsilon,
            )
        ):
            raise ValueError("schedule bounds must be finite")
        if vocab_size < 2 or not 0 < sigma_min < sigma_max or num_points < 2:
            raise ValueError(
                "require vocabulary >=2, ordered positive sigmas, >=2 lookup points"
            )
        if (
            not 0 <= min_percentile < max_percentile <= 1
            or not 0 <= training_epsilon < 1
        ):
            raise ValueError("invalid percentile range or training epsilon")
        if inversion not in {"lookup", "analytic"}:
            raise ValueError("inversion must be lookup or analytic")
        if coupling not in {"linear", "geometric"}:
            raise ValueError("coupling must be linear or geometric")
        if normalization not in {"vocabulary", "competitors"}:
            raise ValueError("normalization must be vocabulary or competitors")
        self.rank_scale = (
            (vocab_size - 1) / vocab_size if normalization == "vocabulary" else 1.0
        )
        if coupling == "geometric" and min_percentile <= 0:
            raise ValueError("geometric rank progression requires a positive minimum")
        self.inversion, self.coupling, self.normalization = (
            inversion,
            coupling,
            normalization,
        )
        self.representation_requirements = RepresentationRequirements(
            kinds=("one_hot",), vocab_size=vocab_size
        )
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.min_percentile, self.max_percentile = min_percentile, max_percentile
        self.training_epsilon = training_epsilon
        sigmas = torch.linspace(sigma_min, sigma_max, num_points)
        self.register_buffer("sigmas", sigmas)
        ranks = expected_rank(vocab_size, torch.tensor(1.0), sigmas)
        if normalization == "competitors":
            ranks = ranks / ((vocab_size - 1) / vocab_size)
        self.register_buffer("ranks", ranks)

    def training_time(self, time):
        # Legacy log-linear keep schedule: corruption probability = (1-eps)*t.
        return (1 - self.training_epsilon) * time

    def forward(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("time must be finite and in [0, 1]")
        if self.coupling == "linear":
            target = self.min_percentile + time * (
                self.max_percentile - self.min_percentile
            )
            rank_derivative = torch.full_like(
                time, self.max_percentile - self.min_percentile
            )
        else:
            log_ratio = math.log(self.max_percentile / self.min_percentile)
            target = self.min_percentile * torch.exp(time * log_ratio)
            rank_derivative = target * log_ratio
        sigma, slope = self._inverse_with_slope(target)
        return GaussianPath(
            torch.ones_like(time),
            sigma,
            torch.zeros_like(time),
            slope * rank_derivative,
        )

    def _inverse_with_slope(self, target):
        if self.inversion == "analytic":
            return self._analytic_inverse(target)
        return self._lookup_inverse(target)

    def _lookup_inverse(self, target):
        sigma = sigma_from_time_vectorized(target, self.sigmas, self.ranks)
        index = torch.searchsorted(self.ranks, target, right=True).clamp(
            1, len(self.ranks) - 1
        )
        slope = (self.sigmas[index] - self.sigmas[index - 1]) / (
            self.ranks[index] - self.ranks[index - 1] + 1e-8
        )
        slope = torch.where(
            (target < self.ranks[0]) | (target > self.ranks[-1]), 0.0, slope
        )
        return sigma, slope

    def _analytic_inverse(self, target):
        # Both inverses honor the same configured sigma interval. Evaluate the
        # inverse CDF in float64 to avoid tail cancellation at small sigmas.
        lower = (
            torch.special.ndtr(
                target.new_tensor(
                    -1 / (math.sqrt(2) * self.sigma_min), dtype=torch.float64
                )
            )
            * self.rank_scale
        )
        upper = (
            torch.special.ndtr(
                target.new_tensor(
                    -1 / (math.sqrt(2) * self.sigma_max), dtype=torch.float64
                )
            )
            * self.rank_scale
        )
        rank = target.double().clamp(lower, upper)
        quantile = torch.special.ndtri(rank / self.rank_scale)
        sigma = -1 / (math.sqrt(2) * quantile)
        density = torch.exp(-0.5 * quantile.square()) / math.sqrt(2 * math.pi)
        slope = 1 / (math.sqrt(2) * quantile.square() * density * self.rank_scale)
        slope = torch.where(
            (target.double() < lower) | (target.double() > upper) | (rank <= 0),
            0.0,
            slope,
        )
        return sigma.to(target).clamp(self.sigma_min, self.sigma_max), slope.to(target)


# Saved configs and callers used this less-specific name.
GaussianRankSchedule = OneHotRankSchedule
