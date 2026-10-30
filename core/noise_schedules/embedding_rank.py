"""Rank calibration for an iid uniform unit-sphere embedding approximation.

The pairwise-rank expectation is integrated by Gauss-Jacobi quadrature.
It is not calibrated to the geometry of a particular learned codebook.
"""

import math
import numpy as np
import torch
from torch import nn
from scipy.special import ndtr
from scipy.linalg import eigh_tridiagonal
from core.representations.spec import RepresentationRequirements
from .gaussian_path import GaussianPath


def sphere_cosine_quadrature(dimension, order):
    """Normalized Gauss-Jacobi rule without high-dimension polynomial overflow."""
    if dimension < 2 or order < 2:
        raise ValueError("dimension and quadrature order must be at least two")
    n = np.arange(1, order, dtype=np.float64)
    if dimension == 2:
        off_diagonal = np.full(order - 1, 0.5)
        off_diagonal[0] = 1 / np.sqrt(2)
    else:
        off_diagonal = np.sqrt(
            n * (n + dimension - 3) / ((2 * n + dimension - 3) ** 2 - 1)
        )
    nodes, vectors = eigh_tridiagonal(np.zeros(order), off_diagonal)
    return nodes, vectors[0] ** 2


def embedding_table_rank(embeddings, progress):
    """Exact mean normalized rank for z=(1-u)e_i+u*N(0,I).

    Averages uniformly over target tokens and competitors. Distinct unit rows
    are required; coincident rows have ambiguous ranks at the clean endpoint.
    """
    e = torch.nn.functional.normalize(embeddings.double(), dim=-1)
    count = e.shape[0]
    if count < 2:
        raise ValueError("at least two embeddings are required")
    cosine = (e @ e.T).clamp(-1, 1)
    distances = ((1 - cosine) / 2).clamp_min(0).sqrt()
    distances = distances[~torch.eye(count, dtype=torch.bool, device=e.device)]
    u = torch.as_tensor(progress, device=e.device, dtype=e.dtype)
    ratio = (1 - u) / u.clamp_min(torch.finfo(u.dtype).tiny)
    return torch.special.ndtr(-ratio[..., None] * distances).mean(-1)


class EmbeddingSphereRankSchedule(nn.Module):
    def __init__(
        self,
        dimension,
        rank_min=0.01,
        rank_max=0.49,
        endpoint_width=0.01,
        grid_size=8193,
        quadrature_order=256,
        coupling="geometric",
    ):
        super().__init__()
        if not all(math.isfinite(v) for v in (rank_min, rank_max, endpoint_width)):
            raise ValueError("rank and endpoint bounds must be finite")
        if not 0 < rank_min < rank_max < 0.5 or not 0 <= endpoint_width < 0.5:
            raise ValueError(
                "require 0 < rank_min < rank_max < .5 and endpoint_width in [0,.5)"
            )
        if not isinstance(grid_size, int) or grid_size < 3:
            raise ValueError("grid_size must be an integer >= 3")
        if coupling not in {"linear", "geometric"}:
            raise ValueError("coupling must be linear or geometric")
        self.representation_requirements = RepresentationRequirements(
            kinds=("embedding", "centered_embedding"), dimension=dimension, unit_normalized=True
        )
        self.rank_min, self.rank_max = rank_min, rank_max
        self.endpoint_width, self.coupling = endpoint_width, coupling
        c, weights = sphere_cosine_quadrature(dimension, quadrature_order)
        weights /= weights.sum()
        u = np.linspace(0, 1, grid_size)
        ratio = (1 - u[1:]) / u[1:]
        ranks = np.concatenate(
            ([0.0], ndtr(-ratio[:, None] * np.sqrt((1 - c) / 2)) @ weights)
        )
        ranks, indices = np.unique(ranks, return_index=True)
        self.register_buffer("ranks", torch.tensor(ranks, dtype=torch.float64))
        self.register_buffer(
            "interpolations", torch.tensor(u[indices], dtype=torch.float64)
        )

    def training_time(self, time):
        return time

    def _inverse_with_slope(self, rank):
        rank64 = rank.double()
        index = torch.searchsorted(self.ranks, rank64.contiguous()).clamp(
            1, len(self.ranks) - 1
        )
        lo, hi = self.ranks[index - 1], self.ranks[index]
        slope = (self.interpolations[index] - self.interpolations[index - 1]) / (
            hi - lo
        )
        value = self.interpolations[index - 1] + (rank64 - lo) * slope
        return value.to(rank.dtype), slope.to(rank.dtype)

    def forward(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("time must be finite and in [0, 1]")
        width = self.endpoint_width
        interior = ((time - width) / (1 - 2 * width)).clamp(0, 1)
        if self.coupling == "linear":
            target = self.rank_min + interior * (self.rank_max - self.rank_min)
            derivative = torch.full_like(time, self.rank_max - self.rank_min)
        else:
            ratio = self.rank_max / self.rank_min
            target = self.rank_min * ratio**interior
            derivative = target * math.log(ratio)
        value, slope = self._inverse_with_slope(target)
        derivative = slope * derivative / (1 - 2 * width)
        if width > 0:
            lower, _ = self._inverse_with_slope(torch.full_like(time, self.rank_min))
            upper, _ = self._inverse_with_slope(torch.full_like(time, self.rank_max))
            value = torch.where(time < width, lower * time / width, value)
            value = torch.where(
                time > 1 - width,
                upper + (1 - upper) * (time - (1 - width)) / width,
                value,
            )
            derivative = torch.where(time < width, lower / width, derivative)
            derivative = torch.where(time > 1 - width, (1 - upper) / width, derivative)
        return GaussianPath(1 - value, value, -derivative, derivative)
