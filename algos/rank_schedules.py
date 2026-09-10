"""Rank degradation in one-hot and unit-embedding Gaussian paths.

CANDI equations 5/6/10: https://arxiv.org/html/2510.22510v3#S4.SS1
Embedding ranks use linearity of expectation, not independent rank events.
"""

import numpy as np
import torch
from scipy.special import ndtr
from scipy.linalg import eigh_tridiagonal


def sphere_cosine_quadrature(dimension, order):
    """Normalized Gauss-Jacobi rule without high-dimension polynomial overflow."""
    if dimension < 2 or order < 2:
        raise ValueError("dimension and quadrature order must be at least two")
    n = np.arange(1, order, dtype=np.float64)
    if dimension == 2:
        off_diagonal = np.full(order-1, .5)
        off_diagonal[0] = 1/np.sqrt(2)
    else:
        off_diagonal = np.sqrt(n*(n+dimension-3)/((2*n+dimension-3)**2-1))
    nodes, vectors = eigh_tridiagonal(np.zeros(order), off_diagonal)
    return nodes, vectors[0]**2


def onehot_rank(sigma):
    sigma = torch.as_tensor(sigma)
    return torch.special.ndtr(-1 / (2**0.5 * sigma))


def sigma_from_rank(rank):
    if not torch.isfinite(rank).all() or not ((rank > 0) & (rank < .5)).all():
        raise ValueError("rank degradation must be strictly between 0 and .5")
    return (-1 / (2**0.5 * torch.special.ndtri(rank.double()))).to(rank.dtype)


def geometric_rank(progress, minimum=.01, maximum=.49):
    if not 0 < minimum < maximum < .5:
        raise ValueError("expected 0 < rank_min < rank_max < .5")
    return minimum * (maximum / minimum) ** progress


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


class SphereRankSchedule(torch.nn.Module):
    """Invert expected rank for iid uniform unit embeddings, Gaussian noise.

    Pairwise cosine density is proportional to (1-c*c)**((d-3)/2).
    Gauss-Jacobi quadrature handles the endpoint singularities for d=2.
    Interior rank is geometric; explicit linear-u boundary segments connect
    it to the exact clean and pure Gaussian endpoints.
    """

    def __init__(self, dimension, rank_min=.01, rank_max=.49,
                 endpoint_width=.01, grid_size=8193, quadrature_order=256):
        super().__init__()
        geometric_rank(torch.tensor(0.), rank_min, rank_max)
        if dimension < 2 or not 0 <= endpoint_width < .5:
            raise ValueError("need dimension >= 2 and 0 <= endpoint_width < .5")
        self.rank_min, self.rank_max = rank_min, rank_max
        self.endpoint_width = endpoint_width
        c, weights = sphere_cosine_quadrature(dimension, quadrature_order)
        weights /= weights.sum()
        u = np.linspace(0, 1, grid_size)
        ratio = (1 - u[1:]) / u[1:]
        ranks = np.concatenate(([0.], ndtr(
            -ratio[:, None] * np.sqrt((1 - c) / 2)
        ) @ weights))
        # Underflow creates duplicate zeros. Inversion needs unique abscissae.
        ranks, indices = np.unique(ranks, return_index=True)
        self.register_buffer("ranks", torch.tensor(ranks, dtype=torch.float64))
        self.register_buffer("interpolations", torch.tensor(u[indices], dtype=torch.float64))

    def inverse(self, rank):
        rank64 = rank.double()
        indices = torch.searchsorted(self.ranks, rank64.contiguous()).clamp(1, len(self.ranks)-1)
        lo, hi = self.ranks[indices-1], self.ranks[indices]
        fraction = (rank64 - lo) / (hi - lo)
        return (self.interpolations[indices-1] + fraction * (
            self.interpolations[indices] - self.interpolations[indices-1]
        )).to(rank.dtype)

    def forward(self, progress, coupling="geometric"):
        width = self.endpoint_width
        if width == 0:
            target = (
                self.rank_min
                + progress * (self.rank_max - self.rank_min)
                if coupling == "linear"
                else geometric_rank(progress, self.rank_min, self.rank_max)
            )
            return self.inverse(target)
        interior = ((progress - width) / (1 - 2*width)).clamp(0, 1)
        target = (self.rank_min + interior*(self.rank_max-self.rank_min)
                  if coupling == "linear" else geometric_rank(interior, self.rank_min, self.rank_max))
        value = self.inverse(target)
        lower = self.inverse(torch.full_like(progress, self.rank_min))
        upper = self.inverse(torch.full_like(progress, self.rank_max))
        value = torch.where(progress < width, lower * progress / width, value)
        return torch.where(progress > 1-width,
                           upper + (1-upper)*(progress-(1-width))/width, value)


class SLERPRankSchedule(torch.nn.Module):
    """Expected normalized rank for iid uniform codewords along SLERP.

    r_d(u) = E[F_d(u Theta)], with F_d the random spherical-angle CDF.
    Width zero preserves the requested truncated rank endpoints exactly.
    Positive width adds linear-u bridges to clean and independent noise.
    """

    def __init__(self, dimension, rank_min=.01, rank_max=.49,
                 endpoint_width=0., grid_size=8193, quadrature_order=256):
        super().__init__()
        from scipy.special import betainc
        geometric_rank(torch.tensor(0.), rank_min, rank_max)
        if dimension < 2 or not 0 <= endpoint_width < .5:
            raise ValueError('need dimension >= 2 and 0 <= endpoint_width < .5')
        self.rank_min, self.rank_max = rank_min, rank_max
        self.endpoint_width = endpoint_width
        self.register_buffer('dimension', torch.tensor(dimension))
        c, weights = sphere_cosine_quadrature(dimension, quadrature_order)
        weights /= weights.sum()
        theta = np.arccos(c)
        u = np.linspace(0, 1, grid_size)
        k = (dimension-1)/2
        ranks = betainc(k, k, np.sin(u[:, None]*theta/2)**2) @ weights
        ranks[0], ranks[-1] = 0., .5
        ranks, indices = np.unique(ranks, return_index=True)
        self.register_buffer('ranks', torch.tensor(ranks, dtype=torch.float64))
        self.register_buffer('interpolations', torch.tensor(u[indices], dtype=torch.float64))

    @staticmethod
    def _interp(value, xs, ys):
        v = value.double()
        idx = torch.searchsorted(xs, v.contiguous()).clamp(1, len(xs)-1)
        fraction = (v-xs[idx-1])/(xs[idx]-xs[idx-1])
        return (ys[idx-1]+fraction*(ys[idx]-ys[idx-1])).to(value.dtype)

    def inverse(self, rank):
        if not torch.isfinite(rank).all() or not ((rank >= 0) & (rank <= .5)).all():
            raise ValueError('rank must be in [0,.5]')
        return self._interp(rank, self.ranks, self.interpolations)

    def expected_rank(self, interpolation):
        return self._interp(interpolation, self.interpolations, self.ranks)

    def forward(self, progress, coupling="geometric"):
        w = self.endpoint_width
        interior = ((progress-w)/(1-2*w)).clamp(0, 1)
        target = (self.rank_min + interior*(self.rank_max-self.rank_min)
                  if coupling == "linear" else geometric_rank(interior, self.rank_min, self.rank_max))
        result = self.inverse(target)
        if w == 0:
            return result
        lower = self.inverse(torch.full_like(progress, self.rank_min))
        upper = self.inverse(torch.full_like(progress, self.rank_max))
        result = torch.where(progress < w, lower*progress/w, result)
        return torch.where(progress > 1-w, upper+(1-upper)*(progress-(1-w))/w, result)
