"""Expected-rank schedule for SLERP and iid uniform spherical codewords."""

import numpy as np
from scipy.special import betainc
from torch import nn
from core.representations.spec import RepresentationRequirements
from .embedding_rank import sphere_cosine_quadrature
from .rank_linear import RankLinearClock


def spherical_expected_rank(progress, dimension, quadrature_order=256):
    cosine, weights = sphere_cosine_quadrature(dimension, quadrature_order)
    theta = np.arccos(cosine)
    u = np.asarray(progress, dtype=float)
    # Competitor dot product C has (C+1)/2 ~ Beta((d-1)/2,(d-1)/2).
    # Use symmetry to evaluate the survival probability without subtracting 1.
    threshold = np.cos(u[..., None] * theta)
    a = (dimension - 1) / 2
    result = betainc(a, a, (1 - threshold) / 2) @ (weights / weights.sum())
    return np.where(u == 0, 0.0, np.where(u == 1, 0.5, result))


class SphericalRankSchedule(nn.Module):
    output_kind = "keep_rate"

    def __init__(
        self,
        dimension,
        grid_size=8193,
        quadrature_order=256,
        rank_min=0.01,
        rank_max=0.49,
        endpoint_width=0.01,
    ):
        super().__init__()
        if grid_size < 3:
            raise ValueError("grid_size must be at least three")
        u = np.linspace(0, 1, grid_size)
        self.clock = RankLinearClock(
            u,
            spherical_expected_rank(u, dimension, quadrature_order),
            rank_min,
            rank_max,
            endpoint_width,
        )
        self.representation_requirements = RepresentationRequirements(
            kinds=("embedding",), dimension=dimension, unit_normalized=True
        )

    def forward(self, time):
        progress, derivative = self.clock(time)
        return 1 - progress, -derivative
