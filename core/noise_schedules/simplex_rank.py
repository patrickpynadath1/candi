"""Exact pairwise rank calibration of the uniform-reference Dirichlet path."""

import numpy as np
import torch
from torch import nn
from scipy.special import betainc
from .simplex import SimplexSchedule
from .rank_linear import RankLinearClock


def simplex_expected_rank(progress, concentration, vocab_size):
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least two")
    u, c = np.broadcast_arrays(
        np.asarray(progress, dtype=float), np.asarray(concentration, dtype=float)
    )
    # z_i/(z_i+z_j) ~ Beta(a_i,a_j); competitor wins iff this is below 1/2.
    interior = (u > 0) & (u < 1)
    result = np.where(u == 0, 0.0, 0.5)
    result[interior] = betainc(
        c[interior] * (1 - u[interior] + u[interior] / vocab_size),
        c[interior] * u[interior] / vocab_size,
        0.5,
    )
    return result


class SimplexRankSchedule(nn.Module):
    def __init__(
        self,
        vocab_size,
        base=None,
        grid_size=8193,
        rank_min=0.01,
        rank_max=0.49,
        endpoint_width=0.01,
    ):
        super().__init__()
        if grid_size < 3:
            raise ValueError("grid_size must be at least three")
        self.base = base if base is not None else SimplexSchedule()
        u = torch.linspace(0, 1, grid_size, dtype=torch.float64)
        c = self.base(u).concentration.detach().cpu().numpy()
        self.clock = RankLinearClock(
            u.numpy(),
            simplex_expected_rank(u.numpy(), c, vocab_size),
            rank_min,
            rank_max,
            endpoint_width,
        )
        self.vocab_size = vocab_size

    def validate_process(self, process):
        reference = getattr(process, "reference", None)
        if (
            reference is None
            or len(reference) != self.vocab_size
            or not torch.allclose(
                reference, torch.full_like(reference, 1 / self.vocab_size)
            )
        ):
            raise ValueError(
                "simplex rank calibration requires a uniform reference of matching size"
            )

    def forward(self, time):
        progress, _ = self.clock(time)
        return self.base(progress)
