"""Numerical inversion of a monotone normalized expected-rank curve."""

import numpy as np
import torch
from torch import nn


class RankLinearClock(nn.Module):
    def __init__(
        self, progress, ranks, rank_min=0.01, rank_max=0.49, endpoint_width=0.01
    ):
        super().__init__()
        if not 0 <= rank_min < rank_max <= 0.5 or not 0 <= endpoint_width < 0.5:
            raise ValueError("invalid rank interval or endpoint width")
        progress, ranks = np.asarray(progress), np.asarray(ranks)
        if not np.isfinite(ranks).all() or np.any(np.diff(ranks) < -1e-12):
            raise ValueError("expected rank must be finite and monotone")
        ranks, indices = np.unique(np.maximum.accumulate(ranks), return_index=True)
        if ranks[0] > rank_min or ranks[-1] < rank_max:
            raise ValueError("requested ranks are outside the calibrated path")
        self.register_buffer("ranks", torch.tensor(ranks, dtype=torch.float64))
        self.register_buffer(
            "progress", torch.tensor(progress[indices], dtype=torch.float64)
        )
        self.rank_min, self.rank_max, self.endpoint_width = (
            rank_min,
            rank_max,
            endpoint_width,
        )

    def inverse(self, rank):
        index = torch.searchsorted(self.ranks, rank.contiguous()).clamp(
            1, len(self.ranks) - 1
        )
        slope = (self.progress[index] - self.progress[index - 1]) / (
            self.ranks[index] - self.ranks[index - 1]
        )
        return self.progress[index - 1] + (rank - self.ranks[index - 1]) * slope, slope

    def forward(self, time):
        if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("time must be finite and in [0,1]")
        time = time.double()
        width = self.endpoint_width
        fraction = ((time - width) / (1 - 2 * width)).clamp(0, 1)
        value, slope = self.inverse(
            self.rank_min + fraction * (self.rank_max - self.rank_min)
        )
        derivative = slope * (self.rank_max - self.rank_min) / (1 - 2 * width)
        if width:
            lower, _ = self.inverse(torch.full_like(time, self.rank_min))
            upper, _ = self.inverse(torch.full_like(time, self.rank_max))
            value = torch.where(time < width, lower * time / width, value)
            value = torch.where(
                time > 1 - width,
                upper + (1 - upper) * (time - (1 - width)) / width,
                value,
            )
            derivative = torch.where(time < width, lower / width, derivative)
            derivative = torch.where(time > 1 - width, (1 - upper) / width, derivative)
            value = torch.where(time == 0, 0.0, torch.where(time == 1, 1.0, value))
        return value, derivative
