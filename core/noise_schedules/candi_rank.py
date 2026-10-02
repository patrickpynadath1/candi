"""The CANDI rank-calibrated variance-exploding path."""

import torch
from torch import nn
from core.noise_schedules.gaussian_path import GaussianPath
from core.representations.spec import RepresentationRequirements


class CANDIRankSchedule(nn.Module):
    representation_requirements = RepresentationRequirements(kinds=("one_hot",))

    def __init__(self, vocab_size, min_percentile=0.05, max_percentile=0.4):
        super().__init__()
        if vocab_size < 2 or not 0 < min_percentile < max_percentile < .5:
            raise ValueError("CANDI rank endpoints must satisfy 0 < min < max < 0.5")
        self.vocab_size = int(vocab_size)
        self.min_percentile = float(min_percentile)
        self.max_percentile = float(max_percentile)
        sigmas = torch.linspace(.2, 4.0, 1000)
        normal = torch.distributions.Normal(0.0, 1.0)
        errors = (vocab_size - 1) * normal.cdf(
            -1 / (sigmas * torch.sqrt(torch.tensor(2.0)))
        ) / vocab_size
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("errors", errors)

    def training_time(self, time):
        return time

    def forward(self, time):
        rank = self.min_percentile + time * (
            self.max_percentile - self.min_percentile
        )
        rank = rank.clamp(self.errors[0].item(), self.errors[-1].item())
        indices = torch.searchsorted(self.errors, rank, right=True).clamp(
            1, len(self.errors) - 1
        )
        low, high = indices - 1, indices
        fraction = (rank - self.errors[low]) / (
            self.errors[high] - self.errors[low] + 1e-8
        )
        sigma = self.sigmas[low] + fraction * (
            self.sigmas[high] - self.sigmas[low]
        )
        slope = (self.sigmas[high] - self.sigmas[low]) / (
            self.errors[high] - self.errors[low] + 1e-8
        )
        dsigma = slope * (self.max_percentile - self.min_percentile)
        zeros = torch.zeros_like(time)
        return GaussianPath(torch.ones_like(time), sigma, zeros, dsigma)
