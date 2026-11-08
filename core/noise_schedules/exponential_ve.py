"""Variance-exploding Gaussian path with exponentially increasing sigma."""

import math
import torch
from torch import nn

from core.representations.spec import RepresentationRequirements
from .gaussian_path import GaussianPath
from .utils import check_unit_time


class ExponentialVESchedule(nn.Module):
    representation_requirements = RepresentationRequirements()

    def __init__(self, sigma_min=0.01, sigma_max=10.0, training_epsilon=0.0):
        super().__init__()
        if not all(math.isfinite(x) for x in (sigma_min, sigma_max, training_epsilon)):
            raise ValueError("VE schedule parameters must be finite")
        if not 0 < sigma_min < sigma_max or not 0 <= training_epsilon < 1:
            raise ValueError(
                "require 0 < sigma_min < sigma_max and 0 <= training_epsilon < 1"
            )
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.training_epsilon = training_epsilon
        self.log_ratio = math.log(sigma_max / sigma_min)

    def training_time(self, time):
        return (1 - self.training_epsilon) * time

    def forward(self, time):
        check_unit_time(time, "schedule time must be finite and in [0, 1]")
        sigma = self.sigma_min * torch.exp(time * self.log_ratio)
        return GaussianPath(
            torch.ones_like(time), sigma, torch.zeros_like(time), sigma * self.log_ratio
        )
