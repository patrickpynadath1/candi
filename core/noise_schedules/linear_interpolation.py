"""Linear flow-matching path from clean data at zero to Gaussian noise at one."""

from core.representations.spec import RepresentationRequirements

import torch
from torch import nn
from .gaussian_path import GaussianPath


class LinearInterpolationSchedule(nn.Module):
    representation_requirements = RepresentationRequirements()

    def training_time(self, time):
        return time

    def forward(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("time must be finite and in [0, 1]")
        return GaussianPath(
            1 - time, time, -torch.ones_like(time), torch.ones_like(time)
        )
