"""A deterministic training time, composable with any compatible path."""

import math
import torch


class FixedTimeSampler:
    """Every draw is the same time: a point mass, which has no density relative to
    uniform (``time_density = None``), so only ``uncorrected`` weighting applies."""

    time_density = None
    objective_weighting = "uncorrected"

    def __init__(self, time=1.0):
        if not math.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("time must be in [0, 1]")
        self.time = float(time)

    def sample(self, count, device, **context):
        return torch.full((count,), self.time, device=device)
