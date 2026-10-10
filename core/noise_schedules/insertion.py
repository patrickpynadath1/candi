"""Sequence survival and reverse insertion clock (0 clean, 1 empty)."""

import math
import torch


class PowerInsertionSchedule:
    def __init__(self, power=1.0):
        if not math.isfinite(power) or power < 1:
            raise ValueError("insertion power must be finite and at least one")
        self.power = float(power)

    def survival(self, time):
        if not torch.isfinite(time).all() or ((time < 0) | (time > 1)).any():
            raise ValueError("insertion time must be in [0, 1]")
        return (1 - time).pow(self.power)

    def birth_times(self, shape, device):
        # P(birth >= t) = survival(t); generation runs from 1 to 0.
        return 1 - torch.rand(shape, device=device).pow(1 / self.power)

    def rate(self, time):
        survival = self.survival(time)
        return (
            self.power * (1 - time).pow(self.power - 1) / (1 - survival).clamp_min(1e-7)
        )

    def integrated_rate(self, source, destination):
        if (destination > source).any() or (destination <= 0).any():
            raise ValueError("insertion updates require 0 < destination <= source")
        return torch.log1p(-self.survival(source)) - torch.log1p(
            -self.survival(destination)
        )
