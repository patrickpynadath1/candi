"""Maps from the coordinator clock to a branch-local clock."""

import torch


class IdentityTimeMap:
    def __call__(self, time):
        return time


class PowerTimeMap:
    def __init__(self, power=1.0):
        if power <= 0:
            raise ValueError("time-map power must be positive")
        self.power = float(power)

    def __call__(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("global time must be finite and in [0, 1]")
        return time.pow(self.power)


class LinearTimeMap:
    def __init__(self, scale=1.0, offset=0.0):
        self.scale, self.offset = float(scale), float(offset)

    def __call__(self, time):
        result = self.offset + self.scale * time
        if not torch.isfinite(result).all() or torch.any((result < 0) | (result > 1)):
            raise ValueError("mapped local time must be finite and in [0, 1]")
        return result
