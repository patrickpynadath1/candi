import torch
from .utils import check_unit_time


class LogLinear:
    output_kind = "keep_rate"

    def __init__(self, epsilon=1e-3):
        if not 0 <= epsilon < 1:
            raise ValueError("schedule epsilon must be in [0, 1)")
        self.epsilon = epsilon

    def __call__(self, time):
        check_unit_time(time, "schedule time must be in [0, 1]")
        keep = self.epsilon + (1 - self.epsilon) * (1 - time)
        derivative = torch.full_like(time, -(1 - self.epsilon))
        return keep, derivative
