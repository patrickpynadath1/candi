import torch


class LogLinear:
    output_kind = "keep_rate"

    def __init__(self, epsilon=1e-3):
        if not 0 <= epsilon < 1:
            raise ValueError("schedule epsilon must be in [0, 1)")
        self.epsilon = epsilon

    def __call__(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("schedule time must be in [0, 1]")
        keep = self.epsilon + (1 - self.epsilon) * (1 - time)
        derivative = torch.full_like(time, -(1 - self.epsilon))
        return keep, derivative
