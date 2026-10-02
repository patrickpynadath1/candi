"""Quantize a global time sampler onto positive D3PM training steps."""


class DiscreteTimeSampler:
    def __init__(self, base, steps=1000):
        if not isinstance(steps, int) or steps < 2:
            raise ValueError("D3PM steps must be an integer >= 2")
        self.base, self.steps = base, steps

    def sample(self, *args, **kwargs):
        time = self.base.sample(*args, **kwargs)
        return ((time * self.steps).floor() + 1).clamp_max(self.steps) / self.steps
