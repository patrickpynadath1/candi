"""Quantize a global time sampler onto positive D3PM training steps."""


class DiscreteTimeSampler:
    """Round a base training time sampler onto the grid ``k / steps``.

    Draws are discrete, so ``time_density`` is the probability of each step
    relative to the uniform distribution over the ``steps`` grid points, i.e.
    ``steps * P(k)``; it is the base density at the step's cell midpoint (exact
    for a uniform base, whose draws all have density 1).
    """

    def __init__(self, base, steps=1000):
        if not isinstance(steps, int) or steps < 2:
            raise ValueError("D3PM steps must be an integer >= 2")
        self.base, self.steps = base, steps

    @property
    def objective_weighting(self):
        return getattr(self.base, "objective_weighting", "uncorrected")

    @property
    def time_density(self):
        base = getattr(self.base, "time_density", None)
        if base is None:
            return None

        def density(time):
            return base((time - 0.5 / self.steps).clamp(0, 1))

        return density

    def sample(self, *args, **kwargs):
        time = self.base.sample(*args, **kwargs)
        return ((time * self.steps).floor() + 1).clamp_max(self.steps) / self.steps
