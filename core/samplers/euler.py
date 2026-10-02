"""First-order Euler integration of a continuous velocity field."""

from core.samplers.continuous import FixedStepSampler


class EulerSampler(FixedStepSampler):
    def __init__(self, steps=180, epsilon=1e-5, start_time=0.999, **kwargs):
        super().__init__(steps=steps, epsilon=epsilon, start_time=start_time, **kwargs)

    def step(self, field, state, time, next_time):
        return state + (next_time - time) * field(time, state)
