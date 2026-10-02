"""Second-order explicit trapezoidal integration (two evaluations per step)."""

from core.samplers.continuous import FixedStepSampler


class HeunSampler(FixedStepSampler):
    def step(self, field, state, time, next_time):
        delta = next_time - time
        first = field(time, state)
        second = field(next_time, state + delta * first)
        return state + 0.5 * delta * (first + second)
