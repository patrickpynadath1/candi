import math


class DuoCurriculum:
    def __init__(
        self,
        start_step=100000,
        end_step=200000,
        log10_temperature_start=-1.0,
        log10_temperature_end=-2.0,
    ):
        if not 0 <= start_step < end_step:
            raise ValueError("require 0 <= start_step < end_step")
        if not all(
            math.isfinite(x) for x in (log10_temperature_start, log10_temperature_end)
        ):
            raise ValueError("curriculum temperatures must be finite")
        self.start_step, self.end_step = start_step, end_step
        self.log10_start, self.log10_end = (
            log10_temperature_start,
            log10_temperature_end,
        )

    def use_relaxation(self, step, training):
        # Preserve the legacy inclusive final curriculum step.
        return training and step <= self.end_step

    def inverse_temperature(self, step):
        if step < self.start_step:
            value = self.log10_start
        elif step < self.end_step:
            fraction = (step - self.start_step) / (self.end_step - self.start_step)
            value = self.log10_start + fraction * (self.log10_end - self.log10_start)
        else:
            value = -10.0
        return 10 ** (-value)
