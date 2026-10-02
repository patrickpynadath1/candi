"""Public preset for the composable EulerUpdate integration rule."""

from core.samplers.continuous import FixedStepSampler
from core.samplers.integration import EulerUpdate


class EulerSampler(FixedStepSampler):
    def __init__(self, steps=180, epsilon=1e-5, start_time=0.999, **kwargs):
        kwargs.setdefault("update_rule", EulerUpdate())
        super().__init__(steps=steps, epsilon=epsilon, start_time=start_time, **kwargs)
