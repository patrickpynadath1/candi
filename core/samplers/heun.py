"""Public preset for the composable HeunUpdate integration rule."""

from core.samplers.continuous import FixedStepSampler
from core.samplers.integration import HeunUpdate


class HeunSampler(FixedStepSampler):
    def __init__(self, steps=180, **kwargs):
        kwargs.setdefault("update_rule", HeunUpdate())
        super().__init__(steps=steps, **kwargs)
