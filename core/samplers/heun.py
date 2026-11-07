"""Public preset for the composable HeunUpdate integration rule."""

from core.samplers.continuous import FixedStepSampler
from core.samplers.integration import HeunUpdate


class HeunSampler(FixedStepSampler):
    def __init__(self, steps=180, **kwargs):
        kwargs.setdefault("update_rule", HeunUpdate())
        super().__init__(steps=steps, **kwargs)

    def steps_for_nfe(self, nfe):
        """Two field evaluations per step plus the terminal decoding call."""
        if isinstance(nfe, bool) or int(nfe) != nfe:
            raise ValueError("NFE budget must be an integer")
        return max(1, (int(nfe) - 1) // 2)
