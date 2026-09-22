"""CANDI recurrence restricted to a learned register bottleneck."""

from .candi_looped import CANDILooped
from .registry import algo_registry


@algo_registry.register("candi_register_looped")
class CANDIRegisterLooped(CANDILooped):
    def _validate_configuration(self):
        if self.config.algo.backbone != "register_looped_cont_dit":
            raise ValueError(
                "candi_register_looped requires backbone=register_looped_cont_dit"
            )
        loops = self.config.algo.validation_loops
        if isinstance(loops, bool) or not isinstance(loops, int) or loops < 1:
            raise ValueError("algo.validation_loops must be a positive integer")
