"""Adaptive Dormand–Prince integration with explicit work budgets."""

import math
import torch

from core.samplers.continuous import ContinuousSampler


class ODESampler(ContinuousSampler):
    def __init__(self, rtol=1e-3, atol=1e-4, max_nfe=2000, max_steps=1000, **kwargs):
        super().__init__(**kwargs)
        if not all(math.isfinite(x) and x > 0 for x in (rtol, atol)):
            raise ValueError("rtol and atol must be finite and positive")
        for name, value in (("max_nfe", max_nfe), ("max_steps", max_steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_nfe <= 1:
            raise ValueError("max_nfe must leave room for velocity evaluation")
        self.rtol, self.atol = rtol, atol
        self.max_nfe, self.max_steps = max_nfe, max_steps

    def integrate(self, field, state):
        from torchdiffeq import odeint

        times = torch.tensor(
            [self.start_time, self.epsilon], device=state.device, dtype=state.dtype
        )
        # Force a step at the endpoint: adaptive internal stages must not evaluate
        # a schedule beyond its supported time interval.
        try:
            return odeint(
                field,
                state,
                times,
                method="dopri5",
                rtol=self.rtol,
                atol=self.atol,
                options={"max_num_steps": self.max_steps, "step_t": times[-1:]},
            )[-1]
        except AssertionError as error:
            raise RuntimeError(f"adaptive ODE integration failed: {error}") from error
