"""Inference clocks copied from the existing simplex implementation."""

import dataclasses
from typing import Literal

import torch

from typing import Any


GridSchedule = Literal["linear", "cosine", "adaptive"]


@dataclasses.dataclass(frozen=True)
class SimplexInferenceGrid:
    schedule: GridSchedule = "linear"
    adaptive_sampler: Any | None = None

    def build(self, steps: int, device: torch.device | str) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("sampling steps must be positive")
        u = torch.linspace(0, 1, steps + 1, device=device)
        if self.schedule == "linear":
            times = u
        elif self.schedule == "cosine":
            times = torch.sin(torch.pi * u / 2)
        elif self.schedule == "adaptive":
            sampler = self.adaptive_sampler
            if sampler is None or not sampler.adaptive or not sampler.ready:
                raise ValueError(
                    "adaptive inference requires a fitted training time sampler"
                )
            times = sampler(u)
        else:
            raise ValueError(f"unknown sampling grid: {self.schedule}")
        times[0], times[-1] = 0, 1
        return times.flip(0)
