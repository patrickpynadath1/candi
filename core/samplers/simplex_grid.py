"""Inference clocks copied from the existing simplex implementation.

``adaptive`` applies the recipe's learned time map (``algorithm.time_map``, an
information-path component) to a uniform grid; generation never reads the
training time sampler.
"""

import dataclasses
from typing import Any, Literal

import torch


GridSchedule = Literal["linear", "cosine", "adaptive"]


@dataclasses.dataclass(frozen=True)
class SimplexInferenceGrid:
    schedule: GridSchedule = "linear"
    time_map: Any | None = None

    def build(self, steps: int, device: torch.device | str) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("sampling steps must be positive")
        u = torch.linspace(0, 1, steps + 1, device=device)
        if self.schedule == "linear":
            times = u
        elif self.schedule == "cosine":
            times = torch.sin(torch.pi * u / 2)
        elif self.schedule == "adaptive":
            time_map = self.time_map
            if time_map is None or not time_map.adaptive or not time_map.ready:
                raise ValueError(
                    "the adaptive grid requires a fitted learned time map "
                    "(algorithm.time_map, e.g. information_path/time_map=loss_derivative)"
                )
            times = time_map(u)
        else:
            raise ValueError(f"unknown sampling grid: {self.schedule}")
        times[0], times[-1] = 0, 1
        return times.flip(0)
