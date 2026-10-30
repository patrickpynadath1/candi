"""Optional score-based correction after a predictor transition."""

import math
import torch
from core.processes.spherical import sphere_normalize


class SphericalLangevinCorrector:
    """Algorithm 2 projected Langevin step with a fresh posterior each iteration."""

    def __init__(self, steps=1, step_size=0.001):
        if steps < 1 or int(steps) != steps or step_size <= 0:
            raise ValueError("positive integer steps and positive step_size required")
        self.steps, self.step_size = int(steps), float(step_size)

    def applies_to(self, view):
        return getattr(view.process, "state_geometry", None) == "sphere" and callable(
            getattr(view.process, "score", None)
        )

    def validate(self, views):
        if not any(self.applies_to(view) for view in views):
            raise ValueError(
                "spherical correction requires a spherical view with a score"
            )

    def __call__(self, view, state, probabilities, point):
        if not self.applies_to(view):
            return state
        x = sphere_normalize(state.float())
        score = view.process.score(
            x, view.representation.codebook(), probabilities, point.coefficients[0]
        )
        delta = self.step_size * score + math.sqrt(
            2 * self.step_size
        ) * torch.randn_like(x)
        delta = delta - (delta * x).sum(-1, keepdim=True) * x
        return sphere_normalize(x + delta).to(state.dtype)
