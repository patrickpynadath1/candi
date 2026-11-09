"""Random reopening of committed positions, independent of token selection."""
import math
import torch
from core.processes.absorbing import AbsorbingProcess
from .policies import select_state


class RandomUncommitment:
    """Bernoulli release per step, tapered linearly to zero at stop_fraction.

    Released absorbing states are always masked; other states are drawn from
    their process-specific forward kernel at the current time.
    This is a refinement heuristic, not an ancestral reverse transition.
    """
    def __init__(self, rate=0.05, stop_fraction=0.8):
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("churn rate must lie in [0, 1]")
        if not math.isfinite(stop_fraction) or not 0 < stop_fraction <= 1:
            raise ValueError("stop_fraction must lie in (0, 1]")
        self.rate, self.stop_fraction = float(rate), float(stop_fraction)

    def probability(self, index, steps):
        return self.rate * max(0.0, 1 - index / max(1, steps - 1) / self.stop_fraction)

    def validate(self, views, commitment):
        if commitment != "freeze":
            raise ValueError("random uncommitment requires freeze commitment")
        for view in views:
            if not isinstance(view.process, AbsorbingProcess) and not callable(
                getattr(view.process, "corrupt_state", None)
            ):
                raise TypeError("random uncommitment requires a forward corrupt_state kernel")
            if not callable(getattr(view.representation, "encode", None)):
                raise TypeError("random uncommitment requires a token-encoding representation")

    def __call__(self, views, states, points, tokens, committed, prompt_mask, index, steps):
        probability = self.probability(index, steps)
        released = torch.zeros_like(committed)
        if probability == 0:
            return states, committed, released
        released = committed & ~prompt_mask & (torch.rand(committed.shape, device=committed.device) < probability)
        if not released.any():
            return states, committed, released
        updated = []
        for view, state, point in zip(views, states, points):
            if isinstance(view.process, AbsorbingProcess):
                noisy = torch.full_like(state, view.process.mask_index)
            else:
                noisy = view.process.corrupt_state(view.representation.encode(tokens), point.coefficients, released)
            updated.append(select_state(released, noisy, state))
        return tuple(updated), committed & ~released, released
