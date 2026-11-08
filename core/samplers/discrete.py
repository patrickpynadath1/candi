"""Composable fixed-grid discrete sampling lifecycle.

A rule owns initialization, a reverse update and terminal decoding. It obtains
predictions from the algorithm and delegates transition equations to the process.
Update and finalize each return ``(state, number_of_model_evaluations)``.
The loop deliberately does not assume logits, a vocabulary, or one-hot states.
"""

import torch
from core.modifiers.self_conditioning.session import prediction_session
from .loop import FixedGridLoop, LinearTimeGrid


class DiscreteSampler:
    def __init__(self, update_rule, steps=180, epsilon=1e-5, time_grid=None):
        self.update_rule = update_rule
        self.steps, self.epsilon = steps, epsilon
        self.time_grid = time_grid or LinearTimeGrid(1, epsilon, steps)
        self.loop = FixedGridLoop()

    def _composed(self, precision, transition_prediction="sampling"):
        """The shared-loop ComposedSampler equivalent to this sampler's default rule."""
        from .composed import ComposedSampler
        from .policies import ReverseTransition, TokenSelection

        if self._custom or self.final_denoise == "none":
            raise ValueError(
                "view-based algorithms support the default rule and grid with "
                "final_denoise=ancestral or greedy"
            )
        ancestral = self.final_denoise == "ancestral"
        return ComposedSampler(
            self.steps,
            self.epsilon,
            tokens=TokenSelection("posterior"),
            transition=ReverseTransition(),
            final_denoise="reverse" if ancestral else "greedy",
            final_prediction="sampling" if ancestral else "greedy",
            precision=precision,
            transition_prediction=transition_prediction,
        )

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if self.update_rule.prediction_feedback:
            algo = prediction_session(algo)
        state = self.update_rule.initialize(algo, prompt_tokens, prompt_mask)
        times = self.time_grid(state.device)
        nfe = 0

        def update(value, time, next_time, index):
            nonlocal nfe
            value, evaluations = self.update_rule.update(
                algo, value, time, next_time, prompt_tokens, prompt_mask
            )
            nfe += evaluations
            return value

        state = self.loop(state, times, update)
        state, final_nfe = self.update_rule.finalize(
            algo, state, times[-1], prompt_tokens, prompt_mask
        )
        nfe += final_nfe
        return state, {"nfe": nfe, "nfe_per_sample": [nfe] * len(state)}
