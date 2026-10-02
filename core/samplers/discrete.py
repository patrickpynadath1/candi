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
