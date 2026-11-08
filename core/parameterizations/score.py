"""SEDD absorbing log scores; these are not normalized token distributions."""

import math
import torch


class AbsorbingScoreParameterization:
    def __call__(self, logits, state, sigma):
        if torch.any(sigma <= 0):
            raise ValueError("score prediction requires positive noise")
        log_scale = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        scores = logits - log_scale[:, None, None] - math.log(logits.shape[-1] - 1)
        return scores.scatter(-1, state[..., None], torch.zeros_like(scores[..., :1]))

    # Score pathway endpoints (``<use>_score``). Scores have no sampling
    # transform, so every use computes the same log scores.
    def training_score(self, logits, state, sigma):
        return self(logits, state, sigma)

    sampling_score = greedy_score = training_score

    def at_noise(self, sigma):
        """The score endpoints with ``sigma`` bound, called as ``(logits, state)``."""
        return _ScoresAtNoise(self, sigma)


class _ScoresAtNoise:
    def __init__(self, parameterization, sigma):
        self.parameterization, self.sigma = parameterization, sigma

    def training_score(self, logits, state):
        return self.parameterization.training_score(logits, state, self.sigma)

    sampling_score = greedy_score = training_score
