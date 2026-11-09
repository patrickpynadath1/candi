"""Sampling-only Gaussian score tempering, shared across reverse updates."""
import math
import torch


class GaussianScoreTemperature:
    """Plaid convention: divide predicted noise (and score) by temperature.

    For z = alpha*x + sigma*epsilon, transform the clean prediction so that
    (z-alpha*x_tempered)/sigma = epsilon_pred/temperature. Do not use the
    transformed prediction as self-conditioning feedback or terminal logits.
    """

    def __init__(self, temperature=1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("score_temperature must be finite and positive")
        self.temperature = float(temperature)

    def __call__(self, state, clean, path):
        if self.temperature == 1.0:
            return clean
        alpha = path.alpha.to(state)[:, None, None]
        if torch.any(~torch.isfinite(alpha) | (alpha <= 0)):
            raise ValueError("non-unit score_temperature requires finite positive source alpha")
        return clean + (1 / self.temperature - 1) * (clean - state / alpha)
