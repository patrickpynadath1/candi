"""Spherical keep coordinates and dimension-aware truncation."""

import torch
import math


def spherical_truncation_alpha_max(vocab_size, dimension, delta=0.1):
    """Dimension-aware S-FLM truncation bound for iid spherical codewords."""
    if vocab_size < 2 or dimension < 2:
        raise ValueError("vocab_size and dimension must be at least two")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    bound = min(
        1.0,
        math.sqrt(2 * math.log(2 * (vocab_size - 1) / delta) / dimension),
    )
    return 2 / math.pi * math.asin(bound)


class SphericalSchedule:
    output_kind = "keep_rate"

    def __init__(
        self,
        epsilon=0.001,
        alpha_max=1.0,
        auto_truncation=False,
        vocab_size=12,
        dimension=512,
        truncation_delta=0.1,
    ):
        self.alpha_max = (
            spherical_truncation_alpha_max(vocab_size, dimension, truncation_delta)
            if auto_truncation
            else alpha_max
        )
        if not 0 <= epsilon < self.alpha_max <= 1:
            raise ValueError("need 0 <= epsilon < alpha_max <= 1")
        self.epsilon = epsilon

    def __call__(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("spherical time must be in [0, 1]")
        return self.alpha_max + (self.epsilon - self.alpha_max) * time, torch.full_like(
            time, self.epsilon - self.alpha_max
        )
