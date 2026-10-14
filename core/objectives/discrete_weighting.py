"""Tokenwise weights for interpolating discrete kernels; no loss terms."""

import math
import torch
from torch import nn


class InterpolatingDiscreteWeight(nn.Module):
    def forward(self, process, coefficients, target, observed):
        if torch.any(coefficients.alpha <= 0) or torch.any(coefficients.alpha >= 1):
            raise ValueError("interpolating discrete weights require interior times")
        # For q_t = alpha_t*x + b_t, incoming rate is b'_t - alpha'_t/alpha_t*b_t.
        rate = process.mixing_derivative(coefficients) - (
            coefficients.dalpha / coefficients.alpha
        )[:, None] * process.mixing(coefficients)
        selected_rate = (
            rate[:, None, :]
            .expand_as(target.probs)
            .gather(-1, observed[..., None])
            .squeeze(-1)
        )
        probability = target.probs.gather(-1, observed[..., None]).squeeze(-1)
        if torch.any(probability <= 0):
            raise ValueError("observed token is outside the forward marginal support")
        return selected_rate.clamp_min(0) / probability


class ClippedWeight(nn.Module):
    """Wrap any weight component with the same explicitly bound arguments."""

    def __init__(self, base, maximum=1.0):
        super().__init__()
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("maximum must be positive and finite")
        self.base, self.maximum = base, maximum

    def forward(self, **kwargs):
        return self.base(**kwargs).clamp(max=self.maximum)
