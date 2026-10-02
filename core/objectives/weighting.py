"""Independent token-loss weighting supplied with explicit Gaussian coordinates."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from core.models.layers import TimestepEmbedder


@dataclass(frozen=True)
class ObjectiveContext:
    """Per-example coordinates supplied by the algorithm, without implicit detach."""

    time: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor


def _positive_finite(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


class LossCoordinate(nn.Module):
    """Select one scalar per example; ratio endpoints use explicit epsilon floors."""

    names = {"time", "alpha", "sigma", "effective_noise", "snr", "log_snr"}

    def __init__(self, name="time", epsilon=1e-6):
        super().__init__()
        if name not in self.names:
            raise ValueError(f"unknown loss coordinate: {name}")
        _positive_finite(epsilon, "epsilon")
        self.name, self.epsilon = name, float(epsilon)

    def forward(self, context):
        if context is None:
            raise ValueError("loss weighting requires ObjectiveContext")
        # Float32 avoids half-precision overflow when taking endpoint ratios.
        t, a, s = (x.float() for x in (context.time, context.alpha, context.sigma))
        if t.ndim != 1 or a.shape != t.shape or s.shape != t.shape:
            raise ValueError("objective coordinates must have matching shape (batch,)")
        if self.name in {"time", "alpha", "sigma"}:
            return {"time": t, "alpha": a, "sigma": s}[self.name]
        a, s = a.clamp_min(self.epsilon), s.clamp_min(self.epsilon)
        if self.name == "effective_noise":
            return s / a
        log_snr = 2 * (a.log() - s.log())
        return log_snr if self.name == "log_snr" else log_snr.exp()


class PowerLossWeighting(nn.Module):
    """Multiply CE by scale * max(coordinate, epsilon)**exponent.

    This is an explicit configurable weighting rule, not a claim that a particular
    exponent is the likelihood objective for every process.
    """

    requires_context = True

    def __init__(self, coordinate="sigma", exponent=0.0, scale=1.0, epsilon=1e-6):
        super().__init__()
        if coordinate == "log_snr":
            raise ValueError("power weighting requires a nonnegative coordinate")
        if not math.isfinite(exponent):
            raise ValueError("exponent must be finite")
        _positive_finite(scale, "scale")
        self.coordinate = LossCoordinate(coordinate, epsilon)
        self.exponent, self.scale = float(exponent), float(scale)

    def forward(self, loss, context):
        coordinate = self.coordinate(context).clamp_min(self.coordinate.epsilon)
        weight = self.scale * coordinate.pow(self.exponent)
        if weight.shape != loss.shape[:1]:
            raise ValueError("objective context batch must match token loss")
        return loss * weight[:, None]


class LearnedLossWeighting(nn.Module):
    """FLM's exp(-w(t))*CE+w(t), with a zero-initialized scalar head.

    Coordinate choice is explicit: original FLM conditions on its schedule's alpha
    before its alpha-to-gamma conversion. Merely choosing a new path's alpha is not
    automatically the same legacy coordinate. The default here is global time.
    """

    requires_context = True

    def __init__(self, coordinate="time", cond_dim=128, hidden_dim=128, epsilon=1e-6):
        super().__init__()
        if cond_dim < 1 or hidden_dim < 1:
            raise ValueError("weighting network dimensions must be positive")
        self.coordinate = LossCoordinate(coordinate, epsilon)
        self.time = TimestepEmbedder(cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def log_weight(self, context):
        coordinate = self.coordinate(context)
        return self.mlp(self.time(coordinate)).squeeze(-1).float()

    def forward(self, loss, context):
        w = self.log_weight(context)
        if w.shape != loss.shape[:1]:
            raise ValueError("objective context batch must match token loss")
        return torch.exp(-w[:, None]) * loss + w[:, None]
