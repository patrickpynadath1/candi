"""Independent token-loss weighting supplied with explicit Gaussian coordinates."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from core.models.layers import TimestepEmbedder


@dataclass(frozen=True)
class ObjectiveContext:
    """Coordinates and optional noisy views, supplied without implicit detach."""

    time: torch.Tensor
    alpha: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    processes: tuple = ()
    coefficients: tuple = ()
    states: tuple = ()


def _positive_finite(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


class PowerLossWeighting(nn.Module):
    """Multiply CE by scale * max(coordinate, epsilon)**exponent.

    This is an explicit configurable weighting rule, not a claim that a particular
    exponent is the likelihood objective for every process.
    """

    requires_context = True

    def __init__(self, coordinate="sigma", exponent=0.0, scale=1.0, epsilon=1e-6):
        super().__init__()
        if coordinate not in {
            "time",
            "alpha",
            "sigma",
            "effective_noise",
            "snr",
            "log_snr",
        }:
            raise ValueError(f"unknown loss coordinate: {coordinate}")
        if coordinate == "log_snr":
            raise ValueError("power weighting requires a nonnegative coordinate")
        if not math.isfinite(exponent):
            raise ValueError("exponent must be finite")
        _positive_finite(scale, "scale")
        _positive_finite(epsilon, "epsilon")
        self.coordinate, self.epsilon = coordinate, float(epsilon)
        self.exponent, self.scale = float(exponent), float(scale)

    def forward(self, loss, context):
        if context is None:
            raise ValueError("loss weighting requires ObjectiveContext")
        if self.coordinate in {"time", "alpha", "sigma"}:
            coordinate = getattr(context, self.coordinate)
            if coordinate is None:
                raise ValueError(f"loss weighting requires context.{self.coordinate}")
            coordinate = coordinate.float()
            if coordinate.ndim != 1:
                raise ValueError("objective coordinate must have shape (batch,)")
        else:
            if context.alpha is None or context.sigma is None:
                raise ValueError(
                    f"loss weighting coordinate {self.coordinate} requires "
                    "context.alpha and context.sigma"
                )
            # Float32 avoids half-precision overflow when taking endpoint ratios.
            alpha = context.alpha.float()
            sigma = context.sigma.float()
            if alpha.ndim != 1 or sigma.shape != alpha.shape:
                raise ValueError(
                    "objective coordinates must have matching shape (batch,)"
                )
            alpha = alpha.clamp_min(self.epsilon)
            sigma = sigma.clamp_min(self.epsilon)
            if self.coordinate == "effective_noise":
                coordinate = sigma / alpha
            else:
                coordinate = (2 * (alpha.log() - sigma.log())).exp()
        coordinate = coordinate.clamp_min(self.epsilon)
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
        if coordinate not in {
            "time",
            "alpha",
            "sigma",
            "effective_noise",
            "snr",
            "log_snr",
        }:
            raise ValueError(f"unknown loss coordinate: {coordinate}")
        if cond_dim < 1 or hidden_dim < 1:
            raise ValueError("weighting network dimensions must be positive")
        _positive_finite(epsilon, "epsilon")
        self.coordinate, self.epsilon = coordinate, float(epsilon)
        self.time = TimestepEmbedder(cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def log_weight(self, context):
        if context is None:
            raise ValueError("loss weighting requires ObjectiveContext")
        if self.coordinate in {"time", "alpha", "sigma"}:
            coordinate = getattr(context, self.coordinate)
            if coordinate is None:
                raise ValueError(f"loss weighting requires context.{self.coordinate}")
            coordinate = coordinate.float()
            if coordinate.ndim != 1:
                raise ValueError("objective coordinate must have shape (batch,)")
        else:
            if context.alpha is None or context.sigma is None:
                raise ValueError(
                    f"loss weighting coordinate {self.coordinate} requires "
                    "context.alpha and context.sigma"
                )
            # Float32 avoids half-precision overflow when taking endpoint ratios.
            alpha = context.alpha.float()
            sigma = context.sigma.float()
            if alpha.ndim != 1 or sigma.shape != alpha.shape:
                raise ValueError(
                    "objective coordinates must have matching shape (batch,)"
                )
            alpha = alpha.clamp_min(self.epsilon)
            sigma = sigma.clamp_min(self.epsilon)
            if self.coordinate == "effective_noise":
                coordinate = sigma / alpha
            else:
                log_snr = 2 * (alpha.log() - sigma.log())
                coordinate = log_snr if self.coordinate == "log_snr" else log_snr.exp()
        return self.mlp(self.time(coordinate)).squeeze(-1).float()

    def forward(self, loss, context):
        w = self.log_weight(context)
        if w.shape != loss.shape[:1]:
            raise ValueError("objective context batch must match token loss")
        return torch.exp(-w[:, None]) * loss + w[:, None]
