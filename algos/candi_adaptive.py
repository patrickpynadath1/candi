"""Two-axis corruption curves and diagnostics for CANDI."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class CorruptionCurve:
    """A monotone path from clean ``(0, 0)`` to noisy ``(1, 1)``."""

    def __call__(self, s: torch.Tensor):
        raise NotImplementedError


class PowerCurve(CorruptionCurve):
    def __init__(
        self, orientation: str = "linear", power: float = 1.0,
        linear_mix: float = 0.01,
    ):
        if orientation not in {"linear", "continuous_first", "discrete_first"}:
            raise ValueError(f"unknown CANDI curve orientation: {orientation}")
        if power < 1:
            raise ValueError("CANDI curve power must be at least one")
        if not 0 <= linear_mix < 1:
            raise ValueError("CANDI curve linear_mix must be in [0, 1)")
        self.orientation = orientation
        self.power = power
        self.linear_mix = linear_mix

    def __call__(self, s):
        if self.orientation == "linear":
            ones = torch.ones_like(s)
            return s, s, ones, ones
        p = self.power
        if self.orientation == "continuous_first":
            discrete = s.pow(p)
            continuous = 1 - (1 - s).pow(p)
            d_discrete = p * s.clamp_min(1e-8).pow(p - 1)
            d_continuous = p * (1 - s).clamp_min(1e-8).pow(p - 1)
        else:
            discrete = 1 - (1 - s).pow(p)
            continuous = s.pow(p)
            d_discrete = p * (1 - s).clamp_min(1e-8).pow(p - 1)
            d_continuous = p * s.clamp_min(1e-8).pow(p - 1)
        mix = self.linear_mix
        discrete = mix * s + (1 - mix) * discrete
        continuous = mix * s + (1 - mix) * continuous
        d_discrete = mix + (1 - mix) * d_discrete
        d_continuous = mix + (1 - mix) * d_continuous
        return discrete, continuous, d_discrete, d_continuous


class BezierCurve(CorruptionCurve):
    def __init__(self, control_1, control_2):
        self.p1 = tuple(float(x) for x in control_1)
        self.p2 = tuple(float(x) for x in control_2)
        for axis in range(2):
            if not 0 <= self.p1[axis] <= self.p2[axis] <= 1:
                raise ValueError("Bezier controls must be monotone on both axes")

    def __call__(self, s):
        one_minus = 1 - s
        coords = []
        derivatives = []
        for axis in range(2):
            p1, p2 = self.p1[axis], self.p2[axis]
            value = (
                3 * one_minus.square() * s * p1
                + 3 * one_minus * s.square() * p2
                + s.pow(3)
            )
            derivative = (
                3 * one_minus.square() * p1
                + 6 * one_minus * s * (p2 - p1)
                + 3 * s.square() * (1 - p2)
            )
            coords.append(value)
            derivatives.append(derivative)
        return coords[0], coords[1], derivatives[0], derivatives[1]


class SequentialCurve(CorruptionCurve):
    """Two normalized sigmoid phases with independently placed centers."""

    def __init__(self, discrete_center=0.7, continuous_center=0.3, sharpness=10.0):
        self.discrete_center = float(discrete_center)
        self.continuous_center = float(continuous_center)
        self.sharpness = float(sharpness)
        if not 0 < self.discrete_center < 1 or not 0 < self.continuous_center < 1:
            raise ValueError("sequential curve centers must lie in (0, 1)")
        if self.sharpness <= 0:
            raise ValueError("sequential curve sharpness must be positive")

    def _axis(self, s, center):
        k = self.sharpness
        low = torch.sigmoid(torch.as_tensor(-k * center, dtype=s.dtype, device=s.device))
        high = torch.sigmoid(
            torch.as_tensor(k * (1 - center), dtype=s.dtype, device=s.device)
        )
        raw = torch.sigmoid(k * (s - center))
        scale = high - low
        return (raw - low) / scale, k * raw * (1 - raw) / scale

    def __call__(self, s):
        discrete, d_discrete = self._axis(s, self.discrete_center)
        continuous, d_continuous = self._axis(s, self.continuous_center)
        return discrete, continuous, d_discrete, d_continuous


def build_curve(config):
    kind = config.algo.get("curve_type", "linear")
    if kind in {"linear", "power"}:
        return PowerCurve(
            config.algo.get("curve_orientation", "linear"),
            float(config.algo.get("curve_power", 1.0)),
            float(config.algo.get("curve_linear_mix", 0.01)),
        )
    if kind == "bezier":
        return BezierCurve(
            config.algo.curve_control_1,
            config.algo.curve_control_2,
        )
    if kind == "sequential":
        return SequentialCurve(
            config.algo.get("curve_discrete_center", 0.7),
            config.algo.get("curve_continuous_center", 0.3),
            config.algo.get("curve_sharpness", 10.0),
        )
    raise ValueError(f"unknown CANDI curve type: {kind}")


class JointLossSurface(torch.nn.Module):
    """Checkpointable 2D loss statistics for independently sampled noise axes."""

    def __init__(self, bins=16, smoothing_sigma=1.0):
        super().__init__()
        self.bins = int(bins)
        self.smoothing_sigma = float(smoothing_sigma)
        shape = (self.bins, self.bins)
        self.register_buffer("count", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("loss_sum", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("loss_sq_sum", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def record(self, discrete, continuous, loss):
        d = (discrete.detach().to(self.count) * self.bins).long().clamp(0, self.bins - 1)
        c = (continuous.detach().to(self.count) * self.bins).long().clamp(0, self.bins - 1)
        values = loss.detach().to(self.loss_sum)
        flat = d * self.bins + c
        ones = torch.ones_like(values, dtype=torch.float64)
        self.count.view(-1).scatter_add_(0, flat, ones)
        self.loss_sum.view(-1).scatter_add_(0, flat, values)
        self.loss_sq_sum.view(-1).scatter_add_(0, flat, values.square())
        self.updates += 1

    def snapshot(self):
        count = self.count.detach().cpu().numpy()
        mean = np.divide(
            self.loss_sum.detach().cpu().numpy(),
            count,
            out=np.full_like(count, np.nan),
            where=count > 0,
        )
        second = np.divide(
            self.loss_sq_sum.detach().cpu().numpy(),
            count,
            out=np.zeros_like(count),
            where=count > 0,
        )
        variance = np.maximum(second - np.square(np.nan_to_num(mean)), 0)
        # Fill empty cells from the global observed mean before smoothing. Counts
        # remain available so consumers can distinguish measured and imputed cells.
        fill = np.nanmean(mean) if np.isfinite(mean).any() else 0.0
        filled = np.nan_to_num(mean, nan=fill)
        smooth = gaussian_filter(filled, self.smoothing_sigma, mode="nearest")
        spacing = 1 / self.bins
        grad_d, grad_c = np.gradient(smooth, spacing, spacing)
        centers = ((np.arange(self.bins) + 0.5) / self.bins).tolist()
        return {
            "updates": int(self.updates.item()),
            "centers": centers,
            "count": count.tolist(),
            "mean_loss": mean.tolist(),
            "variance": variance.tolist(),
            "smoothed_loss": smooth.tolist(),
            "grad_discrete": grad_d.tolist(),
            "grad_continuous": grad_c.tolist(),
            "gradient_magnitude": np.hypot(grad_d, grad_c).tolist(),
        }
