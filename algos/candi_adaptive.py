"""Two-axis corruption curves and diagnostics for CANDI."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def joint_surface_svg(
    snapshot,
    size=700,
    arrow_stride=10,
    metric_name="loss",
    smoothed=True,
    confidence_shading=True,
):
    """Render raw or smoothed 2D loss statistics as a dependency-free SVG."""
    field_name = "smoothed_loss" if smoothed else "mean_loss"
    loss = np.asarray(snapshot[field_name], dtype=float)
    observed = np.asarray(snapshot["count"], dtype=float) > 0
    grad_d = np.asarray(snapshot["grad_discrete"], dtype=float)
    grad_c = np.asarray(snapshot["grad_continuous"], dtype=float)
    reliable = np.asarray(snapshot["reliable"], dtype=bool)
    bins = loss.shape[0]
    margin = 60
    plot = size - 2 * margin
    cell = plot / bins
    display_mask = reliable if smoothed and confidence_shading else np.isfinite(loss)
    trusted = loss[display_mask]
    lo = float(np.percentile(trusted, 2)) if trusted.size else float(loss.min())
    hi = float(np.percentile(trusted, 98)) if trusted.size else float(loss.max())
    scale = max(hi - lo, 1e-12)

    def color(value):
        # Navy -> teal -> yellow, with no plotting-library dependency.
        x = float(np.clip((value - lo) / scale, 0, 1))
        if x < 0.5:
            u = 2 * x
            rgb = (30 * (1-u) + 20*u, 45 * (1-u) + 160*u, 105 * (1-u) + 145*u)
        else:
            u = 2 * x - 1
            rgb = (20 * (1-u) + 250*u, 160 * (1-u) + 215*u, 145 * (1-u) + 70*u)
        return "#%02x%02x%02x" % tuple(int(v) for v in rgb)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#172033}.axis{stroke:#172033;stroke-width:2}</style>',
        f'<text x="{size/2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">{"Smoothed" if smoothed else "Raw"} {metric_name}{" and gradient field" if smoothed else " cell means"}</text>',
    ]
    for i in range(bins):
        for j in range(bins):
            x = margin + (i * cell)
            y = margin + ((bins - 1 - j) * cell)
            if not smoothed and not observed[i, j]:
                fill = "#e5e7eb"
            else:
                fill = color(loss[i, j])
            opacity = (
                0.25
                if smoothed and confidence_shading and not reliable[i, j]
                else 1.0
            )
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell+0.2:.2f}" height="{cell+0.2:.2f}" fill="{fill}" opacity="{opacity}"/>'
            )
    # Normalize every displayed arrow independently: direction is the message;
    # magnitude remains available in the table and JSON.
    arrow_length = cell * arrow_stride * 0.32
    if smoothed:
        for i in range(arrow_stride // 2, bins, arrow_stride):
            for j in range(arrow_stride // 2, bins, arrow_stride):
                if confidence_shading and not reliable[i, j]:
                    continue
                gx, gy = grad_d[i, j], grad_c[i, j]
                norm = float(np.hypot(gx, gy))
                if norm <= 1e-12:
                    continue
                x = margin + (i + 0.5) * cell
                y = margin + (bins - j - 0.5) * cell
                dx = arrow_length * gx / norm
                dy = -arrow_length * gy / norm
                x2, y2 = x + dx, y + dy
                parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="white" stroke-width="2.2"/>')
                # Open chevron arrowhead avoids renderer-dependent filled markers.
                angle = np.arctan2(dy, dx)
                for offset in (-0.55, 0.55):
                    hx = x2 - 7 * np.cos(angle + offset)
                    hy = y2 - 7 * np.sin(angle + offset)
                    parts.append(f'<line x1="{x2:.2f}" y1="{y2:.2f}" x2="{hx:.2f}" y2="{hy:.2f}" stroke="white" stroke-width="2.2"/>')
    right = margin + plot
    bottom = margin + plot
    parts.extend([
        f'<line class="axis" x1="{margin}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<line class="axis" x1="{margin}" y1="{margin}" x2="{margin}" y2="{bottom}"/>',
        f'<text x="{size/2}" y="{size-12}" text-anchor="middle" font-size="16">discrete corruption d</text>',
        f'<text x="18" y="{size/2}" text-anchor="middle" font-size="16" transform="rotate(-90 18 {size/2})">continuous degradation c</text>',
        f'<text x="{margin}" y="{bottom+20}" text-anchor="middle" font-size="12">0</text>',
        f'<text x="{right}" y="{bottom+20}" text-anchor="middle" font-size="12">1</text>',
        f'<text x="{margin-12}" y="{bottom+4}" text-anchor="end" font-size="12">0</text>',
        f'<text x="{margin-12}" y="{margin+4}" text-anchor="end" font-size="12">1</text>',
        f'<text x="{right}" y="42" text-anchor="end" font-size="12">{metric_name} range {lo:.3g} – {hi:.3g}{"; faded = low confidence" if smoothed and confidence_shading else ""}</text>',
        '</svg>',
    ])
    return "".join(parts)


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

    def __init__(self, bins=100, smoothing_sigma=3.0, min_effective_count=5.0):
        super().__init__()
        self.bins = int(bins)
        self.smoothing_sigma = float(smoothing_sigma)
        self.min_effective_count = float(min_effective_count)
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
        # Nadaraya-Watson smoothing: sparse and empty cells contribute in
        # proportion to their evidence instead of being filled as observations.
        weighted_loss = np.nan_to_num(mean) * count
        smooth_weight = gaussian_filter(count, self.smoothing_sigma, mode="nearest")
        smooth_sum = gaussian_filter(
            weighted_loss, self.smoothing_sigma, mode="nearest"
        )
        smooth = np.divide(
            smooth_sum,
            smooth_weight,
            out=np.zeros_like(smooth_sum),
            where=smooth_weight > 1e-12,
        )
        spacing = 1 / self.bins
        grad_d, grad_c = np.gradient(smooth, spacing, spacing)
        centers = ((np.arange(self.bins) + 0.5) / self.bins).tolist()
        return {
            "updates": int(self.updates.item()),
            "centers": centers,
            "count": count.tolist(),
            "mean_loss": mean.tolist(),
            "variance": variance.tolist(),
            "standard_error": np.sqrt(
                np.divide(variance, count, out=np.zeros_like(variance), where=count > 0)
            ).tolist(),
            "smoothed_effective_count": smooth_weight.tolist(),
            "reliable": (smooth_weight >= self.min_effective_count).tolist(),
            "smoothed_loss": smooth.tolist(),
            "grad_discrete": grad_d.tolist(),
            "grad_continuous": grad_c.tolist(),
            "gradient_magnitude": np.hypot(grad_d, grad_c).tolist(),
        }
