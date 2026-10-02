"""Monotone VP negative-log-SNR schedule and variance-gradient adaptation."""

from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

from core.representations.spec import RepresentationRequirements
from .gaussian_path import GaussianPath
from core.objectives.reductions import masked_example_mean


@dataclass
class ScheduleGradient:
    """Per-forward hook state; never persisted across batches or checkpoints."""

    weight: torch.Tensor

    def apply(self, gradient):
        return gradient * self.weight

    def observe(self, diffusion, reconstruction_count, selected, loss_mask=None):
        self.weight = 2 * masked_example_mean(diffusion.detach(), loss_mask).double()
        self.weight[:reconstruction_count] = 0
        self.weight[selected] = 0


class LearnedLogSNRSchedule(nn.Module):
    representation_requirements = RepresentationRequirements(unit_normalized=True)

    def __init__(self, gamma_min=-13.3, gamma_max=5.0, hidden_dim=16):
        super().__init__()
        if not math.isfinite(gamma_min + gamma_max) or gamma_max <= gamma_min:
            raise ValueError("gamma endpoints must be finite and increasing")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.gamma_min = nn.Parameter(
            torch.tensor(float(gamma_min), dtype=torch.float64)
        )
        self.gamma_delta_raw = nn.Parameter(
            torch.tensor(
                math.log(math.expm1(gamma_max - gamma_min)), dtype=torch.float64
            )
        )
        self.slope_raw = nn.Parameter(torch.zeros(hidden_dim + 1, dtype=torch.float64))
        self.centers = nn.Parameter(
            torch.linspace(-2, 2, hidden_dim, dtype=torch.float64)
        )

    def shape_and_derivative(self, time):
        time = time.double()
        base = F.softplus(self.slope_raw[0])
        slopes = F.softplus(self.slope_raw[1:])
        offset = F.softplus(self.centers)
        raw = base * time + (
            slopes * (F.softplus(time[:, None] + self.centers) - offset)
        ).sum(-1)
        derivative = base + (slopes * torch.sigmoid(time[:, None] + self.centers)).sum(
            -1
        )
        normalizer = base + (slopes * (F.softplus(1 + self.centers) - offset)).sum()
        return raw / normalizer, derivative / normalizer

    def bounds(self):
        return self.gamma_min, self.gamma_min + F.softplus(self.gamma_delta_raw)

    def gamma(self, time, reconstruction_count=None):
        shape, derivative = self.shape_and_derivative(time)
        adaptation = None
        if reconstruction_count is not None:
            rows = torch.arange(len(time), device=time.device) < reconstruction_count
            shape = torch.where(rows, shape.detach(), shape)
            derivative = torch.where(rows, derivative.detach(), derivative)
            adaptation = ScheduleGradient(torch.ones_like(shape))
            if shape.requires_grad:
                shape.register_hook(adaptation.apply)
                derivative.register_hook(adaptation.apply)
        lower, upper = self.bounds()
        delta = upper - lower
        return lower + delta * shape, delta * derivative, adaptation

    @staticmethod
    def coefficients(gamma, derivative):
        alpha_sq, sigma_sq = torch.sigmoid(-gamma), torch.sigmoid(gamma)
        alpha, sigma = alpha_sq.sqrt(), sigma_sq.sqrt()
        return GaussianPath(
            alpha,
            sigma,
            -0.5 * derivative * sigma_sq * alpha,
            0.5 * derivative * alpha_sq * sigma,
        )

    def forward(self, time):
        gamma, derivative, _ = self.gamma(time)
        return self.coefficients(gamma, derivative)

    def training_time(self, time):
        return time


class PlaidGammaSchedule(LearnedLogSNRSchedule):
    """Normalized positive-weight tanh MLP used by Plaid."""

    def __init__(self, gamma_min, gamma_max, hidden_dim=1024):
        nn.Module.__init__(self)
        if gamma_max <= gamma_min or hidden_dim < 1:
            raise ValueError("require ordered gamma endpoints and positive width")
        self.gamma_min = torch.nn.Parameter(torch.tensor(float(gamma_min)))
        self.gamma_max = torch.nn.Parameter(torch.tensor(float(gamma_max)))
        self.input_weight = torch.nn.Parameter(torch.randn(hidden_dim))
        self.bias = torch.nn.Parameter(torch.randn(hidden_dim))
        self.output_weight = torch.nn.Parameter(torch.randn(hidden_dim))
        self.double()

    def shape_and_derivative(self, t):
        t = t.to(self.gamma_min)
        positive_in = F.softplus(self.input_weight)
        positive_out = F.softplus(self.output_weight)
        activation = torch.tanh((t[:, None] - 0.5) * positive_in + self.bias)
        lower = torch.tanh(self.bias - 0.5 * positive_in)
        upper = torch.tanh(self.bias + 0.5 * positive_in)
        normalizer = ((upper - lower) * positive_out).sum()
        shape = ((activation - lower) * positive_out).sum(-1) / normalizer
        derivative = ((1 - activation.square()) * positive_in * positive_out).sum(
            -1
        ) / normalizer
        return shape, derivative

    def bounds(self):
        return self.gamma_min, self.gamma_max

    def endpoint_parameters(self):
        return [self.gamma_min, self.gamma_max]
