"""LangFlow's clipped Gumbel quantiles and information-profile schedule fit."""

import math

import torch
from torch import nn

from core.representations.spec import RepresentationRequirements
from .gaussian_path import GaussianPath


class GumbelSchedule(nn.Module):
    """VP Gaussian path with gamma=log(sigma²/alpha²) at Gumbel quantiles.

    Schedule parameters are fitted only by detached CE feedback, not by the
    denoiser optimizer. The existing observe hook performs a separate SGD step
    on H(gamma)=entropy*CDF(gamma). Log coordinates enforce positive scale and
    entropy. All fitted state is checkpointed as buffers, like adaptive clocks.
    """

    representation_requirements = RepresentationRequirements()
    loss_signal = "cross_entropy"

    def __init__(
        self,
        loc=4.723,
        scale=0.852,
        entropy=7.02,
        cutoff=1e-5,
        learnable=True,
        learning_rate=1e-3,
    ):
        super().__init__()
        if not all(
            math.isfinite(x) for x in (loc, scale, entropy, cutoff, learning_rate)
        ):
            raise ValueError("Gumbel schedule settings must be finite")
        if (
            scale <= 0
            or entropy <= 0
            or not torch.finfo(torch.float64).eps <= cutoff < 0.5
            or learning_rate <= 0
        ):
            raise ValueError(
                "require positive scale, entropy, learning rate and cutoff in (0,.5)"
            )
        self.cutoff = float(cutoff)
        self.learnable, self.learning_rate = bool(learnable), float(learning_rate)
        self.register_buffer(
            "fit_parameters",
            torch.tensor(
                [loc, math.log(scale), math.log(entropy)], dtype=torch.float64
            ),
        )
        self.register_buffer("fit_steps", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_fit_loss", torch.tensor(0.0, dtype=torch.float64))

    @property
    def loc(self):
        return self.fit_parameters[0]

    @property
    def scale(self):
        return self.fit_parameters[1].exp()

    @property
    def entropy(self):
        return self.fit_parameters[2].exp()

    def training_time(self, time):
        return time

    def gamma(self, time):
        if (
            not time.is_floating_point()
            or not torch.isfinite(time).all()
            or torch.any((time < 0) | (time > 1))
        ):
            raise ValueError("Gumbel schedule time must be finite and in [0,1]")
        # The released reference uses upper bound loc-scale*log(cutoff), whose
        # exact CDF is exp(-cutoff). Compute in FP64 to keep both endpoints safe.
        coordinate = time.double()
        q = coordinate.clamp(self.cutoff, math.exp(-self.cutoff))
        parameters = self.fit_parameters.detach().clone().to(time.device)
        gamma = parameters[0] - parameters[1].exp() * (-q.log()).log()
        derivative = parameters[1].exp() / (-q * q.log())
        derivative = torch.where(
            (coordinate >= self.cutoff) & (coordinate <= math.exp(-self.cutoff)),
            derivative,
            0,
        )
        dtype = torch.float64 if time.dtype == torch.float64 else torch.float32
        return gamma.to(dtype), derivative.to(dtype)

    def forward(self, time):
        gamma, derivative = self.gamma(time)
        alpha_sq, sigma_sq = torch.sigmoid(-gamma), torch.sigmoid(gamma)
        alpha, sigma = alpha_sq.sqrt(), sigma_sq.sqrt()
        return GaussianPath(
            alpha,
            sigma,
            -0.5 * derivative * sigma_sq * alpha,
            0.5 * derivative * alpha_sq * sigma,
        )

    @torch.no_grad()
    def observe(self, time, loss, step, *, training=False):
        if not training or not self.training or not self.learnable:
            return
        if time.shape != loss.shape or time.ndim != 1:
            raise ValueError("schedule feedback must be matching per-example vectors")
        valid = torch.isfinite(time) & torch.isfinite(loss) & (loss >= 0)
        gamma = self.gamma(time[valid].detach())[0].detach().to(self.fit_parameters)
        target = loss[valid].detach().to(self.fit_parameters)
        # Detaching gamma is essential: differentiating CDF(icdf(q)) would
        # cancel the very location/scale gradients that fit the loss profile.
        with torch.enable_grad():
            parameters = self.fit_parameters.detach().clone().requires_grad_()
            standardized = (gamma - parameters[0]) / parameters[1].exp()
            predicted = parameters[2].exp() * (-(-standardized).exp()).exp()
            squared = (predicted - target).square().sum()
            (gradient,) = torch.autograd.grad(squared, parameters)
        statistics = torch.cat(
            (gradient, squared.detach().reshape(1), gradient.new_tensor([len(target)]))
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(statistics)
        count = statistics[-1]
        if count == 0 or not torch.isfinite(statistics).all():
            return
        self.fit_parameters.add_(statistics[:3] / count, alpha=-self.learning_rate)
        self.last_fit_loss.copy_(statistics[3] / count)
        self.fit_steps.add_(1)

    def snapshot(self):
        return dict(
            loc=self.loc.item(),
            scale=self.scale.item(),
            entropy=self.entropy.item(),
            fit_steps=self.fit_steps.item(),
            fit_loss=self.last_fit_loss.item(),
        )
