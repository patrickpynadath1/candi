"""Apply an adaptive corruption clock to a Gaussian path and its derivatives."""

from torch import nn

from core.noise_schedules.gaussian_path import GaussianPath
from core.noise_schedules.log_linear import LogLinear
from .sflm import SFLMAdaptiveSchedule
from .cdcd import CDCDAdaptiveSchedule


class AdaptiveGaussianSchedule(nn.Module):
    clock_type = None

    def __init__(self, base, **clock_options):
        super().__init__()
        if not hasattr(base, "representation_requirements"):
            raise TypeError("adaptive Gaussian schedule requires a Gaussian-path base")
        self.base = base
        self.clock = self.clock_type(LogLinear(epsilon=0), **clock_options)

    @property
    def representation_requirements(self):
        return self.base.representation_requirements

    @property
    def loss_signal(self):
        return self.clock.loss_signal

    def training_time(self, time):
        return self.base.training_time(time)

    def forward(self, time):
        keep, derivative = self.clock(time)
        progress = 1 - keep
        # Reference S-FLM's pointwise CDF can extrapolate below zero. Keep that
        # clock for fidelity, but Gaussian base paths require valid coordinates.
        bounded = progress.clamp(0, 1)
        speed = -derivative * ((progress >= 0) & (progress <= 1))
        path = self.base(bounded)
        return GaussianPath(
            path.alpha, path.sigma, path.dalpha_dt * speed, path.dsigma_dt * speed
        )

    def observe(self, time, loss, step, *, training=False):
        self.clock.observe(time, loss, step, training=training and self.training)

    def snapshot(self):
        result = self.clock.snapshot()
        result["physical_coordinate"] = "base path progress"
        return result


class SFLMGaussianSchedule(AdaptiveGaussianSchedule):
    clock_type = SFLMAdaptiveSchedule


class CDCDGaussianSchedule(AdaptiveGaussianSchedule):
    clock_type = CDCDAdaptiveSchedule
