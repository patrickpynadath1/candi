"""Learned importance clock shared by Gaussian and absorbing information paths."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .gaussian_path import path_from_gamma


class PiecewiseInverseCDFClock(nn.Module):
    def __init__(self, lower=-13.3, upper=5.0, bins=64, uniform_mix=0.01):
        super().__init__()
        if not math.isfinite(lower + upper) or upper <= lower or bins < 1 or not 0 < uniform_mix <= 1:
            raise ValueError("require ordered finite endpoints, positive bins and mixture")
        self.gamma_min = nn.Parameter(torch.tensor(float(lower), dtype=torch.float64))
        self.width_raw = nn.Parameter(torch.tensor(math.log(math.expm1(upper-lower)), dtype=torch.float64))
        self.clock_logits = nn.Parameter(torch.zeros(bins, dtype=torch.float64))
        self.uniform_mix = float(uniform_mix)

    def bounds(self):
        return self.gamma_min, self.gamma_min + F.softplus(self.width_raw)

    def probabilities(self):
        return (1-self.uniform_mix)*self.clock_logits.softmax(0) + self.uniform_mix/self.clock_logits.numel()

    def inverse_cdf(self, time):
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("clock time must lie in [0,1]")
        p = self.probabilities()
        cdf = p.cumsum(0)
        index = torch.searchsorted(cdf.detach(), time.to(p).contiguous()).clamp_max(p.numel()-1)
        left = torch.cat((p.new_zeros(1), cdf[:-1]))[index]
        value = ((index + (time.to(p)-left)/p[index])/p.numel()).clamp(0, 1)
        value = torch.where(time == 0, 0., torch.where(time == 1, 1., value))
        return value, 1/(p.numel()*p[index])

    def gamma_at(self, value):
        lower, upper = self.bounds()
        return lower + (upper-lower)*value

    def draw(self, count):
        p = self.probabilities()
        time = (torch.arange(count, device=p.device, dtype=p.dtype)+torch.rand(count, device=p.device, dtype=p.dtype))/count
        time = time[torch.randperm(count, device=p.device)]
        with torch.no_grad():
            value, _ = self.inverse_cdf(time)
        index = (value*p.numel()).long().clamp_max(p.numel()-1)
        return value, p.numel()*p[index]

    @staticmethod
    def variance_surrogate(estimate, density):
        score = -estimate.detach().square()*density.log()
        return score-score.detach()


class ClockGaussianSchedule(nn.Module):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def forward(self, time):
        value, derivative = self.clock.inverse_cdf(time)
        lower, upper = self.clock.bounds()
        return path_from_gamma(self.clock.gamma_at(value), (upper-lower)*derivative)


class ClockKeepSchedule(nn.Module):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def forward(self, time):
        value, derivative = self.clock.inverse_cdf(time)
        return 1-value, -derivative
