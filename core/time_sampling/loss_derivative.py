"""Bucketed loss-derivative time sampling, SDM Appendix J.1."""

import torch
from torch import nn
from .simplex_grid import SimplexInferenceGrid
from .uniform import UniformTimeSampler


def interpolate(x, xp, fp):
    index = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, len(xp) - 1)
    lo, hi = xp[index - 1], xp[index]
    weight = ((x - lo) / (hi - lo).clamp_min(1e-15)).clamp(0, 1)
    return fp[index - 1] + weight * (fp[index] - fp[index - 1])


class LossDerivativeTimeSampler(nn.Module):
    def __init__(
        self,
        adaptive=False,
        buffer_size=65536,
        bins=50,
        grid_size=1000,
        warmup=1000,
        refit_every=50,
        momentum=0.9,
        epsilon=1e-5,
        antithetic=True,
    ):
        super().__init__()
        if (
            buffer_size < 2
            or bins < 2
            or grid_size < 2
            or refit_every <= 0
            or warmup < 0
            or not 0 <= momentum < 1
        ):
            raise ValueError("invalid loss-derivative time sampler settings")
        self.uniform = UniformTimeSampler(epsilon, antithetic)
        self.adaptive = adaptive
        self.bins, self.warmup, self.refit_every, self.momentum = (
            bins,
            warmup,
            refit_every,
            momentum,
        )
        for name, value in {
            "times": torch.zeros(buffer_size),
            "losses": torch.zeros(buffer_size),
            "position": torch.tensor(0),
            "count": torch.tensor(0),
            "last_refit": torch.tensor(-1),
            "ready": torch.tensor(False),
            "grid": torch.linspace(0, 1, grid_size),
            "cdf": torch.linspace(0, 1, grid_size),
            "density": torch.ones(grid_size),
            "profile": torch.zeros(grid_size),
        }.items():
            self.register_buffer(name, value, persistent=adaptive)

    def sample(self, count, device, **context):
        uniform = self.uniform.sample(count, device, **context)
        return self(uniform) if self.adaptive else uniform

    def forward(self, uniform):
        return interpolate(uniform, self.cdf.to(uniform), self.grid.to(uniform))

    @torch.no_grad()
    def observe(self, times, token_loss, mask, step):
        if not self.adaptive:
            return
        weights = torch.ones_like(token_loss) if mask is None else mask.to(token_loss)
        counts = weights.sum(-1)
        loss = (token_loss.detach() * weights).sum(-1) / counts.clamp_min(1)
        valid = (counts > 0) & (torch.isfinite(loss) & torch.isfinite(times))
        pairs = torch.stack((times.detach()[valid], loss[valid]), -1)
        if torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, pairs.cpu())
            pairs = torch.cat(gathered).to(self.times)
        pairs = pairs[-len(self.times) :]
        n = len(pairs)
        if not n:
            return
        indices = (torch.arange(n, device=self.times.device) + self.position) % len(
            self.times
        )
        self.times[indices], self.losses[indices] = pairs[:, 0], pairs[:, 1]
        self.position.copy_((self.position + n) % len(self.times))
        self.count.copy_((self.count + n).clamp_max(len(self.times)))
        if (
            step >= self.warmup
            and step % self.refit_every == 0
            and step != self.last_refit
        ):
            self.refit()
            self.last_refit.fill_(step)

    @torch.no_grad()
    def refit(self):
        n = int(self.count)
        ids = (self.times[:n] * self.bins).long().clamp(0, self.bins - 1)
        total = torch.zeros(self.bins, device=self.times.device)
        count = torch.zeros_like(total)
        total.scatter_add_(0, ids, self.losses[:n])
        count.scatter_add_(0, ids, torch.ones_like(self.losses[:n]))
        observed = count > 0
        if observed.sum() < 2:
            return
        centers = (torch.arange(self.bins, device=total.device) + 0.5) / self.bins
        profile = interpolate(
            self.grid, centers[observed], (total / count.clamp_min(1))[observed]
        )
        density = torch.gradient(profile, spacing=(self.grid,))[0].clamp_min(0)
        integral = torch.trapezoid(density, self.grid)
        if integral <= 0:
            return
        density = density / integral
        if self.ready:
            density = self.momentum * self.density + (1 - self.momentum) * density
        self.density.copy_(density)
        self.profile.copy_(profile)
        areas = (density[1:] + density[:-1]) * torch.diff(self.grid) / 2
        self.cdf.copy_(torch.cat((areas.new_zeros(1), areas.cumsum(0))) / areas.sum())
        self.ready.fill_(True)

    def sampling_grid(self, steps, schedule, device):

        return SimplexInferenceGrid(schedule, self).build(steps, device)
