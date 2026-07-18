"""Composable diffusion schedules, including S-FLM adaptive remapping."""

import torch


class NoiseSchedule(torch.nn.Module):
    def forward(self, t):
        return self.alpha_prime_t(t), self.alpha_t(t)

    def record_time_loss_pair(self, t, loss, step):
        del t, loss, step


class LogLinear(NoiseSchedule):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def alpha_t(self, t):
        return self.eps + (1 - self.eps) * (1 - t)

    def alpha_prime_t(self, t):
        return torch.full_like(t, -(1 - self.eps))


class CosineSquared(NoiseSchedule):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def alpha_t(self, t):
        angle = torch.pi / 2 * (1 - t)
        return self.eps + (1 - self.eps) * torch.sin(angle).square()

    def alpha_prime_t(self, t):
        angle = torch.pi / 2 * (1 - t)
        return -(1 - self.eps) * torch.pi * torch.sin(angle) * torch.cos(angle)


class TruncatedSchedule(NoiseSchedule):
    def __init__(self, base, alpha_min, alpha_max, eps=1e-6):
        super().__init__()
        if not 0 <= alpha_min < alpha_max <= 1:
            raise ValueError("expected 0 <= alpha_min < alpha_max <= 1")
        self.base = base
        self.eps = eps
        hi = base.alpha_t(torch.tensor(0.0)).item()
        lo = base.alpha_t(torch.tensor(1.0)).item()
        self.scale = (alpha_max - alpha_min) / (hi - lo)
        self.offset = alpha_min - self.scale * lo

    def alpha_t(self, t):
        return (self.scale * self.base.alpha_t(t) + self.offset).clamp_min(self.eps)

    def alpha_prime_t(self, t):
        return self.scale * self.base.alpha_prime_t(t)


class AdaptiveSchedule(NoiseSchedule):
    """Checkpointable loss-gradient importance remapping on a fixed grid."""

    def __init__(self, base, n_grid=1000, uniform_mix=1e-3, ema=0.0):
        super().__init__()
        self.base, self.uniform_mix, self.ema = base, uniform_mix, ema
        self.register_buffer("grid", torch.linspace(0, 1, n_grid))
        self.register_buffer("mapped_t", torch.linspace(0, 1, n_grid))
        self.register_buffer("loss_sum", torch.zeros(n_grid))
        self.register_buffer("loss_count", torch.zeros(n_grid))

    def record_time_loss_pair(self, t, loss, step=None):
        del step
        idx = (
            (t.detach() * (self.grid.numel() - 1))
            .long()
            .clamp(0, self.grid.numel() - 1)
        )
        self.loss_sum.scatter_add_(
            0, idx.cpu().to(self.loss_sum.device), loss.detach().cpu().to(self.loss_sum)
        )
        self.loss_count.scatter_add_(
            0,
            idx.cpu().to(self.loss_count.device),
            torch.ones_like(loss.detach().cpu()).to(self.loss_count),
        )
        means = self.loss_sum / self.loss_count.clamp_min(1)
        grad = torch.relu(torch.diff(means, prepend=means[:1])) + self.uniform_mix
        cdf = torch.cumsum(grad, 0)
        cdf /= cdf[-1]
        remapped = torch.searchsorted(cdf, self.grid).float() / (self.grid.numel() - 1)
        self.mapped_t.lerp_(remapped, 1 - self.ema)

    def _map(self, t):
        pos = t.clamp(0, 1) * (self.grid.numel() - 1)
        lo = pos.long().clamp_max(self.grid.numel() - 2)
        frac = pos - lo
        mapped = self.mapped_t.to(t.device)
        return mapped[lo] + frac * (mapped[lo + 1] - mapped[lo])

    def alpha_t(self, t):
        return self.base.alpha_t(self._map(t))

    def alpha_prime_t(self, t):
        # Stable finite difference includes the derivative of the learned map.
        h = 1e-4
        return (
            self.alpha_t((t + h).clamp_max(1)) - self.alpha_t((t - h).clamp_min(0))
        ) / (2 * h)


def get_noise(config):
    cfg = config.noise
    if cfg.type == "log-linear":
        base = LogLinear(cfg.eps)
    elif cfg.type == "cosine-squared":
        base = CosineSquared(cfg.eps)
    else:
        raise ValueError(f"unknown noise schedule: {cfg.type}")
    if (
        getattr(cfg, "alpha_min", None) is not None
        or getattr(cfg, "alpha_max", None) is not None
    ):
        lo = (
            cfg.alpha_min
            if cfg.alpha_min is not None
            else base.alpha_t(torch.tensor(1.0)).item()
        )
        hi = (
            cfg.alpha_max
            if cfg.alpha_max is not None
            else base.alpha_t(torch.tensor(0.0)).item()
        )
        base = TruncatedSchedule(base, lo, hi, cfg.eps)
    if getattr(cfg, "adaptive", False):
        base = AdaptiveSchedule(
            base, cfg.adaptive_n_grid, cfg.adaptive_uniform_mix, cfg.adaptive_ema
        )
    return base
