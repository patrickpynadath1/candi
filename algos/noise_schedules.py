"""Composable diffusion schedules, including S-FLM adaptive remapping."""

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


class NoiseSchedule(torch.nn.Module):
    def forward(self, t):
        return self.alpha_prime_t(t), self.alpha_t(t)

    def record_time_loss_pair(self, t, loss, step):
        del t, loss, step


class LogLinear(NoiseSchedule):
    type = "log-linear"

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def alpha_t(self, t):
        return self.eps + (1 - self.eps) * (1 - t)

    def alpha_prime_t(self, t):
        return torch.full_like(t, -(1 - self.eps))


class CosineSquared(NoiseSchedule):
    type = "cosine-squared"

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

    @property
    def type(self):
        """Preserve the transition-family identity through composition."""
        return self.base.type

    def alpha_t(self, t):
        return (self.scale * self.base.alpha_t(t) + self.offset).clamp_min(self.eps)

    def alpha_prime_t(self, t):
        return self.scale * self.base.alpha_prime_t(t)


class AdaptiveSchedule(NoiseSchedule):
    """Buffered spline refitting from the official S-FLM implementation."""

    def __init__(
        self, base, buffer_size, refit_every, n_grid, n_knots,
        spline_degree, ridge_alpha, uniform_mix, max_steps, warmup_steps,
        ema,
    ):
        super().__init__()
        self.base = base
        self.buffer_size = buffer_size
        self.refit_every = refit_every
        self.n_knots = n_knots
        self.spline_degree = spline_degree
        self.ridge_alpha = ridge_alpha
        self.uniform_mix = uniform_mix
        self.warmup_steps = warmup_steps
        self.ema = ema
        self.register_buffer("t_buf", torch.zeros(buffer_size, dtype=torch.float64))
        self.register_buffer("loss_buf", torch.zeros(buffer_size, dtype=torch.float64))
        self.register_buffer("buf_pos", torch.tensor(0, dtype=torch.long))
        self.register_buffer("alpha_vals", torch.zeros(n_grid, dtype=torch.float64))
        self.register_buffer("has_schedule", torch.tensor(False))
        self.register_buffer("ema_alpha_vals", torch.zeros(n_grid, dtype=torch.float64))
        self.register_buffer("refit_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_refit_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("loss_profile", torch.zeros(n_grid, dtype=torch.float64))
        self.register_buffer("sampling_density", torch.zeros(n_grid, dtype=torch.float64))
        self._grid = np.linspace(0, 1, n_grid)
        self._alpha_spline = None
        self._dalpha_spline = None

    @property
    def type(self):
        """Preserve the transition-family identity through composition."""
        return self.base.type

    def record_time_loss_pair(self, t, loss, step):
        if step < self.warmup_steps:
            return
        n = len(t)
        pos = self.buf_pos.item()
        end = pos + n
        t_val = t.detach().to(self.t_buf)
        loss_val = loss.detach().to(self.loss_buf)
        if end <= self.buffer_size:
            self.t_buf[pos:end] = t_val
            self.loss_buf[pos:end] = loss_val
        else:
            first = self.buffer_size - pos
            self.t_buf[pos:] = t_val[:first]
            self.loss_buf[pos:] = loss_val[:first]
            self.t_buf[: n - first] = t_val[first:]
            self.loss_buf[: n - first] = loss_val[first:]
        self.buf_pos.fill_(end % self.buffer_size)
        buffer_full = end >= self.buffer_size or self.has_schedule.item()
        if (
            buffer_full
            and step % self.refit_every == 0
            and step != self.last_refit_step.item()
        ):
            self._refit()
            self.last_refit_step.fill_(step)

    def _refit(self):
        model = make_pipeline(
            SplineTransformer(
                n_knots=self.n_knots,
                degree=self.spline_degree,
                extrapolation="continue",
            ),
            Ridge(alpha=self.ridge_alpha),
        )
        model.fit(
            self.t_buf.cpu().numpy().reshape(-1, 1),
            self.loss_buf.cpu().numpy(),
        )
        smooth_loss = model.predict(self._grid.reshape(-1, 1))
        importance = np.maximum(np.gradient(smooth_loss, self._grid), 0)
        importance = (1 - self.uniform_mix) * importance + self.uniform_mix
        self.loss_profile.copy_(torch.from_numpy(smooth_loss))
        self.sampling_density.copy_(torch.from_numpy(importance / importance.sum()))
        cdf = np.cumsum(importance)
        cdf /= cdf[-1]
        remapped_t = PchipInterpolator(cdf, self._grid)(self._grid)
        alpha = self.base.alpha_t(torch.as_tensor(remapped_t)).cpu().numpy()
        alpha_tensor = torch.from_numpy(alpha).to(self.ema_alpha_vals)
        self.refit_count += 1
        if self.ema > 0:
            self.ema_alpha_vals.mul_(self.ema).add_(alpha_tensor, alpha=1 - self.ema)
            correction = 1 - self.ema ** self.refit_count.item()
            alpha = (self.ema_alpha_vals / correction).cpu().numpy()
        self._alpha_spline = PchipInterpolator(self._grid, alpha)
        self._dalpha_spline = self._alpha_spline.derivative()
        self.alpha_vals.copy_(torch.from_numpy(alpha))
        self.has_schedule.fill_(True)

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        result = super().load_state_dict(state_dict, strict=strict, *args, **kwargs)
        if self.has_schedule.item():
            alpha = self.alpha_vals.cpu().numpy()
            self._alpha_spline = PchipInterpolator(self._grid, alpha)
            self._dalpha_spline = self._alpha_spline.derivative()
        return result

    def _evaluate(self, spline, base_fn, t):
        if spline is None and self.has_schedule.item():
            alpha = self.alpha_vals.cpu().numpy()
            self._alpha_spline = PchipInterpolator(self._grid, alpha)
            self._dalpha_spline = self._alpha_spline.derivative()
            spline = (
                self._alpha_spline
                if base_fn.__name__ == "alpha_t"
                else self._dalpha_spline
            )
        if spline is None:
            return base_fn(t)
        values = spline(t.detach().cpu().numpy())
        return torch.as_tensor(values, dtype=t.dtype, device=t.device)

    def alpha_t(self, t):
        return self._evaluate(self._alpha_spline, self.base.alpha_t, t)

    def alpha_prime_t(self, t):
        return self._evaluate(self._dalpha_spline, self.base.alpha_prime_t, t)

    def snapshot(self):
        grid = torch.as_tensor(self._grid, dtype=torch.float64)
        base_alpha = self.base.alpha_t(grid).detach().cpu()
        alpha = self.alpha_t(grid).detach().cpu()
        dalpha = self.alpha_prime_t(grid).detach().cpu()
        return {
            "refit_count": int(self.refit_count.item()),
            "step": int(self.last_refit_step.item()),
            "grid": grid.tolist(),
            "base_alpha": base_alpha.tolist(),
            "alpha": alpha.tolist(),
            "dalpha_dt": dalpha.tolist(),
            "loss_profile": self.loss_profile.detach().cpu().tolist(),
            "sampling_density": self.sampling_density.detach().cpu().tolist(),
        }


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
            base=base,
            buffer_size=cfg.adaptive_buffer_size,
            refit_every=cfg.adaptive_refit_every,
            n_grid=cfg.adaptive_n_grid,
            n_knots=cfg.adaptive_n_knots,
            spline_degree=cfg.adaptive_spline_degree,
            ridge_alpha=cfg.adaptive_ridge_alpha,
            uniform_mix=cfg.adaptive_uniform_mix,
            max_steps=config.trainer.max_steps,
            warmup_steps=cfg.adaptive_warmup_steps,
            ema=cfg.adaptive_ema,
        )
    return base
