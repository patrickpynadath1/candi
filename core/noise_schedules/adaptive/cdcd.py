import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from .clock import BufferedAdaptiveSchedule


class CDCDAdaptiveSchedule(BufferedAdaptiveSchedule):
    """Buffered CDCD-style monotone fit in physical corruption coordinates."""

    def __init__(self, *args, bins=100, max_importance=100.0, **kwargs):
        kwargs.setdefault("coordinate", "physical")
        kwargs.setdefault("loss_signal", "cross_entropy")
        kwargs.setdefault("uniform_mix", 0.01)
        super().__init__(*args, **kwargs)
        if self.coordinate != "physical":
            raise ValueError("CDCD time warping requires physical coordinates")
        if not np.isfinite(max_importance) or max_importance <= 0:
            raise ValueError("max_importance must be finite and positive")
        if bins < 2:
            raise ValueError("CDCD time warping requires at least two bins")
        self.cdcd_bins = int(bins)
        self.cdcd_max_importance = float(max_importance)
        self.register_buffer(
            "importance_buf", torch.ones(self.buffer_size, dtype=torch.float64)
        )

    def _current_density(self, progress):
        if not self.has_schedule.item():
            return torch.ones_like(progress)
        grid = torch.linspace(
            0,
            1,
            len(self.sampling_density),
            dtype=progress.dtype,
            device=progress.device,
        )
        index = torch.searchsorted(grid, progress).clamp(1, len(grid) - 1)
        left, right = grid[index - 1], grid[index]
        weight = (progress - left) / (right - left).clamp_min(1e-12)
        density = (
            self.sampling_density.to(progress)[index - 1] * (1 - weight)
            + self.sampling_density.to(progress)[index] * weight
        )
        # sampling_density is stored as grid-point mass; convert it to a
        # relative continuous density whose uniform value is one.
        return density * len(self.sampling_density)

    def record_time_loss_pair(self, t, loss, step):
        if step < self.warmup_steps:
            return
        clean = self.base.alpha_t(torch.zeros_like(t))
        noisy = self.base.alpha_t(torch.ones_like(t))
        progress = ((clean - self.alpha_t(t)) / (clean - noisy)).clamp(0, 1)
        importance = (
            self._current_density(progress)
            .reciprocal()
            .clamp_max(self.cdcd_max_importance)
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            triples = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                triples,
                (
                    progress.detach().cpu(),
                    loss.detach().cpu(),
                    importance.detach().cpu(),
                ),
            )
            progress = torch.cat([item[0] for item in triples])
            loss = torch.cat([item[1] for item in triples])
            importance = torch.cat([item[2] for item in triples])
        finite = (
            torch.isfinite(progress) & torch.isfinite(loss) & torch.isfinite(importance)
        )
        progress, loss, importance = (
            progress[finite],
            loss[finite],
            importance[finite],
        )
        if not len(progress):
            return
        self.num_observed.add_(len(progress)).clamp_max_(self.buffer_size)
        progress = progress[-self.buffer_size :]
        loss = loss[-self.buffer_size :]
        importance = importance[-self.buffer_size :]
        n = len(progress)
        pos = self.buf_pos.item()
        end = pos + n

        def write_ring(buffer, values):
            values = values.detach().to(buffer)
            if end <= self.buffer_size:
                buffer[pos:end] = values
            else:
                first = self.buffer_size - pos
                buffer[pos:] = values[:first]
                buffer[: n - first] = values[first:]

        write_ring(self.t_buf, progress)
        write_ring(self.loss_buf, loss)
        write_ring(self.importance_buf, importance)
        self.buf_pos.fill_(end % self.buffer_size)
        if (
            self.num_observed.item() >= self.buffer_size
            and step % self.refit_every == 0
            and step != self.last_refit_step.item()
        ):
            self._refit()
            self.last_refit_step.fill_(step)

    def _refit(self):
        progress = self.t_buf.cpu().numpy()
        losses = self.loss_buf.cpu().numpy()
        weights = self.importance_buf.cpu().numpy()
        estimator = IsotonicRegression(increasing=True, y_min=0.0, out_of_bounds="clip")
        estimator.fit(progress, losses, sample_weight=weights)
        bin_grid = np.linspace(0, 1, self.cdcd_bins + 1)
        bin_loss = estimator.predict(bin_grid)
        bin_loss = np.maximum.accumulate(np.maximum(bin_loss, 0))
        span = bin_loss[-1] - bin_loss[0]
        if not np.isfinite(span) or span <= np.finfo(np.float64).eps:
            cdf = bin_grid
        else:
            cdf = (bin_loss - bin_loss[0]) / span
            # CDCD's uniform mixture also supplies a strictly positive minimum
            # bin width, making the CDF stable and exactly invertible.
            cdf = (1 - self.uniform_mix) * cdf + self.uniform_mix * bin_grid
        cdf[0], cdf[-1] = 0.0, 1.0
        remapped_t = np.interp(self._grid, cdf, bin_grid)
        hi = self.base.alpha_t(torch.tensor(0.0, dtype=torch.float64)).item()
        lo = self.base.alpha_t(torch.tensor(1.0, dtype=torch.float64)).item()
        alpha = hi + (lo - hi) * remapped_t
        alpha[0], alpha[-1] = hi, lo
        alpha_tensor = torch.from_numpy(alpha).to(self.ema_alpha_vals)
        self.refit_count += 1
        if self.ema > 0:
            self.ema_alpha_vals.mul_(self.ema).add_(alpha_tensor, alpha=1 - self.ema)
            correction = 1 - self.ema ** self.refit_count.item()
            alpha = (self.ema_alpha_vals / correction).cpu().numpy()
            alpha[0], alpha[-1] = hi, lo
        loss_profile = np.interp(self._grid, bin_grid, bin_loss)
        density = np.gradient(cdf, bin_grid)
        density = np.interp(self._grid, bin_grid, density)
        density = np.maximum(density, np.finfo(np.float64).eps)
        self.loss_profile.copy_(torch.from_numpy(loss_profile))
        self.sampling_density.copy_(torch.from_numpy(density / density.sum()))
        self._alpha_spline = PchipInterpolator(self._grid, alpha)
        self._dalpha_spline = self._alpha_spline.derivative()
        self.alpha_vals.copy_(torch.from_numpy(alpha))
        self.has_schedule.fill_(True)
