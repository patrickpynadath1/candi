"""Shared observation buffers, schedule state, and checkpoint restoration.

An adaptive noise schedule fits what each time means from (time, loss) pairs of
training batches. Those times come from the training time sampler, which need not
be uniform: ``observe(..., density=d)`` passes each draw's density relative to
uniform, and the fit weights every observation by ``1 / d`` so it targets the
true per-time loss whatever the training time sampler (``density=None`` means
uniform draws, the behaviour before the sampler reported densities).
"""

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from core.noise_schedules.utils import check_unit_time


class BufferedAdaptiveSchedule(torch.nn.Module):
    def __init__(
        self,
        base,
        buffer_size=12800,
        refit_every=50,
        n_grid=1000,
        n_knots=20,
        spline_degree=3,
        ridge_alpha=1e-3,
        uniform_mix=1e-3,
        warmup_steps=1000,
        ema=0.9,
        coordinate="legacy_time",
        loss_signal="objective",
    ):
        super().__init__()
        self.base = KeepRateAdapter(base)
        self.loss_signal = loss_signal
        if loss_signal not in {"objective", "cross_entropy"}:
            raise ValueError("loss_signal must be objective or cross_entropy")
        if buffer_size < 1 or refit_every < 1 or n_grid < 3 or warmup_steps < 0:
            raise ValueError("invalid adaptive buffer, refit, grid, or warmup settings")
        if not 0 < uniform_mix <= 1 or not 0 <= ema < 1:
            raise ValueError("need 0 < uniform_mix <= 1 and 0 <= ema < 1")
        self.buffer_size = buffer_size
        self.refit_every = refit_every
        self.n_knots = n_knots
        self.spline_degree = spline_degree
        self.ridge_alpha = ridge_alpha
        self.uniform_mix = uniform_mix
        self.warmup_steps = warmup_steps
        self.ema = ema
        if coordinate not in {"legacy_time", "physical"}:
            raise ValueError("adaptive coordinate must be legacy_time or physical")
        self.coordinate = coordinate
        self.register_buffer("num_observed", torch.tensor(0, dtype=torch.long))
        self.register_buffer("t_buf", torch.zeros(buffer_size, dtype=torch.float64))
        self.register_buffer("loss_buf", torch.zeros(buffer_size, dtype=torch.float64))
        # 1 / (training time sampler density) of each buffered draw.
        self.register_buffer(
            "draw_weight_buf", torch.ones(buffer_size, dtype=torch.float64)
        )
        self.register_buffer("buf_pos", torch.tensor(0, dtype=torch.long))
        self.register_buffer("alpha_vals", torch.zeros(n_grid, dtype=torch.float64))
        self.register_buffer("has_schedule", torch.tensor(False))
        self.register_buffer("ema_alpha_vals", torch.zeros(n_grid, dtype=torch.float64))
        self.register_buffer("refit_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_refit_step", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("loss_profile", torch.zeros(n_grid, dtype=torch.float64))
        self.register_buffer(
            "sampling_density", torch.zeros(n_grid, dtype=torch.float64)
        )
        self._grid = np.linspace(0, 1, n_grid)
        self._alpha_spline = None
        self._dalpha_spline = None

    @staticmethod
    def draw_weight(density, like):
        """Importance weight ``1 / density`` of each training time draw."""
        if density is None:
            return torch.ones_like(like, dtype=torch.float64)
        return density.detach().to(like.device, torch.float64).reciprocal()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from before draw weights were recorded hold uniform draws.
        key = prefix + "draw_weight_buf"
        if "draw_weight_buf" in self._buffers and key not in state_dict:
            state_dict[key] = torch.ones_like(self.draw_weight_buf)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Recursive parent-module loading bypasses load_state_dict below.
        self._alpha_spline = None
        self._dalpha_spline = None

    def record_time_loss_pair(self, t, loss, step, density=None):
        if step < self.warmup_steps:
            return
        weight = self.draw_weight(density, t)
        if self.coordinate == "physical":
            # Store the corruption at observation time, not a clock whose
            # meaning changes every time the schedule is refitted.
            clean = self.base.alpha_t(torch.zeros_like(t))
            noisy = self.base.alpha_t(torch.ones_like(t))
            t = ((clean - self.alpha_t(t)) / (clean - noisy)).clamp(0, 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            pairs = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(
                pairs, (t.detach().cpu(), loss.detach().cpu(), weight.cpu())
            )
            t = torch.cat([pair[0] for pair in pairs])
            loss = torch.cat([pair[1] for pair in pairs])
            weight = torch.cat([pair[2] for pair in pairs])
        finite = torch.isfinite(t) & torch.isfinite(loss) & torch.isfinite(weight)
        t, loss, weight = t[finite], loss[finite], weight[finite]
        if not len(t):
            return
        self.num_observed.add_(len(t)).clamp_max_(self.buffer_size)
        t, loss = t[-self.buffer_size :], loss[-self.buffer_size :]
        weight = weight[-self.buffer_size :]
        n = len(t)
        pos = self.buf_pos.item()
        end = pos + n
        for buffer, values in (
            (self.t_buf, t),
            (self.loss_buf, loss),
            (self.draw_weight_buf, weight),
        ):
            values = values.detach().to(buffer)
            if end <= self.buffer_size:
                buffer[pos:end] = values
            else:
                first = self.buffer_size - pos
                buffer[pos:] = values[:first]
                buffer[: n - first] = values[first:]
        self.buf_pos.fill_(end % self.buffer_size)
        buffer_full = self.num_observed.item() >= self.buffer_size
        if (
            buffer_full
            and step % self.refit_every == 0
            and step != self.last_refit_step.item()
        ):
            self._refit()
            self.last_refit_step.fill_(step)

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
        snapshot = {
            "refit_count": int(self.refit_count.item()),
            "step": int(self.last_refit_step.item()),
            "grid": grid.tolist(),
            "base_alpha": base_alpha.tolist(),
            "alpha": alpha.tolist(),
            "dalpha_dt": dalpha.tolist(),
            "loss_profile": self.loss_profile.detach().cpu().tolist(),
            "sampling_density": self.sampling_density.detach().cpu().tolist(),
        }
        if self.coordinate == "physical":
            progress = (base_alpha[0] - alpha) / (base_alpha[0] - base_alpha[-1])
            snapshot["coordinate"] = self.coordinate
            snapshot["progress"] = progress.tolist()
            warped_loss = np.interp(
                progress.numpy(), self._grid, self.loss_profile.cpu().numpy()
            )
            snapshot["warped_loss_profile"] = warped_loss.tolist()
            snapshot["linear_loss_target"] = np.linspace(
                warped_loss[0], warped_loss[-1], len(warped_loss)
            ).tolist()
        return snapshot

    output_kind = "keep_rate"

    def forward(self, time):
        check_unit_time(time, "schedule time must be finite and in [0, 1]")
        keep, derivative = self.alpha_t(time), self.alpha_prime_t(time)
        clean = self.base.alpha_t(torch.zeros_like(time))
        noisy = self.base.alpha_t(torch.ones_like(time))
        # Preserve the reference fitted spline internally, while its legacy CDF
        # extrapolation must not emit invalid transition probabilities publicly.
        # Positive training times must stay strictly noisy: objectives contain
        # 1 / (1 - keep). Preserve the exact clean endpoint only at t=0.
        upper = torch.where(
            (time > 0) & (clean >= 1),
            torch.nextafter(clean, noisy),
            clean,
        )
        within = (keep >= noisy) & (keep <= upper)
        return keep.clamp(min=noisy, max=upper), derivative * within

    @torch.no_grad()
    def observe(self, time, loss, step, *, training=False, density=None):
        """Accept detached, masked per-example feedback only during training.

        ``density`` is the training time sampler's density of each draw relative
        to uniform (None: uniform draws); rows without a finite positive density
        (e.g. a point-mass estimator) are dropped.
        """
        if training and self.training:
            if time.shape != loss.shape or time.ndim != 1:
                raise ValueError(
                    "adaptive observations must be matching per-example vectors"
                )
            if density is not None and density.shape != time.shape:
                raise ValueError("draw densities must match the observed times")
            finite = torch.isfinite(time) & torch.isfinite(loss)
            if density is not None:
                finite = finite & torch.isfinite(density) & (density > 0)
                density = density[finite].detach()
            self.record_time_loss_pair(
                time[finite].detach(), loss[finite].detach(), int(step), density=density
            )

    def training_time(self, time):
        return self.base.training_time(time)


class KeepRateAdapter(torch.nn.Module):
    """Expose scalar keep-rate components without changing the public schedule API."""

    def __init__(self, base):
        super().__init__()
        if getattr(base, "output_kind", None) != "keep_rate":
            raise TypeError("adaptive keep-rate schedule requires a keep-rate base")
        self.base = base

    def alpha_t(self, time):
        bounded = time.clamp(0, 1)
        keep, derivative = self.base(bounded)
        return keep + derivative * (time - bounded)

    def alpha_prime_t(self, time):
        return self.base(time)[1]

    def training_time(self, time):
        return getattr(self.base, "training_time", lambda t: t)(time)
