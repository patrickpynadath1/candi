"""Learned time maps: a monotone map ``u = Q(t)`` from a uniform clock to path time.

A learned time map is part of what a time means, so it belongs to the information
path next to the noise schedule (config group ``information_path/time_map``,
placed at ``algorithm.time_map``), not to the training time sampler. Training and
generation both read it:

- with ``algorithm.adaptive_time_views``, the listed views evaluate their schedules
  at ``Q(t)`` of the coordinator time ``t``, in training and generation;
- in simplex diffusion, the path's training times are ``Q(t)`` and the sampler's
  ``time_grid=adaptive`` is ``Q`` applied to a uniform grid.

The training time sampler of such a recipe draws ``t`` uniformly; the mapped time
``u = Q(t)`` then has density ``time_density(u)`` relative to uniform, which the
algorithm combines with the sampler's own draw density (the contract in
``core/time_sampling/density.py``) wherever a consumer needs the density of ``u``.

Every map here exposes ``forward(t)`` (the map, detached), ``time_density(u)``
and how it learns: ``observe(times, token_loss, mask, step)`` for the loss-CDF
fitters (training-loss feedback, optional ``post_optimizer_step``), or
``proposal_density``/``adaptation_loss`` for the learned importance density
(a variance surrogate added to the training loss).
"""

import math

import torch
from torch import nn


def interpolate(x, xp, fp):
    index = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, len(xp) - 1)
    lo, hi = xp[index - 1], xp[index]
    weight = ((x - lo) / (hi - lo).clamp_min(1e-15)).clamp(0, 1)
    return fp[index - 1] + weight * (fp[index] - fp[index - 1])


class CDCDTimeMap(nn.Module):
    """CDCD Appendix A.1--A.4 learned piecewise-linear loss CDF as a time map.

    https://arxiv.org/abs/2211.15089
    Both axes have learned bin widths. The unnormalised output heights fit sequence
    cross entropy with inverse proposal-density weighted MSE. EMA logits drive the
    map. Positive bin pseudocounts are numerical safeguards, NOT the optional
    uniform mixture from A.6 (which is deliberately not implemented).

    Fitting state is buffers, separate from denoiser parameters/EMA/AdamW. Gradients
    are accumulated across microbatches and ranks, then a separate Adam update is
    committed by TrainingModule's post_optimizer_step hook. Checkpoints include all
    fitting and pending-gradient state.

    ``forward(t)`` is ``epsilon + (1 - epsilon) F^-1((t - epsilon) / (1 - epsilon))``
    once fitted (the identity before), so uniform draws on ``[epsilon, 1)`` map to
    CDCD's warped times.
    """

    def __init__(self, adaptive=True, bins=100, min_bin_size=1e-4,
                 learning_rate=1e-3, momentum=0.99, warmup=1000,
                 refit_every=1, epsilon=1e-5):
        super().__init__()
        if (not isinstance(bins, int) or bins < 2 or
                not math.isfinite(min_bin_size) or not 0 < min_bin_size < 1 / bins or
                not math.isfinite(learning_rate) or learning_rate <= 0 or
                not 0 <= momentum < 1 or warmup < 0 or refit_every < 1 or
                not 0 < epsilon < 1):
            raise ValueError('invalid CDCD time map settings')
        self.adaptive, self.bins = adaptive, bins
        self.min_bin_size, self.learning_rate = min_bin_size, learning_rate
        self.momentum, self.warmup, self.refit_every = momentum, warmup, refit_every
        self.epsilon = epsilon
        initial = torch.full((2, bins), -math.log(bins))
        for name, value in {
            'logits': initial, 'ema_logits': initial.clone(),
            'adam_first': torch.zeros_like(initial),
            'adam_second': torch.zeros_like(initial),
            'pending_gradient': torch.zeros_like(initial),
            'pending_count': torch.tensor(0., dtype=torch.float32),
            'count': torch.tensor(0, dtype=torch.long),
            'updates': torch.tensor(0, dtype=torch.long),
            'last_step': torch.tensor(-1, dtype=torch.long),
            'ready': torch.tensor(False),
        }.items():
            self.register_buffer(name, value)

    def _widths(self, logits):
        # Adding the same pseudocount to BOTH axes does not mix F(t) with t.
        widths = logits.float().softmax(-1) + self.min_bin_size
        return widths / widths.sum(-1, keepdim=True)

    @staticmethod
    def _piecewise(x, widths_in, widths_out):
        left_in = torch.cat((widths_in.new_zeros(1), widths_in.cumsum(0)[:-1]))
        left_out = torch.cat((widths_out.new_zeros(1), widths_out.cumsum(0)[:-1]))
        index = torch.searchsorted(left_in.detach().contiguous(), x.contiguous(), right=True)
        index = (index - 1).clamp(0, len(widths_in) - 1)
        fraction = ((x - left_in[index]) / widths_in[index]).clamp(0, 1)
        value = left_out[index] + fraction * widths_out[index]
        slope = widths_out[index] / widths_in[index]
        return value, slope

    def _unit_time(self, times):
        return ((times.float() - self.epsilon) / (1 - self.epsilon)).clamp(0, 1)

    def cdf(self, times):
        widths = self._widths(self.ema_logits)
        return self._piecewise(self._unit_time(times), widths[0], widths[1])[0]

    def density(self, times):
        widths = self._widths(self.ema_logits)
        return self._piecewise(self._unit_time(times), widths[0], widths[1])[1] / (1 - self.epsilon)

    def time_density(self, times):
        """Density of mapped uniform times relative to uniform (1 until fitted).

        Mapped times are ``epsilon + (1 - epsilon) F^-1(u)``, so this is the CDF
        slope ``F'`` at the unit time (``density`` is the same slope per unit of
        physical time, including the ``1 / (1 - epsilon)`` floor factor)."""
        if not self.adaptive or not bool(self.ready):
            return torch.ones_like(times)
        widths = self._widths(self.ema_logits)
        slope = self._piecewise(self._unit_time(times), widths[0], widths[1])[1]
        return slope.to(times.dtype)

    @torch.no_grad()
    def inverse_cdf(self, probabilities):
        widths = self._widths(self.ema_logits)
        x = self._piecewise(probabilities.float().clamp(0, 1), widths[1], widths[0])[0]
        # Normalized float32 bin widths can accumulate slightly past one.
        # Bound the physical time after interpolation/rescaling; downstream
        # schedules correctly reject even a one-ULP excursion outside [0, 1].
        return (self.epsilon + (1 - self.epsilon) * x).clamp(self.epsilon, 1.0)

    @torch.no_grad()
    def forward(self, uniform_times):
        if not self.adaptive or not bool(self.ready):
            return uniform_times
        # Epsilon bounds physical time, not CDF probability: clipping u would
        # once again remove low-mass regions from the sampling support.
        return self.inverse_cdf(self._unit_time(uniform_times))

    def fitting_loss(self, logits, times, losses, proposal_density):
        """Per-observation weighted MSE; no gradients to observations/proposal."""
        widths = self._widths(logits)
        fitted, _ = self._piecewise(self._unit_time(times.detach()), widths[0], widths[1])
        fitted = fitted * logits[1].float().exp().sum()
        return (fitted - losses.detach()).square() / proposal_density.detach()

    @torch.no_grad()
    def observe(self, times, token_loss, mask, step):
        if not self.adaptive:
            return
        weights = torch.ones_like(token_loss) if mask is None else mask.to(token_loss)
        lengths = weights.sum(-1)
        losses = (torch.where(weights > 0, token_loss.detach(), 0) * weights).sum(-1) / lengths.clamp_min(1)
        valid = ((lengths > 0) & torch.isfinite(losses) & torch.isfinite(times)
                 & (times >= self.epsilon) & (times <= 1))
        times, losses = times.detach()[valid].float(), losses[valid].float()
        # Map logits stay fixed until post_optimizer_step, so this is the
        # actual proposal used for every microbatch in the full global batch.
        proposal = (self.density(times) if bool(self.ready) else
                    torch.full_like(times, 1 / (1 - self.epsilon)))
        gradient = torch.zeros_like(self.logits)
        if len(times):
            with torch.enable_grad(), torch.autocast(times.device.type, enabled=False):
                logits = self.logits.detach().clone().requires_grad_(True)
                objective = self.fitting_loss(logits, times, losses, proposal).sum()
                gradient = torch.autograd.grad(objective, logits)[0].detach()
        packed = torch.cat((gradient.flatten(), gradient.new_tensor([len(times)])))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed)
        if not torch.isfinite(packed).all():
            raise FloatingPointError('nonfinite CDCD CDF fitting gradient')
        self.pending_gradient.add_(packed[:-1].reshape_as(gradient))
        self.pending_count.add_(packed[-1])
        self.count.add_(packed[-1].long())
        self.last_step.fill_(step)

    @torch.no_grad()
    def post_optimizer_step(self):
        if (not self.adaptive or self.pending_count <= 0 or
                (int(self.last_step) + 1) % self.refit_every):
            return
        gradient = self.pending_gradient / self.pending_count
        # A dedicated Adam, persisted as buffers so the denoiser optimizer and
        # model EMA never update these logits or change their parameter ordering.
        self.updates.add_(1)
        n = int(self.updates)
        self.adam_first.lerp_(gradient, 0.1)
        self.adam_second.lerp_(gradient.square(), 0.001)
        update = (self.adam_first / (1 - 0.9 ** n)) / (
            (self.adam_second / (1 - 0.999 ** n)).sqrt() + 1e-8)
        self.logits.add_(update, alpha=-self.learning_rate)
        self.ema_logits.lerp_(self.logits, 1 - self.momentum)
        self.pending_gradient.zero_()
        self.pending_count.zero_()
        # Warmup belongs to the fitter, not the denoiser global step: an
        # explicitly reset map on a resumed run needs its own bootstrap.
        self.ready.fill_(n >= self.warmup)


class LossDerivativeTimeMap(nn.Module):
    """Bucketed loss-derivative CDF (SDM Appendix J.1) as a time map.

    ``forward(t)`` inverts the fitted CDF of the loss derivative, so uniform ``t``
    maps to times concentrated where the loss changes fastest. With
    ``adaptive=false`` the map is the identity and keeps no state.
    """

    def __init__(
        self,
        adaptive=True,
        buffer_size=65536,
        bins=50,
        grid_size=1000,
        warmup=1000,
        refit_every=50,
        momentum=0.9,
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
            raise ValueError("invalid loss-derivative time map settings")
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

    def inverse_cdf(self, uniform):
        return interpolate(uniform, self.cdf.to(uniform), self.grid.to(uniform))

    def forward(self, uniform):
        return self.inverse_cdf(uniform) if self.adaptive else uniform

    def time_density(self, time):
        """Density of mapped times: the slope of the piecewise-linear CDF that
        ``forward`` inverts, constant on each grid cell (1 when not adaptive)."""
        if not self.adaptive:
            return torch.ones_like(time)
        grid, cdf = self.grid.to(time), self.cdf.to(time)
        slopes = torch.diff(cdf) / torch.diff(grid).clamp_min(1e-15)
        cell = torch.searchsorted(grid.contiguous(), time.contiguous(), right=True) - 1
        return slopes[cell.clamp(0, len(slopes) - 1)]

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


class PiecewiseImportanceTimeMap(nn.Module):
    """Learned piecewise-constant importance density ``q(u)`` and its quantile map.

    A mixture of a piecewise-constant density on ``bins`` equal cells and a
    ``uniform_mix`` floor. ``forward(t) = Q(t)`` (detached) sends uniform ``t`` to
    ``u`` distributed with density ``q(u)``. ``proposal_density(u)`` is ``q(u)``
    with gradients to the bin logits; objectives divide their Monte Carlo
    integrand by its detached value. ``adaptation_loss`` is the score-function
    surrogate whose gradient (and only its gradient) minimizes the estimator's
    second moment, so the logits train for variance, never for the loss value.
    The logits are ordinary parameters (optimizer groups, EMA, DDP).
    """

    def __init__(self, bins=64, uniform_mix=0.01):
        super().__init__()
        if bins < 1 or not 0 < uniform_mix <= 1:
            raise ValueError("require positive bins and a mixture weight in (0, 1]")
        self.clock_logits = nn.Parameter(torch.zeros(bins, dtype=torch.float64))
        self.uniform_mix = float(uniform_mix)

    def probabilities(self):
        bins = self.clock_logits.numel()
        return (1 - self.uniform_mix) * self.clock_logits.softmax(0) + self.uniform_mix / bins

    def inverse_cdf(self, time):
        """Local time ``Q(t)`` and its derivative ``dQ/dt``; exact at 0 and 1."""
        if not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("clock time must lie in [0,1]")
        p = self.probabilities()
        cdf = p.cumsum(0)
        index = torch.searchsorted(cdf.detach(), time.to(p).contiguous()).clamp_max(p.numel() - 1)
        left = torch.cat((p.new_zeros(1), cdf[:-1]))[index]
        value = ((index + (time.to(p) - left) / p[index]) / p.numel()).clamp(0, 1)
        value = torch.where(time == 0, 0.0, torch.where(time == 1, 1.0, value))
        return value, 1 / (p.numel() * p[index])

    def forward(self, time):
        # The map only places times; its logits learn from adaptation_loss.
        with torch.no_grad():
            return self.inverse_cdf(time)[0]

    def proposal_density(self, value):
        p = self.probabilities()
        index = (value * p.numel()).long().clamp_max(p.numel() - 1)
        return p.numel() * p[index]

    def time_density(self, value):
        """Detached density ``q(u)`` of mapped uniform times."""
        return self.proposal_density(value).detach()

    @staticmethod
    def variance_surrogate(estimate, density):
        """Zero-valued; its gradient is that of E_q[(f/q)^2] w.r.t. q."""
        score = -estimate.detach().square() * density.log()
        return score - score.detach()

    def adaptation_loss(self, estimate, density):
        return self.variance_surrogate(estimate, density)
