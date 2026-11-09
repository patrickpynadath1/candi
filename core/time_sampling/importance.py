"""Learned piecewise-constant importance density over a view-local time.

The sampler is a coordinator clock for ``adaptive_time_views``: training draws
stratified uniform times ``t`` (``uniform``), the quantile map ``Q`` (``__call__``)
sends them to local times ``u = Q(t)`` distributed with density ``q(u)``, and
generation uses the same map on its uniform grid. ``proposal_density(u)`` is
``q(u)`` with gradients to the bin logits; objectives divide their Monte Carlo
integrand by its detached value. ``adaptation_loss`` is the score-function
surrogate whose gradient (and only its gradient) minimizes the estimator's second
moment, so the density logits train for variance, never for the loss value.
"""

import torch
from torch import nn


class StratifiedUniformTimeSampler:
    """One float64 draw per stratum, ``(i + U_i) / n``, in random row order."""

    def sample(self, count, device, **context):
        if count < 1:
            raise ValueError("stratified time sampling requires a nonempty batch")
        time = (
            torch.arange(count, device=device, dtype=torch.float64)
            + torch.rand(count, device=device, dtype=torch.float64)
        ) / count
        return time[torch.randperm(count, device=device)]


class PiecewiseImportanceTimeSampler(nn.Module):
    """Learned mixture of a piecewise-constant density and a uniform floor."""

    def __init__(self, bins=64, uniform_mix=0.01):
        super().__init__()
        if bins < 1 or not 0 < uniform_mix <= 1:
            raise ValueError("require positive bins and a mixture weight in (0, 1]")
        self.clock_logits = nn.Parameter(torch.zeros(bins, dtype=torch.float64))
        self.uniform_mix = float(uniform_mix)
        self.uniform = StratifiedUniformTimeSampler()

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

    def draw(self, count):
        """Local times and their proposal densities (one stratified batch)."""
        time = self.uniform.sample(count, self.clock_logits.device)
        value = self(time)
        return value, self.proposal_density(value)

    @staticmethod
    def variance_surrogate(estimate, density):
        """Zero-valued; its gradient is that of E_q[(f/q)^2] w.r.t. q."""
        score = -estimate.detach().square() * density.log()
        return score - score.detach()

    def adaptation_loss(self, estimate, density):
        return self.variance_surrogate(estimate, density)
