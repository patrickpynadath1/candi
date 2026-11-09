"""Conditional reveal/Gaussian ELBO with a forced-target proposal.

Views: a Gaussian content view and an absorbing reveal view sharing one local
time ``u``, with keep rate ``k(u)`` and Gaussian path coefficients. Under the
forced-target proposal (core/processes/forced_mask.py) a row with masked set ``m``
estimates

    (1 / |m|) * sum_{l in m} [-k' CE_l + (1 - k) w MSE_l] / q(u),

with ``w = -1/2 dSNR/du`` and ``q`` the importance density of the local time
(``context.time_density``; 1 without one), plus the terminal Gaussian prior KL on
every target. ``estimate`` is that per-row value, for the importance sampler's
variance surrogate. Requires binary loss masks and ``corruption_mask == loss_mask``.
"""
import torch

from .gaussian_elbo import gaussian_diffusion_weight, gaussian_prior_kl, token_cross_entropy
from .weighting import ObjectiveResult


class HybridELBO:
    requires_context = True
    supports_sequence_layout = False

    def __init__(self, gaussian_view=0, reveal_view=1):
        self.gaussian_view, self.reveal_view = gaussian_view, reveal_view

    def __call__(self, log_probs, tokens, *, context=None):
        if context is None or context.loss_mask is None:
            raise ValueError("hybrid ELBO requires ObjectiveContext with masks")
        loss_mask = context.loss_mask
        if not torch.all((loss_mask == 0) | (loss_mask == 1)):
            raise ValueError("hybrid ELBO requires binary loss masks")
        valid = loss_mask.bool()
        corruption = context.corruption_mask
        if corruption is not None and not torch.equal(corruption.bool(), valid):
            raise ValueError("hybrid conditional ELBO requires corruption_mask == loss_mask")
        g, r = self.gaussian_view, self.reveal_view
        representation, path = context.representations[g], context.coefficients[g]
        keep, keep_derivative = context.coefficients[r]
        _, known = context.processes[r].observed_tokens(context.states[r])
        masked = ~known & valid
        density = context.time_density
        density = torch.ones_like(keep) if density is None else density.detach()
        ce = token_cross_entropy(log_probs, tokens)
        clean = representation.encode(tokens)
        predicted = representation.posterior_mean(log_probs.exp())
        mse = (predicted - clean).double().square().sum(-1)
        continuous = ((1 - keep) * gaussian_diffusion_weight(path))[:, None] * mse
        integrand = ((-keep_derivative)[:, None] * ce + continuous) * masked
        count = valid.sum(-1)
        masked_count = masked.sum(-1).clamp_min(1)
        estimate = integrand.sum(-1) / masked_count / density
        interior = integrand * (count / masked_count / density)[:, None]
        prior = gaussian_prior_kl(context.schedules[g], clean)
        return ObjectiveResult((interior + prior) * valid, estimate=estimate)
