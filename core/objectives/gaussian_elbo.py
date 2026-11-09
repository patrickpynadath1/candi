"""Likelihood terms over learned embeddings and the allocated Gaussian ELBO.

Shared terms (every term is tokenwise FP64):

- ``token_cross_entropy``: -log p(x_l);
- ``gaussian_diffusion_weight``: -1/2 dSNR/dt from a view's GaussianPath, the
  continuous-time weight of the embedding MSE;
- ``gaussian_prior_kl``: KL(q(z_1 | x) || N(0, I)) at the schedule's terminal time.

``GaussianELBO`` assembles them for an estimator allocation (see
core/time_sampling/adaptive_allocation.py): reconstruction rows estimate the
t = 0 term, diffusion rows the integral, each scaled by its share of the batch.
"""

import torch

from core.modifiers.self_conditioning.gradient_routing import detach_selected, embedding_mean
from .weighting import ObjectiveResult


def token_cross_entropy(log_probs, tokens):
    return -log_probs.gather(-1, tokens[..., None]).squeeze(-1).double()


def gaussian_diffusion_weight(path):
    """-1/2 dSNR/dt per example, from alpha, sigma and their time derivatives."""
    alpha, sigma = path.alpha.double(), path.sigma.double()
    snr_derivative = (
        2 * alpha * (path.dalpha_dt.double() * sigma - alpha * path.dsigma_dt.double())
        / sigma**3
    )
    return -0.5 * snr_derivative


def gaussian_prior_kl(schedule, clean):
    """Tokenwise KL of the terminal Gaussian marginal from the standard normal."""
    path = schedule(torch.ones(1, device=clean.device, dtype=torch.float64))
    alpha_sq, sigma_sq = path.alpha[0].double().square(), path.sigma[0].double().square()
    return 0.5 * (
        alpha_sq * clean.double().square().sum(-1)
        + clean.shape[-1] * (sigma_sq - 1 - sigma_sq.log())
    )


def _context_view(context, view):
    if context is None:
        raise ValueError("this objective requires ObjectiveContext")
    return (
        context.representations[view],
        context.coefficients[view],
        context.schedules[view],
    )


class GaussianELBO:
    """Prior KL + reconstruction CE (t = 0 rows) + weighted embedding MSE.

    Requires an allocated time draw (``reconstruction`` rows); denoiser-only rows
    (``context.detached_rows``) use detached embeddings for the MSE, while the
    prior keeps its ordinary gradient path.
    """

    requires_context = True

    def __init__(self, view=0):
        self.view = view

    def __call__(self, log_probs, tokens, *, context=None):
        representation, path, schedule = _context_view(context, self.view)
        rows = getattr(context.time_draw, "reconstruction", None)
        if rows is None:
            raise ValueError(
                "GaussianELBO requires an allocated time draw (reconstruction rows)"
            )
        batch, recon = len(tokens), int(rows.sum())
        if not 0 <= recon < batch:
            raise ValueError("allocation must include at least one diffusion example")
        detached = context.detached_rows
        table = representation.codebook()
        clean = representation.encode(tokens)
        target = clean if detached is None else detach_selected(clean, detached)
        predicted = embedding_mean(log_probs.exp(), table, detached)
        reconstruction = token_cross_entropy(log_probs, tokens)
        diffusion = gaussian_diffusion_weight(path)[:, None] * (
            (predicted - target).double().square().sum(-1)
        )
        prior = gaussian_prior_kl(schedule, clean)
        total = prior + torch.where(
            rows[:, None],
            reconstruction * (batch / recon if recon else 0.0),
            diffusion * (batch / (batch - recon)),
        )
        return ObjectiveResult(
            total,
            terms={"prior": prior, "reconstruction": reconstruction, "diffusion": diffusion},
        )
