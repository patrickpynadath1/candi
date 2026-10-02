"""Unit-embedding Gaussian likelihood correction before posterior normalization."""

import torch
from .posterior import PosteriorParameterization


class GaussianPriorPosterior(PosteriorParameterization):
    def __init__(self, output_prior=True, **kwargs):
        super().__init__(**kwargs)
        self.output_prior = output_prior

    def _constrain(self, values, observed):
        # The Gaussian prior can yield large logits near the clean endpoint.
        # Fused log_softmax avoids cancellation in x - logsumexp(x).
        return self.constraints.condition(torch.log_softmax(values, dim=-1), observed)

    def corrected_logits(
        self, logits, state, alpha, sigma, embedding_table, detach_rows=None
    ):
        logits = logits.float()
        if not self.output_prior:
            return logits
        with torch.autocast(state.device.type, enabled=False):
            prior = state.float() @ embedding_table.float().T
            if detach_rows is not None:
                detached = state.float() @ embedding_table.detach().float().T
                prior = torch.where(detach_rows[:, None, None], detached, prior)
            scale = alpha.float() / sigma.float().square().clamp_min(1e-12)
            return logits + scale[:, None, None] * prior
