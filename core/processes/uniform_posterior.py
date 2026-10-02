"""Uniform replacement with reverse bridges mixed by a clean-token posterior."""

import torch
from torch.nn import functional as F

from .uniform import UniformProcess


class PosteriorUniformProcess(UniformProcess):
    """Consume p(x_0 | noisy views), rather than unconditioned plug-in weights."""

    prediction_kind = "posterior"

    def reverse_probabilities(
        self,
        state,
        posterior,
        keep_t,
        keep_s,
    ):

        if posterior.shape != (*state.shape, self.vocab_size):
            raise ValueError("posterior must have shape [batch, length, vocab_size]")
        if keep_t.shape != state.shape[:1] or keep_s.shape != keep_t.shape:
            raise ValueError("keep rates must have shape [batch]")
        if not torch.isfinite(keep_t).all() or not torch.isfinite(keep_s).all():
            raise ValueError("keep rates must be finite")
        if torch.any((keep_t < 0) | (keep_s > 1) | (keep_s < keep_t)):
            raise ValueError("require 0 <= keep_t <= keep_s <= 1")
        # At identical times the bridge is identity, including the singular
        # clean endpoint. Evaluate a nonsingular dummy bridge for those rows.
        same = (keep_t == keep_s)[:, None, None]
        at = torch.where(same, 0, keep_t[:, None, None]).to(posterior)
        ass = torch.where(same, 1, keep_s[:, None, None]).to(posterior)
        observed = F.one_hot(state, self.vocab_size).to(posterior)
        likelihood_clean = at * observed + (1 - at) / self.vocab_size
        weighted = posterior / likelihood_clean
        ratio = at / ass
        likelihood_previous = ratio * observed + (1 - ratio) / self.vocab_size
        result = likelihood_previous * (
            ass * weighted
            + (1 - ass) / self.vocab_size * weighted.sum(-1, keepdim=True)
        )
        return torch.where(same, observed, result)
