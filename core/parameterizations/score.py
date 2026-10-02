"""SEDD absorbing log scores; these are not normalized token distributions."""

import math
import torch


class AbsorbingScoreParameterization:
    def __call__(self, logits, state, sigma):
        if torch.any(sigma <= 0):
            raise ValueError("score prediction requires positive noise")
        log_scale = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        scores = logits - log_scale[:, None, None] - math.log(logits.shape[-1] - 1)
        return scores.scatter(-1, state[..., None], torch.zeros_like(scores[..., :1]))
