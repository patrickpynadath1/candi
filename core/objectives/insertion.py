"""Losses for missing counts and clean content in insertion gaps."""

import math
import torch
from torch import nn


class InsertionObjective(nn.Module):
    def __init__(self, count_weight=1.0, content_weight=1.0):
        super().__init__()
        if any(not math.isfinite(w) or w < 0 for w in (count_weight, content_weight)):
            raise ValueError("insertion loss weights must be finite and nonnegative")
        self.count_weight, self.content_weight = count_weight, content_weight

    def forward(
        self,
        expected,
        gap_log_probs,
        sample,
        clean_tokens,
        gap_mask,
        rate,
        *,
        reduction="sum",
    ):
        if reduction not in {"sum", "none"}:
            raise ValueError("insertion reduction must be sum or none")
        # Poisson/Bregman loss; omitted target-only constant does not affect grads.
        count = expected - sample.gap_counts * expected.log()
        count = (torch.where(gap_mask, count, 0) * rate[:, None]).sum(-1)
        # Sparse equivalent of a per-gap bag-of-deleted-tokens target. Multiple
        # births in a gap use this same distribution; their order is approximate.
        if gap_log_probs is None:
            if self.content_weight != 0:
                raise ValueError("Gap-token supervision requires gap predictions")
            content = torch.zeros_like(count)
        else:
            gap = sample.gap_for_token.clamp_max(gap_log_probs.shape[1] - 1)
            rows = torch.arange(len(clean_tokens), device=clean_tokens.device)[:, None]
            content = -gap_log_probs[rows, gap, clean_tokens]
            content = (torch.where(sample.deleted, content, 0) * rate[:, None]).sum(-1)
        if reduction == "sum":
            count, content = count.sum(), content.sum()
        return self.count_weight * count + self.content_weight * content, count, content
