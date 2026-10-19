"""Standalone insertion terms and an adapter for aligned gap supervision."""

import torch
from torch import nn

from core.objectives.composition import Component, SumTerms


class GapCountLoss(nn.Module):
    def forward(self, expected, counts, mask, rate):
        # Poisson NLL without the target-only log-factorial constant.
        loss = expected - counts * expected.log()
        return (torch.where(mask, loss, 0) * rate[:, None]).sum(-1)


class GapTokenLoss(nn.Module):
    def forward(self, log_probs, targets, gap_indices, deleted, rate):
        if log_probs is None:
            return torch.zeros_like(rate)
        gap = gap_indices.clamp_max(log_probs.shape[1] - 1)
        rows = torch.arange(len(targets), device=targets.device)[:, None]
        loss = -log_probs[rows, gap, targets]
        return (torch.where(deleted, loss, 0) * rate[:, None]).sum(-1)


class InsertionObjective(nn.Module):
    """Align insertion inputs and preserve the existing reporting/reduction API.

    All loss mathematics lives in the configured components. Legacy weight
    arguments remain supported for saved experiment configs.
    """

    def __init__(self, count_weight=1.0, content_weight=1.0, count=None, content=None):
        super().__init__()
        self.combination = SumTerms(
            [
                count
                if count is not None
                else Component(
                    GapCountLoss(),
                    dict(
                        expected="expected", counts="counts", mask="mask", rate="rate"
                    ),
                    count_weight,
                ),
                content
                if content is not None
                else Component(
                    GapTokenLoss(),
                    dict(
                        log_probs="log_probs",
                        targets="targets",
                        gap_indices="gap_indices",
                        deleted="deleted",
                        rate="rate",
                    ),
                    content_weight,
                ),
            ]
        )
        available = {
            "expected",
            "counts",
            "mask",
            "rate",
            "log_probs",
            "targets",
            "gap_indices",
            "deleted",
        }
        for term in self.combination.terms:
            term.validate(available)

    @property
    def count_weight(self):
        return self.combination.terms[0].coefficient

    @count_weight.setter
    def count_weight(self, value):
        self.combination.terms[0].coefficient = value

    @property
    def content_weight(self):
        return self.combination.terms[1].coefficient

    @content_weight.setter
    def content_weight(self, value):
        self.combination.terms[1].coefficient = value

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
        if gap_log_probs is None and self.content_weight != 0:
            raise ValueError("Gap-token supervision requires gap predictions")
        total, (count, content) = self.combination(
            dict(
                expected=expected,
                counts=sample.gap_counts,
                mask=gap_mask,
                rate=rate,
                log_probs=gap_log_probs,
                targets=clean_tokens,
                gap_indices=sample.gap_for_token,
                deleted=sample.deleted,
            )
        )
        if reduction == "sum":
            return total.sum(), count.sum(), content.sum()
        return total, count, content
