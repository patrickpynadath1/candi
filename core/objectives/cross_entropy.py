"""Clean-token cross entropy with optional smoothing and independent weighting."""

import math

import torch
from torch import nn


class CrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.0, weighting=None):
        super().__init__()
        if not math.isfinite(label_smoothing) or not 0 <= label_smoothing <= 1:
            raise ValueError("label_smoothing must be between zero and one")
        if weighting is not None and not isinstance(weighting, nn.Module):
            raise TypeError("loss weighting must be an nn.Module")
        self.label_smoothing = float(label_smoothing)
        self.weighting = weighting

    @property
    def requires_context(self):
        return self.weighting is not None and self.weighting.requires_context

    def forward(self, log_probs, targets, *, context=None):
        ce = -log_probs.gather(-1, targets[..., None]).squeeze(-1)
        if self.label_smoothing:
            if not torch.isfinite(log_probs).all():
                raise ValueError(
                    "label smoothing requires finite full-vocabulary log probabilities; "
                    "constrained predictions with excluded tokens are unsupported"
                )
            ce = (
                1 - self.label_smoothing
            ) * ce - self.label_smoothing * log_probs.mean(-1)
        return ce if self.weighting is None else self.weighting(ce, context)
