"""Optional model features and reusable architectural feedback injection."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelPrediction:
    logits: torch.Tensor | None
    hidden: torch.Tensor
    auxiliary_logits: torch.Tensor | None = None


class ConditioningProjection(nn.Module):
    def __init__(self, input_dim, hidden_size, initialization="zero"):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_size, bias=False)
        if initialization not in {"zero", "default"}:
            raise ValueError("conditioning initialization must be zero or default")
        if initialization == "zero":
            nn.init.zeros_(self.projection.weight)

    def forward(self, hidden, conditioning):
        if conditioning is None:
            return hidden
        if conditioning.shape[:-1] != hidden.shape[:-1]:
            raise ValueError(
                "self-conditioning must match the batch and sequence dimensions"
            )
        if conditioning.shape[-1] != self.projection.in_features:
            raise ValueError(
                "self-conditioning feature dimension must match its input projection"
            )
        return hidden + self.projection(conditioning.to(hidden.dtype))


class HiddenConditioning(nn.Module):
    """Add normalized hidden feedback with an initially inactive affine scale."""

    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.zeros_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, hidden, conditioning):
        if conditioning is None:
            conditioning = torch.zeros_like(hidden)
        if conditioning.shape != hidden.shape:
            raise ValueError(
                "hidden self-conditioning must match the model input shape"
            )
        with torch.autocast(hidden.device.type, enabled=False):
            feedback = self.norm(conditioning.float())
        return hidden + feedback.to(hidden.dtype)


def make_conditioning(source, embedding_dim, hidden_size, initialization="zero"):
    if source == "hidden":
        return HiddenConditioning(hidden_size)
    if source == "posterior_mean":
        return ConditioningProjection(embedding_dim, hidden_size, initialization)
    raise ValueError("unknown self-conditioning source")
