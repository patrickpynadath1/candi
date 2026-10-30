"""Position-dependent fusion of already represented hybrid views."""

import math

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Sum view embeddings with configured weights at masked/revealed sites.

    ``weights`` has one ``{masked: float, revealed: float}`` entry per view.
    A weight of zero excludes a view; one includes it unchanged. The token IDs
    of ``discrete_view`` define the gate, independently of the other view types.
    Representation normalization and noise preconditioning remain outside fusion.
    """

    def __init__(self, mask_index, weights, discrete_view=0):
        super().__init__()
        self.mask_index = int(mask_index)
        self.discrete_view = int(discrete_view)
        if not weights or not 0 <= self.discrete_view < len(weights):
            raise ValueError("discrete_view must index a nonempty list of weights")
        if self.mask_index < 0:
            raise ValueError("mask_index must be nonnegative")
        self.weights = []
        for entry in weights:
            if set(entry) != {"masked", "revealed"}:
                raise ValueError("each fusion weight must define masked and revealed")
            pair = (float(entry["masked"]), float(entry["revealed"]))
            if not all(math.isfinite(weight) for weight in pair):
                raise ValueError("fusion weights must be finite")
            self.weights.append(pair)

    def forward(self, states, points, embeddings, views):
        del points, views
        if len(states) != len(self.weights) or len(embeddings) != len(self.weights):
            raise ValueError("one state and embedding are required per fusion weight")
        tokens = states[self.discrete_view]
        if (
            tokens.is_floating_point()
            or tokens.is_complex()
            or tokens.dtype == torch.bool
        ):
            raise ValueError("the fusion gate requires discrete token IDs")
        masked = tokens.eq(self.mask_index)[..., None]
        contributions = []
        for embedding, (masked_weight, revealed_weight) in zip(
            embeddings, self.weights
        ):
            if (
                embedding.shape[:-1] != tokens.shape
                or embedding.shape != embeddings[0].shape
            ):
                raise ValueError(
                    "fusion embeddings must share token layout and hidden size"
                )
            # Select zero explicitly: an excluded view must not leak nonfinite inputs.
            masked_value = (
                embedding * masked_weight
                if masked_weight
                else torch.zeros_like(embedding)
            )
            revealed_value = (
                embedding * revealed_weight
                if revealed_weight
                else torch.zeros_like(embedding)
            )
            contributions.append(torch.where(masked, masked_value, revealed_value))
        return torch.stack(contributions).sum(0)
