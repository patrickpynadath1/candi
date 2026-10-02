"""Learned small token geometry used by separable blurring diffusion."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .spec import RepresentationSpec


class CenteredEmbeddingRepresentation(nn.Module):
    def __init__(
        self,
        vocab_size,
        dimension=4,
        feature_lambda_min=0.25,
        feature_lambda_max=1e9,
        column_init_alpha=0.25,
    ):
        super().__init__()
        if vocab_size < 2 or dimension < 2:
            raise ValueError("vocabulary and embedding dimension must be at least two")
        if feature_lambda_min <= 0 or feature_lambda_max < feature_lambda_min:
            raise ValueError("feature rates must be positive and ordered")
        if column_init_alpha < 0:
            raise ValueError("column_init_alpha must be nonnegative")
        self.spec = RepresentationSpec(
            "centered_embedding", dimension, vocab_size, True
        )
        rates = torch.logspace(
            math.log10(feature_lambda_min), math.log10(feature_lambda_max), dimension
        )
        table = torch.randn(vocab_size, dimension) / dimension**0.5
        table *= (rates[0] / rates).pow(column_init_alpha)[None]
        self.table = nn.Parameter(table)
        self.project_(64)

    def codebook(self, *, detach=False):
        table = F.normalize(self.table, dim=-1)
        return table.detach() if detach else table

    def encode(self, values):
        table = self.codebook()
        if values.dtype.is_floating_point:
            if values.shape[-1] != self.spec.vocab_size:
                raise ValueError(
                    "probability blends must end in the vocabulary dimension"
                )
            return values.to(table.dtype) @ table
        return F.embedding(values, table)

    def posterior_mean(self, probabilities, *, detach=False):
        with torch.autocast(probabilities.device.type, enabled=False):
            return probabilities.float() @ self.codebook(detach=detach).float()

    def decode(self, state):
        return (state.float() @ self.codebook(detach=True).float().T).argmax(-1)

    @torch.no_grad()
    def project_(self, iterations=4):
        for _ in range(iterations):
            self.table.sub_(self.table.mean(0, keepdim=True))
            self.table.div_(self.table.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        self.table.sub_(self.table.mean(0, keepdim=True))

    @torch.no_grad()
    def post_optimizer_step(self):
        self.project_()
